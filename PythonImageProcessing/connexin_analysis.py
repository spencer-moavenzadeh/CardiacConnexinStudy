import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # Example: Setting limit to 2^40 pixels
import cv2 # Import cv2 AFTER setting the environment variable
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform
from skimage import filters, measure, morphology, segmentation, feature
from skimage.color import rgb2gray, separate_stains, rgb2hed
from sklearn.cluster import DBSCAN
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ConnexinAnalyzer:
    """
    Comprehensive analyzer for connexin plaques in H&E + DAB stained cardiac histology slides
    """
    
    def __init__(self, pixel_size_um=0.25):
        """
        Initialize analyzer
        
        Args:
            pixel_size_um: Pixel size in micrometers for area calculations
        """
        self.pixel_size_um = pixel_size_um
        self.pixel_area_um2 = pixel_size_um ** 2
        
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load TIFF image and perform initial preprocessing
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def separate_stains_hed(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Separate H&E and DAB stains using color deconvolution
        """
        # Use HED color deconvolution (Hematoxylin-Eosin-DAB)
        hed = rgb2hed(image)
        
        # Extract individual stains
        hematoxylin = hed[:, :, 0]  # Nuclei (blue/purple)
        eosin = hed[:, :, 1]        # Cytoplasm (pink)
        dab = hed[:, :, 2]          # Connexins (brown)
        
        return hematoxylin, eosin, dab
    
    def segment_nuclei(self, hematoxylin: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment nuclei from hematoxylin channel
        """
        # Threshold hematoxylin channel
        threshold = filters.threshold_otsu(hematoxylin)
        nuclei_binary = hematoxylin > threshold
        
        # Clean up binary image
        nuclei_binary = morphology.remove_small_objects(nuclei_binary, min_size=50)
        nuclei_binary = morphology.remove_small_holes(nuclei_binary, area_threshold=30)
        
        # Distance transform and watershed for separation
        distance = ndimage.distance_transform_edt(nuclei_binary)
        local_maxima = feature.peak_local_maxima(distance, min_distance=10, threshold_abs=2)
        markers = np.zeros_like(nuclei_binary, dtype=int)
        markers[local_maxima] = np.arange(1, len(local_maxima[0]) + 1)
        
        # Watershed segmentation
        nuclei_labeled = segmentation.watershed(-distance, markers, mask=nuclei_binary)
        
        # Extract nuclei properties
        nuclei_props = []
        for region in measure.regionprops(nuclei_labeled):
            if region.area > 20:  # Filter very small objects
                nuclei_props.append({
                    'label': region.label,
                    'centroid': region.centroid,
                    'area': region.area * self.pixel_area_um2,
                    'eccentricity': region.eccentricity,
                    'orientation': region.orientation,
                    'bbox': region.bbox
                })
        
        return nuclei_labeled, nuclei_props
    
    def segment_connexins(self, dab: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment connexin plaques from DAB channel
        """
        # Threshold DAB channel for connexins
        threshold = filters.threshold_otsu(dab)
        connexin_binary = dab > threshold
        
        # Morphological operations to clean up
        connexin_binary = morphology.opening(connexin_binary, morphology.disk(2))
        connexin_binary = morphology.remove_small_objects(connexin_binary, min_size=10)
        
        # Label connected components
        connexin_labeled = measure.label(connexin_binary)
        
        # Extract connexin properties
        connexin_props = []
        for region in measure.regionprops(connexin_labeled, intensity_image=dab):
            if region.area > 5:  # Filter very small objects
                connexin_props.append({
                    'label': region.label,
                    'centroid': region.centroid,
                    'area': region.area * self.pixel_area_um2,
                    'perimeter': region.perimeter * self.pixel_size_um,
                    'eccentricity': region.eccentricity,
                    'orientation': region.orientation,
                    'mean_intensity': region.mean_intensity,
                    'bbox': region.bbox,
                    'coords': region.coords
                })
        
        return connexin_labeled, connexin_props
    
    def estimate_fiber_orientation(self, image: np.ndarray, window_size: int = 64) -> np.ndarray:
        """
        Estimate local fiber orientation using structure tensor
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Compute gradients
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)
        
        # Structure tensor components
        Ixx = grad_x ** 2
        Iyy = grad_y ** 2
        Ixy = grad_x * grad_y
        
        # Gaussian smoothing
        sigma = window_size / 6
        Ixx = filters.gaussian(Ixx, sigma=sigma)
        Iyy = filters.gaussian(Iyy, sigma=sigma)
        Ixy = filters.gaussian(Ixy, sigma=sigma)
        
        # Compute orientation
        orientation = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        
        return orientation
    
    def classify_connexin_location(self, connexin_props: List[Dict], nuclei_props: List[Dict], 
                                 fiber_orientation: np.ndarray) -> List[Dict]:
        """
        Classify connexins as lateralized or at intercalated discs
        """
        # Create spatial index for nuclei
        nuclei_centroids = np.array([n['centroid'] for n in nuclei_props])
        
        for i, conn in enumerate(connexin_props):
            conn_centroid = np.array(conn['centroid'])
            
            # Find nearest nuclei
            if len(nuclei_centroids) > 0:
                distances = np.sqrt(np.sum((nuclei_centroids - conn_centroid)**2, axis=1))
                nearest_nuclei_idx = np.argsort(distances)[:3]  # Consider 3 nearest nuclei
                
                # Get local fiber orientation at connexin location
                y, x = int(conn_centroid[0]), int(conn_centroid[1])
                if 0 <= y < fiber_orientation.shape[0] and 0 <= x < fiber_orientation.shape[1]:
                    local_fiber_angle = fiber_orientation[y, x]
                    
                    # Compare connexin orientation with fiber orientation
                    angle_diff = abs(conn['orientation'] - local_fiber_angle)
                    angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wrapping
                    
                    # Classification criteria
                    # Intercalated discs: perpendicular to fiber direction, between cell ends
                    # Lateral: parallel to fiber direction, along cell sides
                    
                    if angle_diff > np.pi/3:  # ~60 degrees
                        location_type = 'intercalated_disc'
                    else:
                        location_type = 'lateral'
                        
                    # Additional check: distance to nearest nuclei
                    min_distance = distances[nearest_nuclei_idx[0]] * self.pixel_size_um
                    
                    if min_distance > 50:  # Far from nuclei, likely intercalated
                        location_type = 'intercalated_disc'
                    
                else:
                    location_type = 'unknown'
                    angle_diff = np.nan
                    min_distance = np.nan
            else:
                location_type = 'unknown'
                angle_diff = np.nan
                min_distance = np.nan
            
            # Update connexin properties
            connexin_props[i]['location_type'] = location_type
            connexin_props[i]['fiber_angle_diff'] = angle_diff
            connexin_props[i]['nearest_nucleus_distance'] = min_distance
        
        return connexin_props
    
    def calculate_statistics(self, connexin_props: List[Dict], nuclei_props: List[Dict], 
                           image_shape: Tuple[int, int]) -> Dict:
        """
        Calculate comprehensive statistics
        """
        if not connexin_props:
            return self._empty_stats()
        
        # Basic connexin statistics
        areas = [c['area'] for c in connexin_props]
        total_area = image_shape[0] * image_shape[1] * self.pixel_area_um2
        
        # Separate by location type
        lateral_props = [c for c in connexin_props if c['location_type'] == 'lateral']
        id_props = [c for c in connexin_props if c['location_type'] == 'intercalated_disc']
        
        lateral_areas = [c['area'] for c in lateral_props] if lateral_props else [0]
        id_areas = [c['area'] for c in id_props] if id_props else [0]
        
        stats = {
            # Overall connexin statistics
            'total_connexins': len(connexin_props),
            'total_connexin_area': sum(areas),
            'mean_connexin_area': np.mean(areas),
            'median_connexin_area': np.median(areas),
            'std_connexin_area': np.std(areas),
            'connexin_density_per_mm2': len(connexin_props) / (total_area / 1e6),
            'connexin_area_fraction': sum(areas) / total_area,
            
            # Cell-normalized statistics
            'total_nuclei': len(nuclei_props),
            'connexins_per_nucleus': len(connexin_props) / max(len(nuclei_props), 1),
            'connexin_area_per_nucleus': sum(areas) / max(len(nuclei_props), 1),
            
            # Lateral connexins
            'lateral_connexins': len(lateral_props),
            'lateral_connexin_area': sum(lateral_areas),
            'percent_lateral_connexins': len(lateral_props) / len(connexin_props) * 100,
            'percent_lateral_area': sum(lateral_areas) / sum(areas) * 100,
            'mean_lateral_area': np.mean(lateral_areas) if lateral_props else 0,
            
            # Intercalated disc connexins
            'id_connexins': len(id_props),
            'id_connexin_area': sum(id_areas),
            'percent_id_connexins': len(id_props) / len(connexin_props) * 100,
            'percent_id_area': sum(id_areas) / sum(areas) * 100,
            'mean_id_area': np.mean(id_areas) if id_props else 0,
            
            # Size distribution
            'small_connexins': len([a for a in areas if a < 1.0]),  # < 1 μm²
            'medium_connexins': len([a for a in areas if 1.0 <= a < 5.0]),  # 1-5 μm²
            'large_connexins': len([a for a in areas if a >= 5.0]),  # >= 5 μm²
        }
        
        return stats
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics dictionary"""
        return {key: 0 for key in [
            'total_connexins', 'total_connexin_area', 'mean_connexin_area',
            'median_connexin_area', 'std_connexin_area', 'connexin_density_per_mm2',
            'connexin_area_fraction', 'total_nuclei', 'connexins_per_nucleus',
            'connexin_area_per_nucleus', 'lateral_connexins', 'lateral_connexin_area',
            'percent_lateral_connexins', 'percent_lateral_area', 'mean_lateral_area',
            'id_connexins', 'id_connexin_area', 'percent_id_connexins',
            'percent_id_area', 'mean_id_area', 'small_connexins',
            'medium_connexins', 'large_connexins'
        ]}
    
    def create_overlay_image(self, image: np.ndarray, nuclei_labeled: np.ndarray, 
                           connexin_labeled: np.ndarray, connexin_props: List[Dict],
                           nuclei_props: List[Dict]) -> np.ndarray:
        """
        Create detailed overlay image for algorithm validation
        """
        # Create overlay on original image
        overlay = image.copy()
        
        # Create separate overlay channels
        height, width = image.shape[:2]
        nuclei_overlay = np.zeros((height, width, 3), dtype=np.float32)
        connexin_overlay = np.zeros((height, width, 3), dtype=np.float32)
        
        # Draw nuclei contours in cyan
        nuclei_contours = measure.find_contours(nuclei_labeled > 0, 0.5)
        for contour in nuclei_contours:
            coords = contour.astype(int)
            for i in range(len(coords)):
                y, x = coords[i]
                if 0 <= y < height and 0 <= x < width:
                    nuclei_overlay[y, x] = [0, 1, 1]  # Cyan
        
        # Draw nucleus centroids
        for nucleus in nuclei_props:
            y, x = int(nucleus['centroid'][0]), int(nucleus['centroid'][1])
            if 0 <= y < height and 0 <= x < width:
                # Draw small cross for centroid
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if (0 <= y+dy < height and 0 <= x+dx < width and 
                            (abs(dy) == 2 or abs(dx) == 2)):
                            nuclei_overlay[y+dy, x+dx] = [0, 1, 1]
        
        # Draw connexin plaques with color coding
        for conn in connexin_props:
            coords = conn['coords']
            
            # Color based on location type
            if conn['location_type'] == 'lateral':
                color = [1, 0, 0]  # Red
            elif conn['location_type'] == 'intercalated_disc':
                color = [0, 0, 1]  # Blue
            else:
                color = [0.5, 0.5, 0.5]  # Gray
            
            # Fill connexin pixels
            for coord in coords:
                y, x = coord[0], coord[1]
                if 0 <= y < height and 0 <= x < width:
                    connexin_overlay[y, x] = color
        
        # Combine overlays with original image
        final_overlay = overlay.copy()
        
        # Add nuclei overlay (cyan)
        nuclei_mask = np.any(nuclei_overlay > 0, axis=2)
        final_overlay[nuclei_mask] = (0.6 * overlay[nuclei_mask] + 
                                     0.4 * nuclei_overlay[nuclei_mask])
        
        # Add connexin overlay (colored by type)
        connexin_mask = np.any(connexin_overlay > 0, axis=2)
        final_overlay[connexin_mask] = (0.4 * overlay[connexin_mask] + 
                                       0.6 * connexin_overlay[connexin_mask])
        
        return final_overlay
    
    def create_validation_panels(self, image: np.ndarray, nuclei_labeled: np.ndarray, 
                               connexin_labeled: np.ndarray, connexin_props: List[Dict],
                               nuclei_props: List[Dict], fiber_orientation: np.ndarray) -> plt.Figure:
        """
        Create detailed validation panels for algorithm performance checking
        """
        fig, axes = plt.subplots(3, 3, figsize=(21, 18))
        
        # Panel 1: Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Panel 2: Nuclei segmentation overlay
        axes[0, 1].imshow(image, alpha=0.8)
        nuclei_contours = measure.find_contours(nuclei_labeled > 0, 0.5)
        for contour in nuclei_contours:
            axes[0, 1].plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=1.5)
        
        # Add nucleus centroids
        for nucleus in nuclei_props:
            y, x = nucleus['centroid']
            axes[0, 1].plot(x, y, 'c+', markersize=8, markeredgewidth=2)
        
        axes[0, 1].set_title(f'Nuclei Overlay (n={len(nuclei_props)})', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Panel 3: Connexin segmentation overlay
        axes[0, 2].imshow(image, alpha=0.8)
        
        # Draw connexin contours with color coding
        lateral_count = sum(1 for c in connexin_props if c['location_type'] == 'lateral')
        id_count = sum(1 for c in connexin_props if c['location_type'] == 'intercalated_disc')
        unknown_count = len(connexin_props) - lateral_count - id_count
        
        for conn in connexin_props:
            coords = conn['coords']
            if conn['location_type'] == 'lateral':
                color = 'red'
            elif conn['location_type'] == 'intercalated_disc':
                color = 'blue'
            else:
                color = 'gray'
            
            # Draw contour
            if len(coords) > 2:
                # Create binary mask for this connexin
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[coords[:, 0], coords[:, 1]] = True
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    axes[0, 2].plot(contour[:, 1], contour[:, 0], color, linewidth=1.5)
        
        axes[0, 2].set_title(f'Connexins: Red=Lateral({lateral_count}), Blue=ID({id_count})', 
                            fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Panel 4: Combined overlay
        combined_overlay = self.create_overlay_image(image, nuclei_labeled, connexin_labeled, 
                                                   connexin_props, nuclei_props)
        axes[1, 0].imshow(combined_overlay)
        axes[1, 0].set_title('Combined Overlay\n(Cyan=Nuclei, Red=Lateral, Blue=ID)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Panel 5: Individual stain channels
        hematoxylin, eosin, dab = self.separate_stains_hed(image)
        
        # Create composite of individual channels
        stain_composite = np.zeros_like(image)
        stain_composite[:, :, 0] = np.clip(dab, 0, 1)      # DAB in red channel
        stain_composite[:, :, 1] = np.clip(eosin, 0, 1)    # Eosin in green channel  
        stain_composite[:, :, 2] = np.clip(hematoxylin, 0, 1)  # Hematoxylin in blue channel
        
        axes[1, 1].imshow(stain_composite)
        axes[1, 1].set_title('Separated Stains\n(Red=DAB, Green=Eosin, Blue=H)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Panel 6: Fiber orientation with connexin overlay
        axes[1, 2].imshow(rgb2gray(image), cmap='gray', alpha=0.7)
        
        # Fiber orientation vectors (subsampled)
        y, x = np.mgrid[0:fiber_orientation.shape[0]:25, 0:fiber_orientation.shape[1]:25]
        u = np.cos(fiber_orientation[::25, ::25])
        v = np.sin(fiber_orientation[::25, ::25])
        axes[1, 2].quiver(x, y, u, v, scale=30, color='yellow', alpha=0.8, width=0.003)
        
        # Overlay connexins
        for conn in connexin_props:
            coords = conn['coords']
            color = 'red' if conn['location_type'] == 'lateral' else 'blue'
            axes[1, 2].scatter(coords[:, 1], coords[:, 0], c=color, s=2, alpha=0.8)
        
        axes[1, 2].set_title('Fiber Orientation + Connexins', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Panel 7: Size-coded connexins
        axes[2, 0].imshow(image, alpha=0.8)
        
        areas = [c['area'] for c in connexin_props]
        if areas:
            # Normalize sizes for visualization
            min_area, max_area = min(areas), max(areas)
            for conn in connexin_props:
                coords = conn['coords']
                # Size-based color (small=green, large=red)
                size_norm = (conn['area'] - min_area) / (max_area - min_area) if max_area > min_area else 0
                color = plt.cm.RdYlGn_r(size_norm)
                axes[2, 0].scatter(coords[:, 1], coords[:, 0], c=[color], s=3, alpha=0.8)
        
        axes[2, 0].set_title('Size-Coded Connexins\n(Green=Small, Red=Large)', 
                            fontsize=14, fontweight='bold')
        axes[2, 0].axis('off')
        
        # Panel 8: Statistics summary
        axes[2, 1].axis('off')
        
        # Calculate key statistics
        total_connexins = len(connexin_props)
        total_nuclei = len(nuclei_props)
        lateral_pct = (lateral_count / total_connexins * 100) if total_connexins > 0 else 0
        
        if areas:
            mean_area = np.mean(areas)
            median_area = np.median(areas)
            std_area = np.std(areas)
        else:
            mean_area = median_area = std_area = 0
        
        stats_text = f"""
SEGMENTATION RESULTS

Nuclei: {total_nuclei}
Total Connexins: {total_connexins}
Connexins/Nucleus: {total_connexins/max(total_nuclei,1):.2f}

LOCATION CLASSIFICATION
Lateral: {lateral_count} ({lateral_pct:.1f}%)
Intercalated Discs: {id_count} ({100-lateral_pct:.1f}%)
Unknown: {unknown_count}

SIZE STATISTICS (μm²)
Mean: {mean_area:.3f}
Median: {median_area:.3f}
Std Dev: {std_area:.3f}
        """
        
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[2, 1].set_title('Analysis Summary', fontsize=14, fontweight='bold')
        
        # Panel 9: Size distribution histogram
        if areas:
            axes[2, 2].hist(areas, bins=min(20, len(areas)), alpha=0.7, 
                           edgecolor='black', color='steelblue')
            axes[2, 2].axvline(mean_area, color='red', linestyle='--', 
                              label=f'Mean: {mean_area:.3f}')
            axes[2, 2].axvline(median_area, color='orange', linestyle='--', 
                              label=f'Median: {median_area:.3f}')
            axes[2, 2].set_xlabel('Connexin Area (μm²)')
            axes[2, 2].set_ylabel('Count')
            axes[2, 2].legend()
        
        axes[2, 2].set_title('Connexin Size Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_overlay_image(self, image: np.ndarray, nuclei_labeled: np.ndarray, 
                          connexin_labeled: np.ndarray, connexin_props: List[Dict],
                          nuclei_props: List[Dict], output_path: str, dpi: int = 300):
        """
        Save high-resolution overlay image for detailed inspection
        """
        overlay = self.create_overlay_image(image, nuclei_labeled, connexin_labeled, 
                                          connexin_props, nuclei_props)
        
        fig, ax = plt.subplots(1, 1, figsize=(image.shape[1]/100, image.shape[0]/100))
        ax.imshow(overlay)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='cyan', alpha=0.7, label='Nuclei'),
            Patch(facecolor='red', alpha=0.7, label='Lateral Connexins'),
            Patch(facecolor='blue', alpha=0.7, label='Intercalated Disc Connexins'),
            Patch(facecolor='gray', alpha=0.7, label='Unknown Connexins')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"High-resolution overlay saved to: {output_path}")
    
    def visualize_results(self, image: np.ndarray, nuclei_labeled: np.ndarray, 
                         connexin_labeled: np.ndarray, connexin_props: List[Dict],
                         nuclei_props: List[Dict], fiber_orientation: np.ndarray) -> plt.Figure:
        """
        Create comprehensive validation visualization
        """
        return self.create_validation_panels(image, nuclei_labeled, connexin_labeled, 
                                           connexin_props, nuclei_props, fiber_orientation)
    
    def analyze_slide(self, image_path: str, visualize: bool = True) -> Tuple[Dict, plt.Figure]:
        """
        Complete analysis pipeline for a single slide
        """
        print(f"Analyzing slide: {image_path}")
        
        # Load and preprocess
        image = self.load_and_preprocess(image_path)
        print("✓ Image loaded and preprocessed")
        
        # Separate stains
        hematoxylin, eosin, dab = self.separate_stains_hed(image)
        print("✓ Stains separated")
        
        # Segment nuclei
        nuclei_labeled, nuclei_props = self.segment_nuclei(hematoxylin)
        print(f"✓ Nuclei segmented: {len(nuclei_props)} nuclei found")
        
        # Segment connexins
        connexin_labeled, connexin_props = self.segment_connexins(dab)
        print(f"✓ Connexins segmented: {len(connexin_props)} connexins found")
        
        # Estimate fiber orientation
        fiber_orientation = self.estimate_fiber_orientation(image)
        print("✓ Fiber orientation estimated")
        
        # Classify connexin locations
        connexin_props = self.classify_connexin_location(connexin_props, nuclei_props, fiber_orientation)
        print("✓ Connexin locations classified")
        
        # Calculate statistics
        stats = self.calculate_statistics(connexin_props, nuclei_props, image.shape[:2])
        print("✓ Statistics calculated")
        
        # Create visualization
        fig = None
        if visualize:
            fig = self.visualize_results(image, nuclei_labeled, connexin_labeled, 
                                       connexin_props, nuclei_props, fiber_orientation)
            print("✓ Validation visualization created")
        
        return stats, fig

# Example usage
def analyze_multiple_slides(slide_paths: List[str], output_csv: str = None) -> pd.DataFrame:
    """
    Analyze multiple slides and compile results
    """
    analyzer = ConnexinAnalyzer()
    results = []
    
    for path in slide_paths:
        try:
            stats, _ = analyzer.analyze_slide(path, visualize=False)
            stats['slide_path'] = path
            stats['slide_name'] = path.split('/')[-1]  # Extract filename
            results.append(stats)
        except Exception as e:
            print(f"Error analyzing {path}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return df

# Enhanced usage example with validation overlays
def analyze_with_validation(image_path: str, output_dir: str = None):
    """
    Complete analysis with validation overlays
    """
    analyzer = ConnexinAnalyzer(pixel_size_um=0.25)
    
    # Run full analysis
    stats, validation_fig = analyzer.analyze_slide(image_path, visualize=True)
    
    # Load image for overlay creation
    image = analyzer.load_and_preprocess(image_path)
    hematoxylin, eosin, dab = analyzer.separate_stains_hed(image)
    nuclei_labeled, nuclei_props = analyzer.segment_nuclei(hematoxylin)
    connexin_labeled, connexin_props = analyzer.segment_connexins(dab)
    fiber_orientation = analyzer.estimate_fiber_orientation(image)
    connexin_props = analyzer.classify_connexin_location(connexin_props, nuclei_props, fiber_orientation)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save validation figure
        validation_fig.savefig(os.path.join(output_dir, 'validation_panels.png'), 
                              dpi=300, bbox_inches='tight')
        
        # Save high-resolution overlay
        analyzer.save_overlay_image(image, nuclei_labeled, connexin_labeled, 
                                  connexin_props, nuclei_props,
                                  os.path.join(output_dir, 'overlay_high_res.png'))
        
        # Save individual overlay images for detailed inspection
        create_individual_overlays(analyzer, image, nuclei_labeled, connexin_labeled, 
                                 connexin_props, nuclei_props, output_dir)
    
    return stats, validation_fig

def create_individual_overlays(analyzer, image, nuclei_labeled, connexin_labeled, 
                             connexin_props, nuclei_props, output_dir):
    """
    Create individual overlay images for detailed inspection
    """
    # 1. Nuclei-only overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image, alpha=0.8)
    nuclei_contours = measure.find_contours(nuclei_labeled > 0, 0.5)
    for contour in nuclei_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=2)
    for nucleus in nuclei_props:
        y, x = nucleus['centroid']
        ax.plot(x, y, 'c+', markersize=10, markeredgewidth=3)
        # Add nucleus ID
        ax.text(x+10, y, f"N{nucleus['label']}", color='white', fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor="cyan", alpha=0.7))
    ax.set_title(f'Nuclei Segmentation Validation (n={len(nuclei_props)})', fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'nuclei_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Connexin-only overlay with labels
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image, alpha=0.8)
    
    for i, conn in enumerate(connexin_props):
        coords = conn['coords']
        color = 'red' if conn['location_type'] == 'lateral' else 'blue' if conn['location_type'] == 'intercalated_disc' else 'gray'
        
        # Draw connexin outline
        if len(coords) > 2:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            contours = measure.find_contours(mask, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color, linewidth=2)
        
        # Add connexin centroid and label
        centroid_y, centroid_x = conn['centroid']
        ax.plot(centroid_x, centroid_y, 'k+', markersize=8, markeredgewidth=2)
        ax.text(centroid_x+10, centroid_y, f"C{i+1}\n{conn['area']:.2f}μm²", 
               color='white', fontsize=6, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
    ax.set_title(f'Connexin Segmentation Validation (n={len(connexin_props)})\nRed=Lateral, Blue=Intercalated Disc', 
                fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'connexin_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Problem detection overlay (edge cases)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image, alpha=0.8)
    
    # Highlight potential issues
    problem_connexins = []
    for conn in connexin_props:
        coords = conn['coords']
        
        # Flag very small or very large connexins
        if conn['area'] < 0.1 or conn['area'] > 20:
            problem_connexins.append((conn, 'size_issue', 'yellow'))
        # Flag connexins with unknown location
        elif conn['location_type'] == 'unknown':
            problem_connexins.append((conn, 'unknown_location', 'orange'))
        # Flag connexins very far from nuclei
        elif conn.get('nearest_nucleus_distance', 0) > 100:
            problem_connexins.append((conn, 'isolated', 'purple'))
    
    for conn, issue_type, color in problem_connexins:
        coords = conn['coords']
        ax.scatter(coords[:, 1], coords[:, 0], c=color, s=10, alpha=0.8)
        
        centroid_y, centroid_x = conn['centroid']
        ax.text(centroid_x+10, centroid_y, f"{issue_type}\n{conn['area']:.2f}μm²", 
               color='white', fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9))
    
    ax.set_title(f'Potential Issues Detection\nYellow=Size, Orange=Unknown Location, Purple=Isolated', 
                fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'problem_detection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual overlay images saved to {output_dir}")

# Usage example:
# import connexin_analysis
# analyzer = connexin_analysis.ConnexinAnalyzer(pixel_size_um=0.1659)  # Adjust pixel size as needed
# stats, fig = connexin_analysis.analyze_with_validation("path/to/your/slide.tiff", "output_validation/")
# print("Analysis complete with validation overlays!")
# plt.show()  # Display validation panels