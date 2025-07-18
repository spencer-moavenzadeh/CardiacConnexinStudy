import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, measure, morphology, color
from skimage.util import img_as_float
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
#warnings.filterwarnings('ignore')
import gc
import os
from tqdm import tqdm
import psutil
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class MegaSlideConnexinAnalyzer:
    """
    Ultra-optimized analyzer for massive histology slides (2.7GB+)
    Uses tile-based processing to avoid memory crashes
    """
    
    def __init__(self, pixel_size_um=0.25, tile_size=1024, overlap=128):
        """
        Initialize analyzer for massive slides
        
        Args:
            pixel_size_um: Pixel size in micrometers
            tile_size: Size of processing tiles (smaller = less memory)
            overlap: Overlap between tiles to avoid edge artifacts
        """
        self.pixel_size_um = pixel_size_um
        self.pixel_area_um2 = pixel_size_um ** 2
        self.tile_size = tile_size
        self.overlap = overlap
        self.memory_limit_gb = self._get_available_memory()
        
        print(f"🔧 Initialized for massive slides:")
        print(f"   Tile size: {tile_size}x{tile_size}")
        print(f"   Available memory: {self.memory_limit_gb:.1f} GB")
        print(f"   Overlap: {overlap} pixels")
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    def _check_memory_usage(self):
        """Monitor memory usage and force cleanup if needed"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:  # If using >85% memory
            print(f"  High memory usage ({memory_percent:.1f}%). Forcing cleanup...")
            gc.collect()
    
    def get_slide_info(self, image_path: str) -> Dict:
        """
        Get slide dimensions without loading the full image
        """
        with Image.open(image_path) as img:
            width, height = img.size
            
        file_size_gb = os.path.getsize(image_path) / (1024**3)
        
        info = {
            'width': width,
            'height': height,
            'file_size_gb': file_size_gb,
            'estimated_memory_gb': (width * height * 3 * 4) / (1024**3),  # RGB float32
            'requires_tiling': file_size_gb > 1.0 or width * height > 50_000_000
        }
        
        print(f"📊 Slide Information:")
        print(f"   Dimensions: {width:,} x {height:,} pixels")
        print(f"   File size: {file_size_gb:.2f} GB")
        print(f"   Estimated memory needed: {info['estimated_memory_gb']:.2f} GB")
        print(f"   Tiling required: {info['requires_tiling']}")
        
        return info
    
    def load_slide_tile(self, image_path: str, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Load a specific tile from the large slide
        """
        with Image.open(image_path) as img:
            # Crop the specific region
            tile = img.crop((x, y, x + width, y + height))
            # Convert to numpy array
            tile_array = np.array(tile)
            
        # Convert to RGB if needed and normalize
        if len(tile_array.shape) == 3 and tile_array.shape[2] == 3:
            tile_float = img_as_float(tile_array)
        else:
            raise ValueError("Image must be RGB")
            
        return tile_float
    
    def generate_tile_coordinates(self, slide_width: int, slide_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate coordinates for overlapping tiles
        """
        tiles = []
        effective_tile_size = self.tile_size - self.overlap
        
        for y in range(0, slide_height, effective_tile_size):
            for x in range(0, slide_width, effective_tile_size):
                # Calculate actual tile dimensions
                tile_width = min(self.tile_size, slide_width - x)
                tile_height = min(self.tile_size, slide_height - y)
                
                # Only process tiles that are large enough
                if tile_width > self.overlap and tile_height > self.overlap:
                    tiles.append((x, y, tile_width, tile_height))
        
        print(f" Generated {len(tiles)} tiles for processing")
        return tiles
    
    def separate_stains_fast(self, tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast stain separation optimized for memory
        """
        # Convert to optical density
        tile_od = -np.log(tile + 1e-6)
        
        # Simple but effective H&DAB separation
        # Hematoxylin: more blue channel
        hematoxylin = tile_od[:, :, 2] - 0.5 * (tile_od[:, :, 0] + tile_od[:, :, 1])
        
        # DAB: more red and green, less blue
        dab = 0.5 * (tile_od[:, :, 0] + tile_od[:, :, 1]) - tile_od[:, :, 2]
        
        # Normalize
        hematoxylin = np.clip(hematoxylin, 0, None)
        dab = np.clip(dab, 0, None)
        
        if hematoxylin.max() > 0:
            hematoxylin = hematoxylin / hematoxylin.max()
        if dab.max() > 0:
            dab = dab / dab.max()
        
        return hematoxylin, dab
    
    def process_tile_for_nuclei(self, tile: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Process a single tile for nuclei detection
        """
        x_offset, y_offset, tile_w, tile_h = tile_coords
        
        # Separate stains
        hematoxylin, _ = self.separate_stains_fast(tile)
        
        # Quick nuclei segmentation
        try:
            threshold = filters.threshold_otsu(hematoxylin)
        except:
            threshold = np.percentile(hematoxylin, 75)
        
        nuclei_binary = hematoxylin > threshold * 0.8  # Slightly lower threshold
        
        # Cleanup
        nuclei_binary = morphology.remove_small_objects(nuclei_binary, min_size=20)
        nuclei_binary = morphology.binary_closing(nuclei_binary, morphology.disk(2))
        
        # Label and extract properties
        nuclei_labeled = measure.label(nuclei_binary)
        
        nuclei_props = []
        for region in measure.regionprops(nuclei_labeled):
            if 10 < region.area < 800:  # Nucleus size filter
                # Convert coordinates to global coordinates
                global_centroid = (
                    region.centroid[0] + y_offset,
                    region.centroid[1] + x_offset
                )
                
                nuclei_props.append({
                    'label': region.label,
                    'centroid': global_centroid,
                    'area': region.area * self.pixel_area_um2,
                    'tile_origin': (x_offset, y_offset)
                })
        
        return nuclei_props
    
    def process_tile_for_connexins(self, tile: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Process a single tile for connexin detection
        """
        x_offset, y_offset, tile_w, tile_h = tile_coords
        
        # Separate stains
        _, dab = self.separate_stains_fast(tile)
        
        # Connexin segmentation
        try:
            threshold = filters.threshold_otsu(dab)
        except:
            threshold = np.percentile(dab, 85)
        
        connexin_binary = dab > threshold
        
        # Cleanup
        connexin_binary = morphology.opening(connexin_binary, morphology.disk(1))
        connexin_binary = morphology.remove_small_objects(connexin_binary, min_size=3)
        
        # Label and extract properties
        connexin_labeled = measure.label(connexin_binary)
        
        connexin_props = []
        for region in measure.regionprops(connexin_labeled, intensity_image=dab):
            if 2 < region.area < 300:  # Connexin size filter
                # Convert coordinates to global coordinates
                global_centroid = (
                    region.centroid[0] + y_offset,
                    region.centroid[1] + x_offset
                )
                
                connexin_props.append({
                    'label': region.label,
                    'centroid': global_centroid,
                    'area': region.area * self.pixel_area_um2,
                    'orientation': region.orientation,
                    'eccentricity': region.eccentricity,
                    'mean_intensity': region.mean_intensity,
                    'tile_origin': (x_offset, y_offset)
                })
        
        return connexin_props
    
    def estimate_fiber_orientation_tile(self, tile: np.ndarray) -> float:
        """
        Estimate average fiber orientation for a tile
        """
        gray = color.rgb2gray(tile)
        
        # Simple gradient-based orientation
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)
        
        # Compute average orientation
        angles = np.arctan2(grad_y, grad_x)
        
        # Use circular mean for angle averaging
        mean_angle = np.arctan2(np.mean(np.sin(2 * angles)), np.mean(np.cos(2 * angles))) / 2
        
        return mean_angle
    
    def classify_connexins_fast(self, connexin_props: List[Dict], nuclei_props: List[Dict]) -> List[Dict]:
        """
        Fast connexin classification using simplified criteria
        """
        if not nuclei_props:
            # If no nuclei, use morphology-based classification
            for conn in connexin_props:
                if conn['eccentricity'] > 0.8:  # Very elongated
                    conn['location_type'] = 'lateral'
                else:
                    conn['location_type'] = 'intercalated_disc'
            return connexin_props
        
        # Build spatial index for nuclei
        nuclei_positions = np.array([n['centroid'] for n in nuclei_props])
        
        for conn in connexin_props:
            conn_pos = np.array(conn['centroid'])
            
            # Find distance to nearest nucleus
            distances = np.sqrt(np.sum((nuclei_positions - conn_pos)**2, axis=1))
            min_distance = np.min(distances) * self.pixel_size_um
            
            # Simple classification rules
            if min_distance < 20:  # Close to nucleus
                if conn['eccentricity'] > 0.7:  # Elongated
                    conn['location_type'] = 'lateral'
                else:
                    conn['location_type'] = 'intercalated_disc'
            else:  # Far from nucleus
                conn['location_type'] = 'intercalated_disc'
            
            conn['nearest_nucleus_distance'] = min_distance
        
        return connexin_props
    
    def analyze_mega_slide(self, image_path: str, progress_callback=None) -> Dict:
        """
        Analyze massive slide using tile-based processing
        """
        print(f" Starting analysis of mega slide: {os.path.basename(image_path)}")
        
        # Get slide info
        slide_info = self.get_slide_info(image_path)
        
        # Generate tile coordinates
        tiles = self.generate_tile_coordinates(slide_info['width'], slide_info['height'])
        
        # Initialize results
        all_nuclei = []
        all_connexins = []
        processed_tiles = 0
        
        print(f" Processing {len(tiles)} tiles...")
        
        # Process each tile
        for i, tile_coords in enumerate(tqdm(tiles, desc="Processing tiles")):
            try:
                # Load tile
                tile = self.load_slide_tile(image_path, *tile_coords)
                
                # Process for nuclei
                tile_nuclei = self.process_tile_for_nuclei(tile, tile_coords)
                all_nuclei.extend(tile_nuclei)
                
                # Process for connexins
                tile_connexins = self.process_tile_for_connexins(tile, tile_coords)
                all_connexins.extend(tile_connexins)
                
                processed_tiles += 1
                
                # Memory management
                del tile
                if i % 10 == 0:  # Every 10 tiles
                    self._check_memory_usage()
                    
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(tiles))
                    
            except Exception as e:
                print(f"  Error processing tile {i}: {str(e)}")
                continue
        
        print(f" Processed {processed_tiles}/{len(tiles)} tiles")
        print(f"   Found {len(all_nuclei)} nuclei")
        print(f"   Found {len(all_connexins)} connexins")
        
        # Classify connexins
        print(" Classifying connexins...")
        all_connexins = self.classify_connexins_fast(all_connexins, all_nuclei)
        
        # Calculate statistics
        print(" Calculating statistics...")
        stats = self.calculate_mega_stats(all_connexins, all_nuclei, slide_info)
        
        return stats
    
    def calculate_mega_stats(self, connexin_props: List[Dict], nuclei_props: List[Dict], 
                           slide_info: Dict) -> Dict:
        """
        Calculate statistics for mega slide analysis
        """
        if not connexin_props:
            return self._empty_mega_stats()
        
        # Basic statistics
        areas = [c['area'] for c in connexin_props]
        total_slide_area = slide_info['width'] * slide_info['height'] * self.pixel_area_um2
        
        # Location-based statistics
        lateral_connexins = [c for c in connexin_props if c.get('location_type') == 'lateral']
        id_connexins = [c for c in connexin_props if c.get('location_type') == 'intercalated_disc']
        
        lateral_areas = [c['area'] for c in lateral_connexins] if lateral_connexins else [0]
        id_areas = [c['area'] for c in id_connexins] if id_connexins else [0]
        
        stats = {
            # Slide information
            'slide_width_pixels': slide_info['width'],
            'slide_height_pixels': slide_info['height'],
            'slide_area_mm2': total_slide_area / 1e6,
            'file_size_gb': slide_info['file_size_gb'],
            
            # Detection counts
            'total_nuclei': len(nuclei_props),
            'total_connexins': len(connexin_props),
            'lateral_connexins': len(lateral_connexins),
            'id_connexins': len(id_connexins),
            
            # Areas
            'total_connexin_area_um2': sum(areas),
            'mean_connexin_area_um2': np.mean(areas),
            'median_connexin_area_um2': np.median(areas),
            'lateral_area_um2': sum(lateral_areas),
            'id_area_um2': sum(id_areas),
            
            # Percentages
            'percent_lateral_count': len(lateral_connexins) / len(connexin_props) * 100,
            'percent_lateral_area': sum(lateral_areas) / sum(areas) * 100 if sum(areas) > 0 else 0,
            
            # Densities
            'nuclei_density_per_mm2': len(nuclei_props) / (total_slide_area / 1e6),
            'connexin_density_per_mm2': len(connexin_props) / (total_slide_area / 1e6),
            'connexins_per_nucleus': len(connexin_props) / max(len(nuclei_props), 1),
            
            # Processing info
            'pixel_size_um': self.pixel_size_um,
            'tile_size': self.tile_size,
            'processing_method': 'tile_based'
        }
        
        return stats
    
    def _empty_mega_stats(self) -> Dict:
        """Empty stats for failed analysis"""
        return {key: 0 for key in [
            'slide_width_pixels', 'slide_height_pixels', 'slide_area_mm2', 'file_size_gb',
            'total_nuclei', 'total_connexins', 'lateral_connexins', 'id_connexins',
            'total_connexin_area_um2', 'mean_connexin_area_um2', 'median_connexin_area_um2',
            'lateral_area_um2', 'id_area_um2', 'percent_lateral_count', 'percent_lateral_area',
            'nuclei_density_per_mm2', 'connexin_density_per_mm2', 'connexins_per_nucleus',
            'pixel_size_um', 'tile_size'
        ]}
    
    def create_downsampled_overview(self, image_path: str, output_path: str, max_size: int = 2048):
        """
        Create a downsampled overview image for visualization
        """
        print(f" Creating overview image...")
        
        with Image.open(image_path) as img:
            # Calculate downsample factor
            width, height = img.size
            max_dim = max(width, height)
            
            if max_dim > max_size:
                scale_factor = max_size / max_dim
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Downsample
                overview = img.resize((new_width, new_height), Image.LANCZOS)
                overview.save(output_path, quality=95)
                
                print(f"   Overview saved: {new_width}x{new_height} -> {output_path}")
                return scale_factor
            else:
                img.save(output_path, quality=95)
                return 1.0

def analyze_massive_slide(image_path: str, output_dir: str = None, tile_size: int = 1024) -> Dict:
    """
    Main function to analyze massive histology slides
    
    Args:
        image_path: Path to the 2.7GB+ TIFF file
        output_dir: Directory to save results (optional)
        tile_size: Size of processing tiles (reduce if memory issues persist)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MegaSlideConnexinAnalyzer(pixel_size_um=0.25, tile_size=tile_size)
    
    # Create overview image first
    if output_dir:
        overview_path = os.path.join(output_dir, "slide_overview.jpg")
        scale_factor = analyzer.create_downsampled_overview(image_path, overview_path)
    
    # Run analysis
    stats = analyzer.analyze_mega_slide(image_path)
    
    # Save results
    if output_dir:
        results_path = os.path.join(output_dir, "analysis_results.csv")
        df = pd.DataFrame([stats])
        df.to_csv(results_path, index=False)
        print(f" Results saved to: {results_path}")
    
    # Print summary
    print(f"\n ANALYSIS COMPLETE!")
    print(f"   Slide: {os.path.basename(image_path)}")
    print(f"   Nuclei: {stats['total_nuclei']:,}")
    print(f"   Connexins: {stats['total_connexins']:,}")
    print(f"   Lateral: {stats['lateral_connexins']:,} ({stats['percent_lateral_count']:.1f}%)")
    print(f"   Intercalated: {stats['id_connexins']:,}")
    print(f"   Density: {stats['connexin_density_per_mm2']:.1f} connexins/mm²")
    
    return stats

# Usage for your 2.7GB slides:
# stats = analyze_massive_slide("massive_slide.tiff", "results/", tile_size=512)
# 
# If still having memory issues, try smaller tiles:
# stats = analyze_massive_slide("massive_slide.tiff", "results/", tile_size=256)