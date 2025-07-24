#!/usr/bin/env python3
"""
Advanced Connexin Analysis for QuPath Export

"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
sns.set_palette("husl")


class SettingsManager:
    """Manages analysis settings and configuration."""
    
    def __init__(self, settings_path: Optional[str] = None):
        self.settings = self._load_settings(settings_path)
    
    def _load_settings(self, settings_path: Optional[str] = None) -> Dict:
        """Load settings from JSON file."""
        if settings_path is None:
            script_dir = Path(__file__).parent
            settings_path = script_dir / "IHC_settings.json"
        
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            logger.info(f"Loaded settings from {settings_path}")
            return settings
        except FileNotFoundError:
            logger.warning(f"Settings file not found: {settings_path}, using defaults")
            return self._get_default_settings()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing settings file: {e}")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict:
        """Return default analysis settings."""
        return {
            "analysis_parameters": {
                "total_objects_threshold": 200000,
                "max_nuclei_subsample": 100000,
                "max_connexins_subsample": 200000,
                "max_association_distance_um": 25.0,
                "processing_chunk_size": 1000,
                "cell_radius_multiplier": 3.0,
                "max_connexins_per_cell": 100,
                "grid_size_um": 100,
                "max_objects_per_grid": 1000
            },
            "output_settings": {
                "create_visualizations": False,
                "save_plots": True,
                "suppress_console_output": False,
                "expanded_analysis": False
            }
        }
    
    def get(self, section: str, key: str, default=None):
        """Get a specific setting value."""
        return self.settings.get(section, {}).get(key, default)


class DataLoader:
    """Handles loading and initial processing of QuPath CSV data."""
    
    def __init__(self, csv_file: Union[str, Path]):
        self.csv_file = Path(csv_file)
        self.data = None
        self.annotation_data = None
        
    def load_detection_data(self) -> pd.DataFrame:
        """Load detection data from CSV file."""
        try:
            self.data = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.data)} objects from {self.csv_file}")
            
            # Standardize column names
            self._standardize_column_names()
            
            # Validate required columns
            self._validate_required_columns()
            
            logger.info(f"Classifications found: {self.data['Classification'].value_counts().to_dict()}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading detection data: {e}")
            raise
    
    def load_annotation_data(self) -> Optional[pd.DataFrame]:
        """Load corresponding annotation data."""
        annotation_file = self.csv_file.parent / self.csv_file.name.replace("Detection.csv", "Annotation.csv")
        
        try:
            self.annotation_data = pd.read_csv(annotation_file)
            
            # Standardize column names
            if len(self.annotation_data.columns) > 9:
                self.annotation_data.columns.values[12] = 'Area_um2'
            
            # Filter for tissue annotations
            self.annotation_data = self.annotation_data[self.annotation_data['Name'] == 'Tissue']
            
            if len(self.annotation_data) > 0:
                area = self.annotation_data['Area_um2'].iloc[0]
                logger.info(f"Loaded annotation data. Tissue area: {area} μm²")
            else:
                logger.warning("No tissue annotations found")
            
            return self.annotation_data
            
        except FileNotFoundError:
            logger.warning(f"Annotation file not found: {annotation_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading annotation data: {e}")
            return None
    
    def _standardize_column_names(self) -> None:
        """Standardize column names for consistency."""
        if len(self.data.columns) > 7:
            self.data.columns.values[7] = 'Centroid_X_um'
        if len(self.data.columns) > 8:
            self.data.columns.values[8] = 'Centroid_Y_um'
    
    def _validate_required_columns(self) -> None:
        """Validate that required columns are present."""
        required_cols = ['Classification', 'Centroid_X_um', 'Centroid_Y_um', 'Nucleus: Area']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class AdvancedConnexinAnalyzer:
    """Main analysis class for connexin data."""
    
    def __init__(self, csv_file: Union[str, Path], settings: Optional[SettingsManager] = None):
        self.csv_file = Path(csv_file)
        self.settings = settings or SettingsManager()
        
        # Data containers
        self.data_loader = DataLoader(csv_file)
        self.nuclei = None
        self.connexins = None
        self.annotation_data = None
        self.results = {}
        
        # Load and process data
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Initialize and load all data."""
        # Load data
        data = self.data_loader.load_detection_data()
        self.annotation_data = self.data_loader.load_annotation_data()
        
        # Separate object types
        self._separate_objects(data)
        
        # Calculate basic metrics
        self._calculate_basic_metrics()
    
    def _separate_objects(self, data: pd.DataFrame) -> None:
        """Separate nuclei and connexin objects."""
        try:
            self.nuclei = data[data['Classification'] == 'CardiomyocyteNuclei'].copy().reset_index(drop=True)
            if data['Classification'].isin(['LateralConnexin']).any() and data['Classification'].isin(['IntercalatedConnexin']).any():
                self.lateralFlag = True
                self.lateralconnexin = data[data['Classification'] == 'LateralConnexin'].copy().reset_index(drop=True)
                self.intercalateddisc = data[data['Classification'] == 'IntercalatedConnexin'].copy().reset_index(drop=True)
                self.connexins = pd.concat([self.lateralconnexin, self.intercalateddisc])
                self.connexins['Classification'] = 'Connexin'
                logger.info(f"Separated {len(self.nuclei)} nuclei and {len(self.connexins)} connexins, including {len(self.lateralconnexin)} lateral connexins and {len(self.intercalateddisc)} intercalated discs")
            else:
                self.lateralFlag = False
                self.connexins = data[data['Classification'] == 'Connexin'].copy().reset_index(drop=True)
                logger.info(f"Separated {len(self.nuclei)} nuclei and {len(self.connexins)} connexins")
            
            if len(self.nuclei) == 0:
                raise ValueError("No cardiomyocyte nuclei found!")
            if len(self.connexins) == 0:
                raise ValueError("No connexin objects found!")
                
        except Exception as e:
            logger.error(f"Error separating objects: {e}")
            raise
    
    def _calculate_basic_metrics(self) -> None:
        """Calculate basic tissue-level metrics."""
        self.results.update({
            'total_nuclei': len(self.nuclei),
            'total_connexins': len(self.connexins),
            'total_connexin_area': self.connexins['Nucleus: Area'].sum(),
            'total_mean_plaque_size': self.connexins['Nucleus: Area'].mean(),
            'total_connexin_count_per_cell': len(self.connexins) / len(self.nuclei)
        })
        
        if self.annotation_data is not None and len(self.annotation_data) > 0:
            annotation_area = self.annotation_data['Area_um2'].iloc[0]
            self.results.update({
                'annotation_area': annotation_area,
                'total_connexin_area_per_cell': self.results['total_connexin_area'] / self.results['total_nuclei'],
                'total_connexin_count_per_annotation_area': self.results['total_connexins'] / annotation_area,
                'total_connexin_area_per_annotation_area': self.results['total_connexin_area'] / annotation_area,
                'total_nuclei_count_per_annotation_area': self.results['total_nuclei'] / annotation_area,
                'total_nuclei_area_per_annotation_area': self.nuclei['Nucleus: Area'].sum() / annotation_area
            })

        if self.lateralFlag:
            self.results.update({
                'total_lateral_connexins': len(self.lateralconnexin),
                'total_intercalated_discs': len(self.intercalateddisc),
                'total_percent_lateral_by_count': len(self.lateralconnexin)/len(self.intercalateddisc),
                'total_lateral_area': self.lateralconnexin['Nucleus: Area'].sum(),
                'total_intercalated_area': self.intercalateddisc['Nucleus: Area'].sum(),
                'total_percent_lateral_by_area': self.lateralconnexin['Nucleus: Area'].sum()/self.intercalateddisc['Nucleus: Area'].sum(),
                'total_lateral_mean_plaque_size': self.lateralconnexin['Nucleus: Area'].mean(),
                'total_intercalated_mean_plaque_size': self.intercalateddisc['Nucleus: Area'].mean(),
            })
    
    def subsample_for_analysis(self) -> Tuple[int, int]:
        """Subsample data if dataset is too large."""
        analysis_params = self.settings.settings.get("analysis_parameters", {})
        max_nuclei = analysis_params.get("max_nuclei_subsample", 50000)
        max_connexins = analysis_params.get("max_connexins_subsample", 100000)
        
        original_nuclei_count = len(self.nuclei)
        original_connexin_count = len(self.connexins)
        
        # Subsample nuclei if needed
        if len(self.nuclei) > max_nuclei:
            logger.info(f"Subsampling nuclei: {len(self.nuclei)} → {max_nuclei}")
            self.nuclei = self.nuclei.sample(n=max_nuclei, random_state=42).reset_index(drop=True)
        
        # Subsample connexins if needed
        if len(self.connexins) > max_connexins:
            logger.info(f"Subsampling connexins: {len(self.connexins)} → {max_connexins}")
            self.connexins = self.connexins.sample(n=max_connexins, random_state=42).reset_index(drop=True)
        
        logger.info(f"Subsampled: {original_nuclei_count}→{len(self.nuclei)} nuclei, "
                   f"{original_connexin_count}→{len(self.connexins)} connexins")
        
        return len(self.nuclei), len(self.connexins)
    
    def associate_connexins_with_nuclei(self) -> float:
        """Associate connexins with nearest nuclei using memory-efficient processing."""
        analysis_params = self.settings.settings.get("analysis_parameters", {})
        max_distance = analysis_params.get("max_association_distance_um", 25.0)
        chunk_size = analysis_params.get("processing_chunk_size", 1000)
        suppress_prints = self.settings.get("output_settings", "suppress_console_output", False)
        
        logger.info(f"Associating connexins with nuclei (max distance: {max_distance} μm)")
        
        # Initialize association columns
        self.connexins['associated_nucleus'] = -1
        self.connexins['distance_to_nucleus'] = np.inf
        
        # Get nuclei coordinates
        nuclei_coords = self.nuclei[['Centroid_X_um', 'Centroid_Y_um']].values
        
        # Process connexins in chunks
        num_chunks = (len(self.connexins) + chunk_size - 1) // chunk_size
        associated_count = 0
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(self.connexins))
            
            if chunk_idx % 10 == 0 and not suppress_prints:
                logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            
            # Get connexin coordinates for this chunk
            chunk_connexins = self.connexins.iloc[start_idx:end_idx]
            connexin_coords = chunk_connexins[['Centroid_X_um', 'Centroid_Y_um']].values
            
            # Process each connexin in chunk
            for i, (cx_idx, connexin_coord) in enumerate(zip(chunk_connexins.index, connexin_coords)):
                # Calculate distances to all nuclei
                distances = np.sqrt(
                    (nuclei_coords[:, 0] - connexin_coord[0])**2 + 
                    (nuclei_coords[:, 1] - connexin_coord[1])**2
                )
                
                # Find nearest nucleus within max distance
                min_distance = np.min(distances)
                
                if min_distance <= max_distance:
                    nearest_nucleus_idx = np.argmin(distances)
                    self.connexins.loc[cx_idx, 'associated_nucleus'] = nearest_nucleus_idx
                    self.connexins.loc[cx_idx, 'distance_to_nucleus'] = min_distance
                    associated_count += 1
        
        # Calculate basic nucleus metrics
        self._calculate_nucleus_metrics()
        
        association_rate = associated_count / len(self.connexins) * 100 if len(self.connexins) > 0 else 0
        logger.info(f"Associated {associated_count}/{len(self.connexins)} connexins ({association_rate:.1f}%)")
        
        return association_rate
    
    def _calculate_nucleus_metrics(self) -> None:
        """Calculate basic connexin metrics for each nucleus."""
        # Initialize metrics
        metric_columns = [
            'subsample_connexin_count', 'subsample_connexin_area', 'subsample_avg_plaque_size',
            'subsample_connexin_density', 'subsample_avg_distance_to_connexins'
        ]
        
        for col in metric_columns:
            self.nuclei[col] = 0.0
        
        # Group connexins by associated nucleus
        associated_connexins = self.connexins[self.connexins['associated_nucleus'] >= 0]
        
        if len(associated_connexins) > 0:
            connexin_groups = associated_connexins.groupby('associated_nucleus')
            
            for nucleus_idx, group in connexin_groups:
                if nucleus_idx < len(self.nuclei):
                    count = len(group)
                    total_area = group['Nucleus: Area'].sum()
                    avg_area = group['Nucleus: Area'].mean()
                    avg_distance = group['distance_to_nucleus'].mean()
                    
                    # Calculate density
                    nucleus_area = self.nuclei.loc[nucleus_idx, 'Nucleus: Area']
                    density = count / nucleus_area if nucleus_area > 0 else 0
                    
                    # Update nucleus metrics
                    self.nuclei.loc[nucleus_idx, 'subsample_connexin_count'] = count
                    self.nuclei.loc[nucleus_idx, 'subsample_connexin_area'] = total_area
                    self.nuclei.loc[nucleus_idx, 'subsample_avg_plaque_size'] = avg_area
                    self.nuclei.loc[nucleus_idx, 'subsample_connexin_density'] = density
                    self.nuclei.loc[nucleus_idx, 'subsample_avg_distance_to_connexins'] = avg_distance
        
        nuclei_with_connexins = np.sum(self.nuclei['subsample_connexin_count'] > 0)
        logger.info(f"Nuclei with connexins: {nuclei_with_connexins}/{len(self.nuclei)} "
                   f"({nuclei_with_connexins/len(self.nuclei)*100:.1f}%)")
    
    def calculate_lateralization_analysis(self) -> None:
        """Calculate advanced lateralization indices."""
        analysis_params = self.settings.settings.get("analysis_parameters", {})
        suppress_prints = self.settings.get("output_settings", "suppress_console_output", False)
        
        logger.info("Calculating lateralization analysis...")
        
        # Initialize lateralization metrics
        lateralization_columns = [
            'subsample_lateralization_index_distance', 'subsample_lateralization_index_angular', 
            'subsample_lateralization_index_density', 'subsample_intercalated_disc_connexins',
            'subsample_lateral_membrane_connexins', 'subsample_cell_elongation', 'subsample_cell_orientation',
            'subsample_lateralization_index_composite'
        ]
        
        for col in lateralization_columns:
            self.nuclei[col] = 0.0
        
        # Get nuclei with connexins
        nuclei_with_connexins = self.nuclei[self.nuclei['subsample_connexin_count'] > 0]
        processed_count = 0
        
        for nucleus_idx, nucleus in nuclei_with_connexins.iterrows():
            if nucleus['subsample_connexin_count'] == 0:
                continue
            
            # Get connexins for this nucleus
            nucleus_connexins = self.connexins[self.connexins['associated_nucleus'] == nucleus_idx]
            
            if len(nucleus_connexins) < 2:
                continue
            
            try:
                self._calculate_cell_lateralization(nucleus_idx, nucleus, nucleus_connexins)
                processed_count += 1
                
                if processed_count % 1000 == 0 and not suppress_prints:
                    logger.info(f"Processed {processed_count} cells...")
                    
            except Exception as e:
                logger.warning(f"Error processing nucleus {nucleus_idx}: {e}")
                continue
        
        # Calculate composite lateralization index
        self._calculate_composite_lateralization()
        
        logger.info(f"Calculated lateralization for {processed_count} cells")
    
    def _calculate_cell_lateralization(self, nucleus_idx: int, nucleus: pd.Series, 
                                     nucleus_connexins: pd.DataFrame) -> None:
        """Calculate lateralization metrics for a single cell."""
        nucleus_x = nucleus['Centroid_X_um']
        nucleus_y = nucleus['Centroid_Y_um']
        
        # Get connexin coordinates relative to nucleus
        rel_coords = nucleus_connexins[['Centroid_X_um', 'Centroid_Y_um']].values
        rel_coords[:, 0] -= nucleus_x
        rel_coords[:, 1] -= nucleus_y
        
        # Method 1: Distance-based lateralization
        distances = np.sqrt(rel_coords[:, 0]**2 + rel_coords[:, 1]**2)
        median_distance = np.median(distances)
        lateral_connexins_dist = np.sum(distances < median_distance)
        lateral_index_dist = (lateral_connexins_dist / len(nucleus_connexins)) * 100
        
        # Method 2: Angular distribution analysis
        if len(rel_coords) >= 3:
            angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
            
            # PCA for cell orientation
            coords_centered = rel_coords - np.mean(rel_coords, axis=0)
            cov_matrix = np.cov(coords_centered.T)
            
            # Check if covariance matrix is valid
            if np.all(np.isfinite(cov_matrix)) and cov_matrix.shape == (2, 2):
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    
                    # Ensure eigenvalues are positive
                    eigenvals = np.maximum(eigenvals, 1e-10)
                    
                    # Principal component represents cell's major axis
                    pc1 = eigenvecs[:, -1]
                    cell_angle = np.arctan2(pc1[1], pc1[0])
                    elongation = eigenvals[-1] / (eigenvals[-1] + eigenvals[-2]) if eigenvals[-2] > 0 else 1.0
                    
                    # Define intercalated disc zones
                    perp_angle_1 = cell_angle + np.pi/2
                    perp_angle_2 = cell_angle - np.pi/2
                    
                    def normalize_angle(angle):
                        return np.arctan2(np.sin(angle), np.cos(angle))
                    
                    perp_angle_1 = normalize_angle(perp_angle_1)
                    perp_angle_2 = normalize_angle(perp_angle_2)
                    
                    # Count connexins in intercalated disc vs lateral zones
                    intercalated_count = 0
                    lateral_count = 0
                    
                    for angle in angles:
                        dist_to_perp1 = abs(normalize_angle(angle - perp_angle_1))
                        dist_to_perp2 = abs(normalize_angle(angle - perp_angle_2))
                        min_dist_to_perp = min(dist_to_perp1, dist_to_perp2)
                        
                        if min_dist_to_perp <= np.pi/4:
                            intercalated_count += 1
                        else:
                            lateral_count += 1
                    
                    lateral_index_angular = (lateral_count / len(nucleus_connexins)) * 100
                    
                except np.linalg.LinAlgError:
                    # Fallback if PCA fails
                    lateral_index_angular = lateral_index_dist
                    intercalated_count = len(nucleus_connexins) - lateral_connexins_dist
                    lateral_count = lateral_connexins_dist
                    elongation = 1.0
                    cell_angle = 0.0
            else:
                # Fallback for invalid covariance matrix
                lateral_index_angular = lateral_index_dist
                intercalated_count = len(nucleus_connexins) - lateral_connexins_dist
                lateral_count = lateral_connexins_dist
                elongation = 1.0
                cell_angle = 0.0
            
            # Method 3: Density-based
            max_distance = np.max(distances)
            inner_radius = max_distance * 0.4
            inner_connexins = np.sum(distances <= inner_radius)
            lateral_index_density = (inner_connexins / len(nucleus_connexins)) * 100
        else:
            # Fallback for cells with few connexins
            lateral_index_angular = lateral_index_dist
            lateral_index_density = lateral_index_dist
            intercalated_count = len(nucleus_connexins) - lateral_connexins_dist
            lateral_count = lateral_connexins_dist
            elongation = 1.0
            cell_angle = 0.0
        
        # Store results with subsample_ prefix
        self.nuclei.loc[nucleus_idx, 'subsample_lateralization_index_distance'] = lateral_index_dist
        self.nuclei.loc[nucleus_idx, 'subsample_lateralization_index_angular'] = lateral_index_angular
        self.nuclei.loc[nucleus_idx, 'subsample_lateralization_index_density'] = lateral_index_density
        self.nuclei.loc[nucleus_idx, 'subsample_intercalated_disc_connexins'] = intercalated_count
        self.nuclei.loc[nucleus_idx, 'subsample_lateral_membrane_connexins'] = lateral_count
        self.nuclei.loc[nucleus_idx, 'subsample_cell_elongation'] = elongation
        self.nuclei.loc[nucleus_idx, 'subsample_cell_orientation'] = np.degrees(cell_angle)
    
    def _calculate_composite_lateralization(self) -> None:
        """Calculate composite lateralization index."""
        if 'subsample_lateralization_index_distance' not in self.nuclei.columns:
            self.nuclei['subsample_lateralization_index_distance'] = 0.0

        valid_cells = self.nuclei[self.nuclei['subsample_connexin_count'] >= 2].copy()
        
        if len(valid_cells) > 0:
            # Weighted average of the three methods
            weights = {'distance': 0.3, 'angular': 0.5, 'density': 0.2}
            
            # Initialize composite column
            self.nuclei['subsample_lateralization_index_composite'] = 0.0
            
            # Calculate composite for valid cells
            composite_values = (
                self.nuclei['subsample_lateralization_index_distance'] * weights['distance'] +
                self.nuclei['subsample_lateralization_index_angular'] * weights['angular'] +
                self.nuclei['subsample_lateralization_index_density'] * weights['density']
            )
            
            self.nuclei['subsample_lateralization_index_composite'] = composite_values
            
            # Calculate mean only for cells with sufficient connexins
            valid_composite = valid_cells['subsample_lateralization_index_composite']
            avg_lateralization = valid_composite.mean() if len(valid_composite) > 0 else 0
            logger.info(f"Mean composite lateralization index: {avg_lateralization:.1f}%")
        else:
            self.nuclei['subsample_lateralization_index_composite'] = 0.0
            logger.warning("Not enough cells with connexins for lateralization analysis")
    
    def calculate_spatial_heterogeneity(self) -> Dict:
        """Calculate spatial heterogeneity using grid analysis."""
        analysis_params = self.settings.settings.get("analysis_parameters", {})
        grid_size = analysis_params.get("grid_size_um", 100)
        max_objects_per_grid = analysis_params.get("max_objects_per_grid", 1000)
        suppress_prints = self.settings.get("output_settings", "suppress_console_output", False)
        
        logger.info(f"Calculating spatial heterogeneity (grid size: {grid_size} μm)")
        
        # Get spatial bounds
        all_x = pd.concat([self.nuclei['Centroid_X_um'], self.connexins['Centroid_X_um']])
        all_y = pd.concat([self.nuclei['Centroid_Y_um'], self.connexins['Centroid_Y_um']])
        
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()
        
        # Create grid
        x_bins = np.arange(min_x, max_x + grid_size, grid_size)
        y_bins = np.arange(min_y, max_y + grid_size, grid_size)
        
        # Calculate grid statistics
        metrics = {
            'subsample_connexin_count': [],
            'subsample_connexin_density': [],
            'subsample_avg_plaque_size': [],
            'subsample_avg_lateralization': []
        }
        
        grid_info = []
        processed_grids = 0
        
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                x_min, x_max = x_bins[i], x_bins[i + 1]
                y_min, y_max = y_bins[j], y_bins[j + 1]
                
                # Find objects in this grid square
                grid_connexins = self.connexins[
                    (self.connexins['Centroid_X_um'] >= x_min) &
                    (self.connexins['Centroid_X_um'] < x_max) &
                    (self.connexins['Centroid_Y_um'] >= y_min) &
                    (self.connexins['Centroid_Y_um'] < y_max)
                ]
                
                grid_nuclei = self.nuclei[
                    (self.nuclei['Centroid_X_um'] >= x_min) &
                    (self.nuclei['Centroid_X_um'] < x_max) &
                    (self.nuclei['Centroid_Y_um'] >= y_min) &
                    (self.nuclei['Centroid_Y_um'] < y_max)
                ]
                
                # Limit objects per grid for memory efficiency
                if len(grid_connexins) > max_objects_per_grid:
                    grid_connexins = grid_connexins.sample(n=max_objects_per_grid, random_state=42)
                
                connexin_count = len(grid_connexins)
                grid_area = grid_size * grid_size
                connexin_density = connexin_count / grid_area
                
                avg_plaque_size = grid_connexins['Nucleus: Area'].mean() if connexin_count > 0 else 0
                avg_lateralization = grid_nuclei['subsample_lateralization_index_composite'].mean() if len(grid_nuclei) > 0 else 0
                
                metrics['subsample_connexin_count'].append(connexin_count)
                metrics['subsample_connexin_density'].append(connexin_density)
                metrics['subsample_avg_plaque_size'].append(avg_plaque_size if not np.isnan(avg_plaque_size) else 0)
                metrics['subsample_avg_lateralization'].append(avg_lateralization if not np.isnan(avg_lateralization) else 0)
                
                grid_info.append({
                    'x_center': (x_min + x_max) / 2,
                    'y_center': (y_min + y_max) / 2,
                    'subsample_connexin_count': connexin_count,
                    'subsample_connexin_density': connexin_density,
                    'subsample_avg_plaque_size': avg_plaque_size if not np.isnan(avg_plaque_size) else 0,
                    'subsample_avg_lateralization': avg_lateralization if not np.isnan(avg_lateralization) else 0
                })
                
                processed_grids += 1
                
                if processed_grids % 100 == 0 and not suppress_prints:
                    logger.info(f"Processed {processed_grids} grid squares...")
        
        # Calculate heterogeneity indices (coefficient of variation)
        heterogeneity_results = {}
        
        for metric_name, values in metrics.items():
            values = [v for v in values if v > 0]  # Remove zeros
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                heterogeneity_results[f'{metric_name}_heterogeneity'] = cv
            else:
                heterogeneity_results[f'{metric_name}_heterogeneity'] = 0
        
        self.results['heterogeneity'] = heterogeneity_results
        #self.results['grid_info'] = grid_info[:10000]  # Limit stored grid info
        self.results['n_grid_squares'] = len([g for g in grid_info if g['subsample_connexin_count'] > 0])
        
        logger.info(f"Analyzed {processed_grids} grid squares")
        return heterogeneity_results
    
    def calculate_remodeling_scores(self) -> None:
        """Calculate comprehensive remodeling scores."""
        logger.info("Calculating remodeling scores...")
        
        # Select cells with sufficient connexin data
        valid_cells = self.nuclei[self.nuclei['subsample_connexin_count'] >= 2].copy()
        
        if len(valid_cells) < 5:
            logger.warning("Not enough cells with connexins for remodeling analysis")
            return
        
        # Define remodeling metrics with subsample_ prefix
        remodeling_metrics = [
            'subsample_lateralization_index_composite',
            'subsample_avg_plaque_size',
            'subsample_connexin_density',
            'subsample_cell_elongation'
        ]
        
        # Check available metrics
        available_metrics = [m for m in remodeling_metrics 
                           if m in valid_cells.columns and valid_cells[m].notna().any()]
        
        if len(available_metrics) < 2:
            logger.warning("Not enough valid metrics for remodeling score calculation")
            return
        
        # Standardize metrics (z-scores)
        standardized_data = pd.DataFrame()
        
        for metric in available_metrics:
            values = valid_cells[metric].dropna()
            if len(values) > 1 and values.std() > 0:
                mean_val = values.mean()
                std_val = values.std()
                
                # Adjust sign for pathological direction
                if metric == 'subsample_lateralization_index_composite':
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                elif metric == 'subsample_avg_plaque_size':
                    z_scores = -(valid_cells[metric] - mean_val) / std_val
                elif metric == 'subsample_connexin_density':
                    z_scores = -(valid_cells[metric] - mean_val) / std_val
                elif metric == 'subsample_cell_elongation':
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                else:
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                
                standardized_data[metric] = z_scores
        
        if len(standardized_data.columns) == 0:
            logger.warning("No valid standardized metrics")
            return
        
        # Weighted composite score
        weights = {
            'subsample_lateralization_index_composite': 0.4,
            'subsample_avg_plaque_size': 0.25,
            'subsample_connexin_density': 0.25,
            'subsample_cell_elongation': 0.1
        }
        
        composite_scores = np.zeros(len(valid_cells))
        total_weight = 0
        
        for metric in standardized_data.columns:
            if metric in weights:
                composite_scores += standardized_data[metric].fillna(0) * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            composite_scores /= total_weight
        
        # PCA-based score
        if len(standardized_data.columns) >= 2:
            pca_data = standardized_data.fillna(0)
            pca = PCA(n_components=min(2, len(pca_data.columns)))
            pca_scores = pca.fit_transform(pca_data)
            pca_remodeling_score = pca_scores[:, 0]
            pc1_weight = pca.explained_variance_ratio_[0]
        else:
            pca_remodeling_score = composite_scores
            pc1_weight = 1.0
        
        # Distance from healthy prototype
        healthy_prototype = {
            'subsample_lateralization_index_composite': -1.0,
            'subsample_avg_plaque_size': 0.0,
            'subsample_connexin_density': 1.0,
            'subsample_cell_elongation': 0.0
        }
        
        distance_scores = np.zeros(len(valid_cells))
        for metric in standardized_data.columns:
            if metric in healthy_prototype:
                metric_distances = (standardized_data[metric].fillna(0) - healthy_prototype[metric]) ** 2
                distance_scores += metric_distances
        
        distance_scores = np.sqrt(distance_scores)
        
        # Combine all methods
        final_remodeling_score = (
            composite_scores * 0.4 +
            pca_remodeling_score * (0.4 * pc1_weight) +
            distance_scores * 0.2
        )
        
        # Add scores to nuclei data with subsample_ prefix
        for score_type in ['composite', 'pca', 'distance', 'final']:
            self.nuclei[f'subsample_remodeling_score_{score_type}'] = np.nan
        
        self.nuclei.loc[valid_cells.index, 'subsample_remodeling_score_composite'] = composite_scores
        self.nuclei.loc[valid_cells.index, 'subsample_remodeling_score_pca'] = pca_remodeling_score
        self.nuclei.loc[valid_cells.index, 'subsample_remodeling_score_distance'] = distance_scores
        self.nuclei.loc[valid_cells.index, 'subsample_remodeling_score_final'] = final_remodeling_score
        
        logger.info(f"Calculated remodeling scores for {len(valid_cells)} cells")
        logger.info(f"Mean final remodeling score: {final_remodeling_score.mean():.2f} ± {final_remodeling_score.std():.2f}")
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        output_settings = self.settings.settings.get("output_settings", {})
        
        if not output_settings.get("create_visualizations", False):
            return
        
        logger.info("Creating comprehensive visualizations...")
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Create plots
            self._create_distribution_plots(fig, gs)
            self._create_spatial_plots(fig, gs)
            self._create_correlation_plots(fig, gs)
            self._create_summary_panel(fig, gs)
            
            plt.suptitle('Comprehensive Connexin Analysis Results', fontsize=16, fontweight='bold')
            
            if output_settings.get("save_plots", True):
                plt_filename = str(self.csv_file).replace('-Detection.csv', '_advanced_connexin_analysis.png')
                plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plots to {plt_filename}")
            
            plt.show()
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_distribution_plots(self, fig, gs):
        """Create distribution plots."""
        # Connexin plaque size distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.connexins) > 0:
            ax1.hist(self.connexins['Nucleus: Area'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Plaque Area (μm²)')
            ax1.set_ylabel('Count')
            ax1.set_title('Gap Junction Plaque Sizes')
            ax1.grid(alpha=0.3)
        
        # Connexin count per cell
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.nuclei) > 0:
            max_count = int(self.nuclei['subsample_connexin_count'].max())
            bins = range(0, max_count + 2) if max_count > 0 else [0, 1]
            ax2.hist(self.nuclei['subsample_connexin_count'], bins=bins,
                    alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Connexins per Cell')
            ax2.set_ylabel('Count')
            ax2.set_title('Connexin Count Distribution')
            ax2.grid(alpha=0.3)
    
    def _create_spatial_plots(self, fig, gs):
        """Create spatial distribution plots."""
        # Spatial distribution of connexins
        ax = fig.add_subplot(gs[1, 0])
        if len(self.connexins) > 0:
            scatter = ax.scatter(self.connexins['Centroid_X_um'], self.connexins['Centroid_Y_um'], 
                               c=self.connexins['Nucleus: Area'], cmap='viridis', alpha=0.6, s=20)
            ax.set_xlabel('X Coordinate (μm)')
            ax.set_ylabel('Y Coordinate (μm)')
            ax.set_title('Spatial Distribution of Connexins')
            plt.colorbar(scatter, ax=ax, label='Plaque Area (μm²)')
            ax.set_aspect('equal', adjustable='box')
    
    def _create_correlation_plots(self, fig, gs):
        """Create correlation plots."""
        # Lateralization vs Remodeling Score
        ax = fig.add_subplot(gs[2, 0])
        common_cells = self.nuclei.dropna(subset=['subsample_lateralization_index_composite', 'subsample_remodeling_score_final'])
        if len(common_cells) > 0:
            ax.scatter(common_cells['subsample_lateralization_index_composite'], 
                      common_cells['subsample_remodeling_score_final'], alpha=0.6, color='purple')
            ax.set_xlabel('Lateralization Index (%)')
            ax.set_ylabel('Remodeling Score')
            ax.set_title('Lateralization vs Remodeling')
            ax.grid(alpha=0.3)
            
            # Add correlation coefficient
            if len(common_cells) > 1:
                corr, p_val = pearsonr(common_cells['subsample_lateralization_index_composite'], 
                                     common_cells['subsample_remodeling_score_final'])
                ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_summary_panel(self, fig, gs):
        """Create summary text panel."""
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')
        
        summary_text = self._generate_summary_text()
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _generate_summary_text(self) -> str:
        """Generate formatted summary text."""
        total_nuclei = len(self.nuclei)
        total_connexins = len(self.connexins)
        nuclei_with_connexins = np.sum(self.nuclei['subsample_connexin_count'] > 0)
        
        # Basic metrics
        mean_plaque_size = self.connexins['Nucleus: Area'].mean() if len(self.connexins) > 0 else 0
        
        # Lateralization metrics
        valid_lat = self.nuclei['subsample_lateralization_index_composite'].dropna()
        mean_lateralization = valid_lat.mean() if len(valid_lat) > 0 else 0
        high_lateralization = np.sum(valid_lat > 25) if len(valid_lat) > 0 else 0
        
        # Remodeling metrics
        valid_remod = self.nuclei['subsample_remodeling_score_final'].dropna()
        mean_remodeling = valid_remod.mean() if len(valid_remod) > 0 else 0
        severe_remodeling = np.sum(valid_remod > 1.0) if len(valid_remod) > 0 else 0
        
        summary_text = f"""CONNEXIN ANALYSIS SUMMARY
{'='*50}

BASIC COUNTS:
Total Cardiomyocyte Nuclei: {total_nuclei}
Total Connexin Objects: {total_connexins}
Nuclei with Connexins: {nuclei_with_connexins} ({nuclei_with_connexins/total_nuclei*100:.1f}%)
Average Connexins per Nucleus: {total_connexins/total_nuclei:.1f}

PLAQUE SIZE METRICS:
Mean Plaque Size: {mean_plaque_size:.3f} μm²

LATERALIZATION METRICS:
Mean Lateralization Index: {mean_lateralization:.1f}%
Cells with High Lateralization (>25%): {high_lateralization}

REMODELING METRICS:
Mean Remodeling Score: {mean_remodeling:.2f}
Cells with Severe Remodeling (>1.0): {severe_remodeling}
"""
        return summary_text
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive report...")
        
        expanded_analysis = self.settings.get("output_settings", "expanded_analysis", False)
        
        # Base report structure
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_file': str(self.csv_file),
                'software': 'Advanced Connexin Analyzer v2.0',
                'expanded_analysis': expanded_analysis
            },
            'key_metrics': self.results.copy()
        }
        
        if expanded_analysis:
            # Add detailed metrics
            report.update({
                'data_summary': self._get_data_summary(),
                'plaque_size_metrics': self._get_plaque_metrics(),
                'lateralization_metrics': self._get_lateralization_metrics(),
                'remodeling_metrics': self._get_remodeling_metrics(),
                'spatial_metrics': self.results.get('heterogeneity', {}),
                'distance_metrics': self._get_distance_metrics()
            })
        
        # Save report
        report_filename = str(self.csv_file).replace('-Detection.csv', '_analysis_report.json')
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved as '{report_filename}'")
        self._print_key_findings()
        
        return report
    
    def _get_data_summary(self) -> Dict:
        """Get data summary metrics."""
        return {
            'total_nuclei': int(len(self.nuclei)),
            'total_connexins': int(len(self.connexins)),
            'subsample_nuclei_with_connexins': int(np.sum(self.nuclei['subsample_connexin_count'] > 0)),
            'subsample_association_rate_percent': float(np.sum(self.connexins['associated_nucleus'] >= 0) / len(self.connexins) * 100) if len(self.connexins) > 0 else 0
        }
    
    def _get_plaque_metrics(self) -> Dict:
        """Get plaque size metrics."""
        if len(self.connexins) == 0:
            return {}
        
        return {
            'total_mean_plaque_size_um2': float(self.connexins['Nucleus: Area'].mean()),
            'total_median_plaque_size_um2': float(self.connexins['Nucleus: Area'].median()),
            'total_std_plaque_size_um2': float(self.connexins['Nucleus: Area'].std()),
            'total_min_plaque_size_um2': float(self.connexins['Nucleus: Area'].min()),
            'total_max_plaque_size_um2': float(self.connexins['Nucleus: Area'].max()),
            'total_small_plaques_count': int(np.sum(self.connexins['Nucleus: Area'] < 0.5)),
            'total_large_plaques_count': int(np.sum(self.connexins['Nucleus: Area'] > 2.0))
        }
    
    def _get_lateralization_metrics(self) -> Dict:
        """Get lateralization metrics."""
        if 'subsample_lateralization_index_composite' not in self.nuclei.columns:
            logger.warning("Composite Lateralization column missing - returning empty metrics")
            return {}

        valid_lat = self.nuclei['subsample_lateralization_index_composite'].dropna()
        
        if len(valid_lat) == 0:
            return {}
        
        return {
            'subsample_mean_lateralization_index': float(valid_lat.mean()),
            'subsample_median_lateralization_index': float(valid_lat.median()),
            'subsample_std_lateralization_index': float(valid_lat.std()),
            'subsample_cells_analyzed_for_lateralization': int(len(valid_lat)),
            'subsample_cells_with_low_lateralization_0_15': int(np.sum(valid_lat <= 15)),
            'subsample_cells_with_moderate_lateralization_15_30': int(np.sum((valid_lat > 15) & (valid_lat <= 30))),
            'subsample_cells_with_high_lateralization_30_plus': int(np.sum(valid_lat > 30)),
            'subsample_pathological_lateralization_threshold_25_percent': int(np.sum(valid_lat > 25))
        }
    
    def _get_remodeling_metrics(self) -> Dict:
        """Get remodeling metrics."""
        valid_remod = self.nuclei['subsample_remodeling_score_final'].dropna()
        
        if len(valid_remod) == 0:
            return {}
        
        return {
            'subsample_mean_remodeling_score': float(valid_remod.mean()),
            'subsample_median_remodeling_score': float(valid_remod.median()),
            'subsample_std_remodeling_score': float(valid_remod.std()),
            'subsample_cells_analyzed_for_remodeling': int(len(valid_remod)),
            'subsample_cells_with_mild_remodeling': int(np.sum(valid_remod <= 0.5)),
            'subsample_cells_with_moderate_remodeling': int(np.sum((valid_remod > 0.5) & (valid_remod <= 1.0))),
            'subsample_cells_with_severe_remodeling': int(np.sum(valid_remod > 1.0))
        }
    
    def _get_distance_metrics(self) -> Dict:
        """Get distance metrics."""
        valid_distances = self.connexins[self.connexins['distance_to_nucleus'] < np.inf]['distance_to_nucleus']
        
        if len(valid_distances) == 0:
            return {}
        
        return {
            'subsample_mean_distance_to_nucleus_um': float(valid_distances.mean()),
            'subsample_median_distance_to_nucleus_um': float(valid_distances.median()),
            'subsample_std_distance_to_nucleus_um': float(valid_distances.std()),
            'subsample_min_distance_to_nucleus_um': float(valid_distances.min()),
            'subsample_max_distance_to_nucleus_um': float(valid_distances.max()),
            'subsample_connexins_within_5um': int(np.sum(valid_distances <= 5)),
            'subsample_connexins_within_10um': int(np.sum(valid_distances <= 10)),
            'subsample_connexins_within_15um': int(np.sum(valid_distances <= 15))
        }
    
    def _print_key_findings(self) -> None:
        """Print key findings summary."""
        suppress_output = self.settings.get("output_settings", "suppress_console_output", False)
        
        if suppress_output:
            return
        
        print("\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print("="*60)
        
        print(f"ENTIRE TISSUE:")
        for key, value in self.results.items():
            if isinstance(value, (int, float)):
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def save_processed_data(self) -> Tuple[str, str]:
        """Save processed data to CSV files."""
        logger.info("Saving processed data...")
        
        base_name = str(self.csv_file).replace('-Detection.csv', '')
        
        nuclei_filename = f"{base_name}_processed_nuclei.csv"
        connexin_filename = f"{base_name}_processed_connexins.csv"
        
        self.nuclei.to_csv(nuclei_filename, index=False)
        self.connexins.to_csv(connexin_filename, index=False)
        
        logger.info(f"Saved processed data:")
        logger.info(f"  Nuclei: {nuclei_filename}")
        logger.info(f"  Connexins: {connexin_filename}")
        
        return nuclei_filename, connexin_filename
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete analysis pipeline."""
        try:
            logger.info("Starting Advanced Connexin Analysis Pipeline")
            logger.info("="*70)
            
            # Check data size and determine analysis approach
            total_objects = len(self.nuclei) + len(self.connexins)
            analysis_params = self.settings.settings.get("analysis_parameters", {})
            threshold = analysis_params.get("total_objects_threshold", 200000)
            expanded_analysis = self.settings.get("output_settings", "expanded_analysis", False)
            
            if total_objects > threshold and expanded_analysis:
                logger.info(f"Large dataset detected ({total_objects:,} objects)")
                logger.info("Using memory-efficient processing and subsampling...")
                self.subsample_for_analysis()
            
            # Run analysis pipeline
            if expanded_analysis:
                # Step 1: Associate connexins with nuclei
                association_rate = self.associate_connexins_with_nuclei()
                
                if association_rate < 30:
                    logger.warning("Very low association rate - dataset may be too sparse")
                
                # Step 2: Calculate lateralization analysis
                self.calculate_lateralization_analysis()
                
                # Step 3: Calculate spatial heterogeneity
                self.calculate_spatial_heterogeneity()
                
                # Step 4: Calculate remodeling scores
                self.calculate_remodeling_scores()
                
                # Step 5: Create visualizations
                self.create_visualizations()
            
            # Step 6: Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Step 7: Save processed data
            self.save_processed_data()
            
            logger.info("Analysis complete!")
            
            if expanded_analysis:
                logger.info("\nGenerated files:")
                logger.info("  *_advanced_connexin_analysis.png - Comprehensive visualizations")
                logger.info("  *_analysis_report.json - Detailed analysis report")
                logger.info("  *_processed_nuclei.csv - Enhanced nuclei data")
                logger.info("  *_processed_connexins.csv - Enhanced connexin data")
                
                logger.info("\nInterpretation Guide:")
                logger.info("  • Lateralization Index: <15% normal, 15-30% mild, >30% severe")
                logger.info("  • Remodeling Score: <0.5 mild, 0.5-1.0 moderate, >1.0 severe")
                logger.info("  • Plaque Size: 0.5-2.0 μm² typical range")
                logger.info("  • Heterogeneity Index: <40% uniform, 40-80% moderate, >80% high")
                
                if total_objects > threshold:
                    logger.info(f"\nNote: Analysis used subsampled data ({len(self.nuclei):,} nuclei, {len(self.connexins):,} connexins)")
                    logger.info("Results are representative of the full dataset")
            
            return report
            
        except MemoryError as e:
            logger.error(f"Memory error: {str(e)}")
            logger.error("\nSuggestions to reduce memory usage:")
            logger.error("  1. Use a smaller chunk_size (e.g., 500)")
            logger.error("  2. Increase subsampling (reduce max_nuclei/max_connexins)")
            logger.error("  3. Use a larger grid_size for heterogeneity analysis")
            logger.error("  4. Process the dataset in smaller sections")
            raise
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            raise


def validate_input_file(csv_file: Union[str, Path]) -> Path:
    """Validate input file exists and has correct format."""
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_file}")
    
    if not csv_path.name.endswith('Detection.csv'):
        logger.warning(f"Input file doesn't end with 'Detection.csv': {csv_path.name}")
    
    # Check if corresponding annotation file exists
    annotation_file = csv_path.parent / csv_path.name.replace("Detection.csv", "Annotation.csv")
    if not annotation_file.exists():
        logger.warning(f"Corresponding annotation file not found: {annotation_file}")
    
    return csv_path


def setup_error_handling():
    """Setup enhanced error handling and logging."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Analysis interrupted by user")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error(
            "Uncaught exception occurred",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception


def main(csv_file: Optional[str] = None) -> bool:
    """
    Main analysis workflow with enhanced error handling.
    
    Args:
        csv_file: Path to CSV file to analyze
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    setup_error_handling()
    
    logger.info("\nADVANCED CONNEXIN ANALYSIS PIPELINE")
    logger.info("="*70)
    
    try:
        # Get input file
        if csv_file is None:
            if len(sys.argv) > 1:
                csv_file = sys.argv[1]
            else:
                csv_file = input("Enter the path to your QuPath export CSV file: ")
        
        # Validate input file
        csv_path = validate_input_file(csv_file)
        
        # Load settings
        settings = SettingsManager()
        
        # Initialize and run analyzer
        analyzer = AdvancedConnexinAnalyzer(csv_path, settings)
        report = analyzer.run_complete_analysis()
        
        logger.info("Analysis pipeline completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return False
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
        
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return False
        
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        logger.error("Try reducing dataset size or adjusting memory settings")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)