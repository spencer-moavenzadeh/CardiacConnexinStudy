#!/usr/bin/env python3
"""
Advanced Connexin Analysis for QuPath Export
============================================

This script analyzes connexin gap junctions from QuPath-exported CSV data.
It calculates comprehensive metrics including advanced lateralization analysis.

Input: Single CSV file with both CardiomyocyteNuclei and Connexin objects
Output: Comprehensive analysis with all requested metrics

Usage:
python advanced_connexin_analysis.py your_export_file.csv
"""

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
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedConnexinAnalyzer:
    def __init__(self, csv_file):
        """
        Initialize the advanced connexin analyzer
        
        Parameters:
        csv_file: Path to QuPath export CSV file
        """
        self.csv_file = csv_file
        self.data = None
        self.nuclei = None
        self.connexins = None
        self.results = {}
        
        # Load and process data
        self.load_data()
        self.separate_objects()
        
    def load_data(self):
        """Load data from QuPath CSV export"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} objects from {self.csv_file}")
            
            # Rename columns: 
            self.data.columns.values[7] = 'Centroid X μm'
            self.data.columns.values[8] = 'Centroid Y μm'
            
            # Display column info
            print(f"Columns found: {len(self.data.columns)}")
            print(f"Classifications: {self.data['Classification'].value_counts().to_dict()}")
            
            # Check for required columns
            required_cols = ['Classification', 'Centroid X μm', 'Centroid Y μm', 'Nucleus: Area']
            print(self.data.columns)
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"Missing columns: {missing_cols}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def separate_objects(self):
        """Separate nuclei and connexin objects"""
        try:
            self.nuclei = self.data[self.data['Classification'] == 'CardiomyocyteNuclei'].copy()
            self.connexins = self.data[self.data['Classification'] == 'Connexin'].copy()
            
            print(f"Separated {len(self.nuclei)} cardiomyocyte nuclei and {len(self.connexins)} connexin objects")
            
            if len(self.nuclei) == 0:
                print("No cardiomyocyte nuclei found!")
                return False
                
            if len(self.connexins) == 0:
                print("No connexin objects found!")
                return False
                
            # Reset indices
            self.nuclei.reset_index(drop=True, inplace=True)
            self.connexins.reset_index(drop=True, inplace=True)
            
            return True
            
        except Exception as e:
            print(f"Error separating objects: {e}")
            return False
    
    def associate_connexins_with_nuclei(self, max_distance=25.0):
        """
        Associate each connexin with the nearest nucleus using advanced spatial analysis
        
        Parameters:
        max_distance: Maximum distance in micrometers for association
        """
        print(f"\nAssociating connexins with nuclei (max distance: {max_distance} μm)...")
        
        # Get coordinates
        nuclei_coords = self.nuclei[['Centroid X μm', 'Centroid Y μm']].values
        connexin_coords = self.connexins[['Centroid X μm', 'Centroid Y μm']].values
        
        # Calculate all pairwise distances
        distances = cdist(connexin_coords, nuclei_coords)
        
        # Find nearest nucleus for each connexin
        nearest_nucleus_idx = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        # Only associate if within max distance
        valid_associations = min_distances <= max_distance
        
        # Add association info to connexin data
        self.connexins['associated_nucleus'] = -1
        self.connexins['distance_to_nucleus'] = np.inf
        
        self.connexins.loc[valid_associations, 'associated_nucleus'] = nearest_nucleus_idx[valid_associations]
        self.connexins.loc[valid_associations, 'distance_to_nucleus'] = min_distances[valid_associations]
        
        # Calculate basic metrics per nucleus
        self.calculate_basic_nucleus_metrics()
        
        associated_count = np.sum(valid_associations)
        association_rate = associated_count / len(self.connexins) * 100 if len(self.connexins) > 0 else 0
        
        print(f"Associated {associated_count}/{len(self.connexins)} connexins ({association_rate:.1f}%)")
        
        return association_rate
    
    def calculate_basic_nucleus_metrics(self):
        """Calculate basic connexin metrics for each nucleus"""
        print("Calculating basic nucleus metrics...")
        
        # Initialize metrics
        self.nuclei['connexin_count'] = 0
        self.nuclei['total_connexin_area'] = 0.0
        self.nuclei['avg_plaque_size'] = 0.0
        self.nuclei['connexin_density'] = 0.0
        self.nuclei['avg_distance_to_connexins'] = 0.0
        
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
                    
                    # Calculate density (connexins per nucleus area)
                    nucleus_area = self.nuclei.loc[nucleus_idx, 'Nucleus: Area']
                    density = count / nucleus_area if nucleus_area > 0 else 0
                    
                    # Update nucleus metrics
                    self.nuclei.loc[nucleus_idx, 'connexin_count'] = count
                    self.nuclei.loc[nucleus_idx, 'total_connexin_area'] = total_area
                    self.nuclei.loc[nucleus_idx, 'avg_plaque_size'] = avg_area
                    self.nuclei.loc[nucleus_idx, 'connexin_density'] = density
                    self.nuclei.loc[nucleus_idx, 'avg_distance_to_connexins'] = avg_distance
        
        nuclei_with_connexins = np.sum(self.nuclei['connexin_count'] > 0)
        print(f"Nuclei with connexins: {nuclei_with_connexins}/{len(self.nuclei)} ({nuclei_with_connexins/len(self.nuclei)*100:.1f}%)")
    
    def calculate_advanced_lateralization(self, cell_radius_multiplier=3.0, end_zone_fraction=0.15):
        """
        Calculate advanced lateralization index using multiple approaches
        
        Parameters:
        cell_radius_multiplier: Multiplier for estimated cell radius from nucleus
        end_zone_fraction: Fraction of cell length to consider as intercalated disc zones
        """
        print(f"\nCalculating advanced lateralization indices...")
        
        # Initialize lateralization metrics
        self.nuclei['lateralization_index_distance'] = 0.0
        self.nuclei['lateralization_index_angular'] = 0.0
        self.nuclei['lateralization_index_density'] = 0.0
        self.nuclei['intercalated_disc_connexins'] = 0
        self.nuclei['lateral_membrane_connexins'] = 0
        self.nuclei['cell_elongation'] = 0.0
        self.nuclei['cell_orientation'] = 0.0
        
        for nucleus_idx, nucleus in self.nuclei.iterrows():
            if nucleus['connexin_count'] == 0:
                continue
                
            # Get connexins for this nucleus
            nucleus_connexins = self.connexins[self.connexins['associated_nucleus'] == nucleus_idx]
            
            if len(nucleus_connexins) < 2:
                continue
            
            nucleus_x = nucleus['Centroid X μm']
            nucleus_y = nucleus['Centroid Y μm']
            nucleus_area = nucleus['Nucleus: Area']
            
            # Estimate cell properties from nucleus
            estimated_cell_radius = np.sqrt(nucleus_area / np.pi) * cell_radius_multiplier
            
            # Get connexin coordinates relative to nucleus
            rel_coords = nucleus_connexins[['Centroid X μm', 'Centroid Y μm']].values
            rel_coords[:, 0] -= nucleus_x
            rel_coords[:, 1] -= nucleus_y
            
            # Method 1: Distance-based lateralization
            # Connexins closer to nucleus = more lateral
            distances = np.sqrt(rel_coords[:, 0]**2 + rel_coords[:, 1]**2)
            median_distance = np.median(distances)
            lateral_connexins_dist = np.sum(distances < median_distance)
            lateral_index_dist = (lateral_connexins_dist / len(nucleus_connexins)) * 100
            
            # Method 2: Angular distribution analysis
            # Calculate angles from nucleus to each connexin
            angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
            
            # Find dominant cell orientation using PCA
            if len(rel_coords) >= 3:
                pca = PCA(n_components=2)
                pca.fit(rel_coords)
                
                # Principal component represents cell's major axis
                pc1 = pca.components_[0]
                cell_angle = np.arctan2(pc1[1], pc1[0])
                explained_variance_ratio = pca.explained_variance_ratio_[0]
                
                # Cell elongation factor
                elongation = explained_variance_ratio / (explained_variance_ratio + pca.explained_variance_ratio_[1])
                
                # Define intercalated disc zones (perpendicular to major axis)
                # Angles within ±45° of perpendicular to major axis are considered intercalated disc
                perp_angle_1 = cell_angle + np.pi/2
                perp_angle_2 = cell_angle - np.pi/2
                
                # Normalize angles to [-π, π]
                def normalize_angle(angle):
                    return np.arctan2(np.sin(angle), np.cos(angle))
                
                perp_angle_1 = normalize_angle(perp_angle_1)
                perp_angle_2 = normalize_angle(perp_angle_2)
                
                # Count connexins in intercalated disc zones vs lateral zones
                intercalated_count = 0
                lateral_count = 0
                
                for angle in angles:
                    # Calculate angular distance to perpendicular directions
                    dist_to_perp1 = abs(normalize_angle(angle - perp_angle_1))
                    dist_to_perp2 = abs(normalize_angle(angle - perp_angle_2))
                    min_dist_to_perp = min(dist_to_perp1, dist_to_perp2)
                    
                    # If within 45° of perpendicular, consider intercalated disc
                    if min_dist_to_perp <= np.pi/4:
                        intercalated_count += 1
                    else:
                        lateral_count += 1
                
                lateral_index_angular = (lateral_count / len(nucleus_connexins)) * 100
                
                # Method 3: Density-based lateralization
                # Create distance rings and analyze connexin density distribution
                max_distance = np.max(distances)
                if max_distance > 0:
                    # Inner ring (closer to nucleus = more lateral)
                    inner_radius = max_distance * 0.4
                    inner_connexins = np.sum(distances <= inner_radius)
                    lateral_index_density = (inner_connexins / len(nucleus_connexins)) * 100
                else:
                    lateral_index_density = 0
                
                # Store results
                self.nuclei.loc[nucleus_idx, 'lateralization_index_distance'] = lateral_index_dist
                self.nuclei.loc[nucleus_idx, 'lateralization_index_angular'] = lateral_index_angular
                self.nuclei.loc[nucleus_idx, 'lateralization_index_density'] = lateral_index_density
                self.nuclei.loc[nucleus_idx, 'intercalated_disc_connexins'] = intercalated_count
                self.nuclei.loc[nucleus_idx, 'lateral_membrane_connexins'] = lateral_count
                self.nuclei.loc[nucleus_idx, 'cell_elongation'] = elongation
                self.nuclei.loc[nucleus_idx, 'cell_orientation'] = np.degrees(cell_angle)
            
            else:
                # Fallback for cells with few connexins
                self.nuclei.loc[nucleus_idx, 'lateralization_index_distance'] = lateral_index_dist
                self.nuclei.loc[nucleus_idx, 'lateralization_index_angular'] = 50.0  # Neutral
                self.nuclei.loc[nucleus_idx, 'lateralization_index_density'] = lateral_index_dist
        
        # Calculate composite lateralization index
        valid_cells = self.nuclei[self.nuclei['connexin_count'] >= 2].copy()
        if len(valid_cells) > 0:
            # Weighted average of the three methods
            weights = {'distance': 0.3, 'angular': 0.5, 'density': 0.2}
            
            self.nuclei['lateralization_index_composite'] = (
                self.nuclei['lateralization_index_distance'] * weights['distance'] +
                self.nuclei['lateralization_index_angular'] * weights['angular'] +
                self.nuclei['lateralization_index_density'] * weights['density']
            )
            
            avg_lateralization = valid_cells['lateralization_index_composite'].mean()
            print(f"Calculated lateralization for {len(valid_cells)} cells")
            print(f"Mean composite lateralization index: {avg_lateralization:.1f}%")
        else:
            self.nuclei['lateralization_index_composite'] = 0.0
            print("Not enough cells with connexins for lateralization analysis")
    
    def calculate_spatial_heterogeneity(self, grid_size=50):
        """
        Calculate spatial heterogeneity using advanced grid analysis
        
        Parameters:
        grid_size: Size of grid squares in micrometers
        """
        print(f"\nCalculating spatial heterogeneity (grid size: {grid_size} μm)...")
        
        # Get spatial bounds
        all_x = pd.concat([self.nuclei['Centroid X μm'], self.connexins['Centroid X μm']])
        all_y = pd.concat([self.nuclei['Centroid Y μm'], self.connexins['Centroid Y μm']])
        
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()
        
        # Create grid
        x_bins = np.arange(min_x, max_x + grid_size, grid_size)
        y_bins = np.arange(min_y, max_y + grid_size, grid_size)
        
        # Calculate grid statistics for different metrics
        metrics = {
            'connexin_count': [],
            'connexin_density': [],
            'avg_plaque_size': [],
            'avg_lateralization': []
        }
        
        grid_info = []
        
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                x_min, x_max = x_bins[i], x_bins[i + 1]
                y_min, y_max = y_bins[j], y_bins[j + 1]
                
                # Find connexins in this grid square
                grid_connexins = self.connexins[
                    (self.connexins['Centroid X μm'] >= x_min) &
                    (self.connexins['Centroid X μm'] < x_max) &
                    (self.connexins['Centroid Y μm'] >= y_min) &
                    (self.connexins['Centroid Y μm'] < y_max)
                ]
                
                # Find nuclei in this grid square
                grid_nuclei = self.nuclei[
                    (self.nuclei['Centroid X μm'] >= x_min) &
                    (self.nuclei['Centroid X μm'] < x_max) &
                    (self.nuclei['Centroid Y μm'] >= y_min) &
                    (self.nuclei['Centroid Y μm'] < y_max)
                ]
                
                connexin_count = len(grid_connexins)
                nucleus_count = len(grid_nuclei)
                
                # Calculate metrics for this grid
                grid_area = grid_size * grid_size  # μm²
                connexin_density = connexin_count / grid_area
                
                avg_plaque_size = grid_connexins['Nucleus: Area'].mean() if connexin_count > 0 else 0
                avg_lateralization = grid_nuclei['lateralization_index_composite'].mean() if nucleus_count > 0 else 0
                
                metrics['connexin_count'].append(connexin_count)
                metrics['connexin_density'].append(connexin_density)
                metrics['avg_plaque_size'].append(avg_plaque_size if not np.isnan(avg_plaque_size) else 0)
                metrics['avg_lateralization'].append(avg_lateralization if not np.isnan(avg_lateralization) else 0)
                
                grid_info.append({
                    'x_center': (x_min + x_max) / 2,
                    'y_center': (y_min + y_max) / 2,
                    'connexin_count': connexin_count,
                    'nucleus_count': nucleus_count,
                    'connexin_density': connexin_density,
                    'avg_plaque_size': avg_plaque_size if not np.isnan(avg_plaque_size) else 0,
                    'avg_lateralization': avg_lateralization if not np.isnan(avg_lateralization) else 0
                })
        
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
        self.results['grid_info'] = grid_info
        self.results['n_grid_squares'] = len([g for g in grid_info if g['connexin_count'] > 0])
        
        print(f"Analyzed {len(grid_info)} grid squares")
        for metric, het_index in heterogeneity_results.items():
            print(f"{metric}: {het_index:.1f}%")
        
        return heterogeneity_results
    
    def calculate_remodeling_scores(self):
        """Calculate comprehensive remodeling scores using multiple approaches"""
        print(f"\nCalculating remodeling scores...")
        
        # Select cells with sufficient connexin data
        valid_cells = self.nuclei[self.nuclei['connexin_count'] >= 2].copy()
        
        if len(valid_cells) < 5:
            print("Not enough cells with connexins for remodeling analysis")
            return
        
        # Define remodeling metrics
        remodeling_metrics = [
            'lateralization_index_composite',
            'avg_plaque_size',
            'connexin_density',
            'cell_elongation'
        ]
        
        # Check which metrics are available
        available_metrics = [m for m in remodeling_metrics if m in valid_cells.columns and valid_cells[m].notna().any()]
        
        if len(available_metrics) < 2:
            print("Not enough valid metrics for remodeling score calculation")
            return
        
        print(f"Using metrics: {available_metrics}")
        
        # Standardize metrics (z-scores)
        standardized_data = pd.DataFrame()
        
        for metric in available_metrics:
            values = valid_cells[metric].dropna()
            if len(values) > 1 and values.std() > 0:
                mean_val = values.mean()
                std_val = values.std()
                
                # Adjust sign for pathological direction
                if metric == 'lateralization_index_composite':
                    # Higher lateralization = more pathological
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                elif metric == 'avg_plaque_size':
                    # Smaller plaques = more pathological  
                    z_scores = -(valid_cells[metric] - mean_val) / std_val
                elif metric == 'connexin_density':
                    # Lower density = more pathological
                    z_scores = -(valid_cells[metric] - mean_val) / std_val
                elif metric == 'cell_elongation':
                    # Higher elongation might indicate structural changes
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                else:
                    z_scores = (valid_cells[metric] - mean_val) / std_val
                
                standardized_data[metric] = z_scores
        
        if len(standardized_data.columns) == 0:
            print("No valid standardized metrics")
            return
        
        # Method 1: Weighted composite score
        weights = {
            'lateralization_index_composite': 0.4,
            'avg_plaque_size': 0.25,
            'connexin_density': 0.25,
            'cell_elongation': 0.1
        }
        
        composite_scores = np.zeros(len(valid_cells))
        total_weight = 0
        
        for metric in standardized_data.columns:
            if metric in weights:
                composite_scores += standardized_data[metric].fillna(0) * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            composite_scores /= total_weight
        
        # Method 2: PCA-based score
        if len(standardized_data.columns) >= 2:
            pca_data = standardized_data.fillna(0)
            
            pca = PCA(n_components=min(2, len(pca_data.columns)))
            pca_scores = pca.fit_transform(pca_data)
            
            # Use first principal component as remodeling score
            pca_remodeling_score = pca_scores[:, 0]
            
            # Weight by explained variance
            pc1_weight = pca.explained_variance_ratio_[0]
            
            print(f"PC1 explains {pc1_weight:.1%} of variance")
        else:
            pca_remodeling_score = composite_scores
            pc1_weight = 1.0
        
        # Method 3: Distance from healthy prototype
        # Define "healthy" as low lateralization, normal plaque size, high density
        healthy_prototype = {}
        for metric in available_metrics:
            if metric == 'lateralization_index_composite':
                healthy_prototype[metric] = -1.0  # Low lateralization
            elif metric == 'avg_plaque_size':
                healthy_prototype[metric] = 0.0   # Normal plaque size
            elif metric == 'connexin_density':
                healthy_prototype[metric] = 1.0   # High density
            elif metric == 'cell_elongation':
                healthy_prototype[metric] = 0.0   # Normal elongation
            else:
                healthy_prototype[metric] = 0.0
        
        # Calculate distance from prototype
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
        
        # Add scores to nuclei data
        self.nuclei['remodeling_score_composite'] = np.nan
        self.nuclei['remodeling_score_pca'] = np.nan
        self.nuclei['remodeling_score_distance'] = np.nan
        self.nuclei['remodeling_score_final'] = np.nan
        
        self.nuclei.loc[valid_cells.index, 'remodeling_score_composite'] = composite_scores
        self.nuclei.loc[valid_cells.index, 'remodeling_score_pca'] = pca_remodeling_score
        self.nuclei.loc[valid_cells.index, 'remodeling_score_distance'] = distance_scores
        self.nuclei.loc[valid_cells.index, 'remodeling_score_final'] = final_remodeling_score
        
        print(f"Calculated remodeling scores for {len(valid_cells)} cells")
        print(f"Mean final remodeling score: {final_remodeling_score.mean():.2f} ± {final_remodeling_score.std():.2f}")
        
        # Classify remodeling severity
        remodeling_thresholds = {'mild': 0.5, 'moderate': 1.0, 'severe': 1.5}
        
        mild = np.sum(final_remodeling_score <= remodeling_thresholds['mild'])
        moderate = np.sum((final_remodeling_score > remodeling_thresholds['mild']) & 
                         (final_remodeling_score <= remodeling_thresholds['moderate']))
        severe = np.sum(final_remodeling_score > remodeling_thresholds['moderate'])
        
        print(f"Remodeling classification:")
        print(f"   Mild (≤{remodeling_thresholds['mild']}): {mild} cells ({mild/len(valid_cells)*100:.1f}%)")
        print(f"   Moderate ({remodeling_thresholds['mild']}-{remodeling_thresholds['moderate']}): {moderate} cells ({moderate/len(valid_cells)*100:.1f}%)")
        print(f"   Severe (>{remodeling_thresholds['moderate']}): {severe} cells ({severe/len(valid_cells)*100:.1f}%)")
    
    def create_comprehensive_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print(f"\nCreating comprehensive visualizations...")
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Connexin plaque size distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.connexins) > 0:
            ax1.hist(self.connexins['Nucleus: Area'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Plaque Area (μm²)')
            ax1.set_ylabel('Count')
            ax1.set_title('Gap Junction Plaque Sizes')
            ax1.grid(alpha=0.3)
        
        # 2. Connexin count per cell
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.nuclei) > 0:
            ax2.hist(self.nuclei['connexin_count'], bins=range(0, int(self.nuclei['connexin_count'].max()) + 2), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Connexins per Cell')
            ax2.set_ylabel('Count')
            ax2.set_title('Connexin Count Distribution')
            ax2.grid(alpha=0.3)
        
        # 3. Lateralization index distribution
        ax3 = fig.add_subplot(gs[0, 2])
        valid_lat = self.nuclei['lateralization_index_composite'].dropna()
        if len(valid_lat) > 0:
            ax3.hist(valid_lat, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_xlabel('Lateralization Index (%)')
            ax3.set_ylabel('Count')
            ax3.set_title('Connexin Lateralization')
            ax3.grid(alpha=0.3)
            ax3.axvline(25, color='red', linestyle='--', alpha=0.7, label='Pathological threshold')
            ax3.legend()
        
        # 4. Remodeling score distribution
        ax4 = fig.add_subplot(gs[0, 3])
        valid_remod = self.nuclei['remodeling_score_final'].dropna()
        if len(valid_remod) > 0:
            ax4.hist(valid_remod, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_xlabel('Remodeling Score')
            ax4.set_ylabel('Count')
            ax4.set_title('Remodeling Score Distribution')
            ax4.grid(alpha=0.3)
            ax4.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Severe threshold')
            ax4.legend()
        
        # 5. Spatial distribution of connexins
        ax5 = fig.add_subplot(gs[1, 0])
        if len(self.connexins) > 0:
            scatter = ax5.scatter(self.connexins['Centroid X μm'], self.connexins['Centroid Y μm'], 
                                c=self.connexins['Nucleus: Area'], cmap='viridis', alpha=0.6, s=20)
            ax5.set_xlabel('X Coordinate (μm)')
            ax5.set_ylabel('Y Coordinate (μm)')
            ax5.set_title('Spatial Distribution of Connexins')
            plt.colorbar(scatter, ax=ax5, label='Plaque Area (μm²)')
            ax5.set_aspect('equal', adjustable='box')
        
        # 6. Spatial distribution of lateralization
        ax6 = fig.add_subplot(gs[1, 1])
        valid_cells = self.nuclei[self.nuclei['lateralization_index_composite'].notna()]
        if len(valid_cells) > 0:
            scatter = ax6.scatter(valid_cells['Centroid X μm'], valid_cells['Centroid Y μm'], 
                                c=valid_cells['lateralization_index_composite'], cmap='RdYlBu_r', 
                                alpha=0.7, s=30)
            ax6.set_xlabel('X Coordinate (μm)')
            ax6.set_ylabel('Y Coordinate (μm)')
            ax6.set_title('Spatial Distribution of Lateralization')
            plt.colorbar(scatter, ax=ax6, label='Lateralization Index (%)')
            ax6.set_aspect('equal', adjustable='box')
        
        # 7. Correlation: Lateralization vs Remodeling Score
        ax7 = fig.add_subplot(gs[1, 2])
        common_cells = self.nuclei.dropna(subset=['lateralization_index_composite', 'remodeling_score_final'])
        if len(common_cells) > 0:
            ax7.scatter(common_cells['lateralization_index_composite'], 
                       common_cells['remodeling_score_final'], alpha=0.6, color='purple')
            ax7.set_xlabel('Lateralization Index (%)')
            ax7.set_ylabel('Remodeling Score')
            ax7.set_title('Lateralization vs Remodeling')
            ax7.grid(alpha=0.3)
            
            # Add correlation coefficient
            corr, p_val = pearsonr(common_cells['lateralization_index_composite'], 
                                 common_cells['remodeling_score_final'])
            ax7.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                    transform=ax7.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 8. Connexin density vs cell area
        ax8 = fig.add_subplot(gs[1, 3])
        valid_density = self.nuclei[self.nuclei['connexin_density'] > 0]
        if len(valid_density) > 0:
            ax8.scatter(valid_density['Nucleus: Area'], valid_density['connexin_density'], 
                       alpha=0.6, color='green')
            ax8.set_xlabel('Nucleus Area (μm²)')
            ax8.set_ylabel('Connexin Density (count/μm²)')
            ax8.set_title('Nucleus Size vs Connexin Density')
            ax8.grid(alpha=0.3)
        
        # 9. Distance to nucleus distribution
        ax9 = fig.add_subplot(gs[2, 0])
        valid_distances = self.connexins[self.connexins['distance_to_nucleus'] < np.inf]['distance_to_nucleus']
        if len(valid_distances) > 0:
            ax9.hist(valid_distances, bins=30, alpha=0.7, color='cyan', edgecolor='black')
            ax9.set_xlabel('Distance to Nucleus (μm)')
            ax9.set_ylabel('Count')
            ax9.set_title('Connexin Distance to Nucleus')
            ax9.grid(alpha=0.3)
            ax9.axvline(valid_distances.mean(), color='red', linestyle='--', 
                       label=f'Mean: {valid_distances.mean():.1f} μm')
            ax9.legend()
        
        # 10. Lateralization methods comparison
        ax10 = fig.add_subplot(gs[2, 1])
        methods = ['lateralization_index_distance', 'lateralization_index_angular', 'lateralization_index_density']
        available_methods = [m for m in methods if m in self.nuclei.columns]
        
        if len(available_methods) >= 2:
            method_data = []
            method_names = []
            for method in available_methods:
                valid_data = self.nuclei[method].dropna()
                if len(valid_data) > 0:
                    method_data.append(valid_data)
                    method_names.append(method.replace('lateralization_index_', '').title())
            
            if method_data:
                ax10.boxplot(method_data, labels=method_names)
                ax10.set_ylabel('Lateralization Index (%)')
                ax10.set_title('Lateralization Methods Comparison')
                ax10.grid(alpha=0.3)
                plt.setp(ax10.get_xticklabels(), rotation=45)
        
        # 11. Cell elongation vs lateralization
        ax11 = fig.add_subplot(gs[2, 2])
        elongation_data = self.nuclei.dropna(subset=['cell_elongation', 'lateralization_index_composite'])
        if len(elongation_data) > 0:
            ax11.scatter(elongation_data['cell_elongation'], 
                        elongation_data['lateralization_index_composite'], 
                        alpha=0.6, color='brown')
            ax11.set_xlabel('Cell Elongation Factor')
            ax11.set_ylabel('Lateralization Index (%)')
            ax11.set_title('Cell Shape vs Lateralization')
            ax11.grid(alpha=0.3)
        
        # 12. Heterogeneity heatmap
        ax12 = fig.add_subplot(gs[2, 3])
        if 'grid_info' in self.results:
            grid_df = pd.DataFrame(self.results['grid_info'])
            if len(grid_df) > 0 and 'connexin_density' in grid_df.columns:
                # Create a simple heatmap representation
                x_coords = grid_df['x_center'].values
                y_coords = grid_df['y_center'].values
                densities = grid_df['connexin_density'].values
                
                scatter = ax12.scatter(x_coords, y_coords, c=densities, cmap='Reds', 
                                     s=50, alpha=0.7)
                ax12.set_xlabel('X Coordinate (μm)')
                ax12.set_ylabel('Y Coordinate (μm)')
                ax12.set_title('Spatial Heterogeneity Heatmap')
                plt.colorbar(scatter, ax=ax12, label='Connexin Density')
                ax12.set_aspect('equal', adjustable='box')
        
        # 13-16. Summary statistics panels
        ax13 = fig.add_subplot(gs[3, :])
        ax13.axis('off')
        
        # Create summary text
        summary_text = self.generate_summary_text()
        ax13.text(0.02, 0.98, summary_text, transform=ax13.transAxes, 
                 verticalalignment='top', fontfamily='monospace', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Comprehensive Connexin Analysis Results', fontsize=16, fontweight='bold')
        
        if save_plots:
            plt.savefig('advanced_connexin_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved comprehensive plots as 'advanced_connexin_analysis.png'")
        
        plt.show()
    
    def generate_summary_text(self):
        """Generate formatted summary text for visualization"""
        
        # Basic counts
        total_nuclei = len(self.nuclei)
        total_connexins = len(self.connexins)
        nuclei_with_connexins = np.sum(self.nuclei['connexin_count'] > 0)
        
        # Plaque metrics
        if len(self.connexins) > 0:
            mean_plaque_size = self.connexins['Nucleus: Area'].mean()
            median_plaque_size = self.connexins['Nucleus: Area'].median()
            std_plaque_size = self.connexins['Nucleus: Area'].std()
        else:
            mean_plaque_size = median_plaque_size = std_plaque_size = 0
        
        # Lateralization metrics
        valid_lat = self.nuclei['lateralization_index_composite'].dropna()
        if len(valid_lat) > 0:
            mean_lateralization = valid_lat.mean()
            high_lateralization = np.sum(valid_lat > 25)
        else:
            mean_lateralization = 0
            high_lateralization = 0
        
        # Remodeling metrics
        valid_remod = self.nuclei['remodeling_score_final'].dropna()
        if len(valid_remod) > 0:
            mean_remodeling = valid_remod.mean()
            severe_remodeling = np.sum(valid_remod > 1.0)
        else:
            mean_remodeling = 0
            severe_remodeling = 0
        
        # Heterogeneity metrics
        heterogeneity_info = ""
        if 'heterogeneity' in self.results:
            het_results = self.results['heterogeneity']
            connexin_het = het_results.get('connexin_count_heterogeneity', 0)
            density_het = het_results.get('connexin_density_heterogeneity', 0)
            heterogeneity_info = f"Connexin Count Heterogeneity: {connexin_het:.1f}%\n" \
                               f"Density Heterogeneity: {density_het:.1f}%\n"
        
        # Distance metrics
        valid_distances = self.connexins[self.connexins['distance_to_nucleus'] < np.inf]['distance_to_nucleus']
        if len(valid_distances) > 0:
            mean_distance = valid_distances.mean()
            std_distance = valid_distances.std()
        else:
            mean_distance = std_distance = 0
        
        summary_text = f"""CONNEXIN ANALYSIS SUMMARY
{'='*50}

BASIC COUNTS:
Total Cardiomyocyte Nuclei: {total_nuclei}
Total Connexin Objects: {total_connexins}
Nuclei with Connexins: {nuclei_with_connexins} ({nuclei_with_connexins/total_nuclei*100:.1f}%)
Average Connexins per Nucleus: {total_connexins/total_nuclei:.1f}

PLAQUE SIZE METRICS:
Mean Plaque Size: {mean_plaque_size:.3f} μm²
Median Plaque Size: {median_plaque_size:.3f} μm²
Std Dev Plaque Size: {std_plaque_size:.3f} μm²

LATERALIZATION METRICS:
Mean Lateralization Index: {mean_lateralization:.1f}%
Cells with High Lateralization (>25%): {high_lateralization} ({high_lateralization/len(valid_lat)*100:.1f}% of analyzed)

REMODELING METRICS:
Mean Remodeling Score: {mean_remodeling:.2f}
Cells with Severe Remodeling (>1.0): {severe_remodeling} ({severe_remodeling/len(valid_remod)*100:.1f}% of analyzed)

SPATIAL METRICS:
{heterogeneity_info}Mean Distance to Nucleus: {mean_distance:.1f} μm
Std Dev Distance: {std_distance:.1f} μm

DENSITY METRICS:
Mean Connexin Density: {self.nuclei['connexin_density'].mean():.4f} count/μm²
Max Connexin Density: {self.nuclei['connexin_density'].max():.4f} count/μm²
"""
        return summary_text
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print(f"\nGenerating comprehensive report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_file': self.csv_file,
                'software': 'Advanced Connexin Analyzer v1.0'
            },
            
            'data_summary': {
                'total_nuclei': int(len(self.nuclei)),
                'total_connexins': int(len(self.connexins)),
                'nuclei_with_connexins': int(np.sum(self.nuclei['connexin_count'] > 0)),
                'association_rate_percent': float(np.sum(self.connexins['associated_nucleus'] >= 0) / len(self.connexins) * 100) if len(self.connexins) > 0 else 0,
                'mean_connexins_per_nucleus': float(self.nuclei['connexin_count'].mean()),
                'median_connexins_per_nucleus': float(self.nuclei['connexin_count'].median())
            },
            
            'plaque_size_metrics': {
                'mean_plaque_size_um2': float(self.connexins['Nucleus: Area'].mean()) if len(self.connexins) > 0 else 0,
                'median_plaque_size_um2': float(self.connexins['Nucleus: Area'].median()) if len(self.connexins) > 0 else 0,
                'std_plaque_size_um2': float(self.connexins['Nucleus: Area'].std()) if len(self.connexins) > 0 else 0,
                'min_plaque_size_um2': float(self.connexins['Nucleus: Area'].min()) if len(self.connexins) > 0 else 0,
                'max_plaque_size_um2': float(self.connexins['Nucleus: Area'].max()) if len(self.connexins) > 0 else 0,
                'small_plaques_count': int(np.sum(self.connexins['Nucleus: Area'] < 0.5)) if len(self.connexins) > 0 else 0,
                'large_plaques_count': int(np.sum(self.connexins['Nucleus: Area'] > 2.0)) if len(self.connexins) > 0 else 0
            },
            
            'lateralization_metrics': {},
            'remodeling_metrics': {},
            'spatial_metrics': {},
            'distance_metrics': {}
        }
        
        # Lateralization metrics
        valid_lat = self.nuclei['lateralization_index_composite'].dropna()
        if len(valid_lat) > 0:
            report['lateralization_metrics'] = {
                'mean_lateralization_index': float(valid_lat.mean()),
                'median_lateralization_index': float(valid_lat.median()),
                'std_lateralization_index': float(valid_lat.std()),
                'cells_analyzed_for_lateralization': int(len(valid_lat)),
                'cells_with_low_lateralization_0_15': int(np.sum(valid_lat <= 15)),
                'cells_with_moderate_lateralization_15_30': int(np.sum((valid_lat > 15) & (valid_lat <= 30))),
                'cells_with_high_lateralization_30_plus': int(np.sum(valid_lat > 30)),
                'pathological_lateralization_threshold_25_percent': int(np.sum(valid_lat > 25))
            }
        
        # Remodeling metrics
        valid_remod = self.nuclei['remodeling_score_final'].dropna()
        if len(valid_remod) > 0:
            report['remodeling_metrics'] = {
                'mean_remodeling_score': float(valid_remod.mean()),
                'median_remodeling_score': float(valid_remod.median()),
                'std_remodeling_score': float(valid_remod.std()),
                'cells_analyzed_for_remodeling': int(len(valid_remod)),
                'cells_with_mild_remodeling': int(np.sum(valid_remod <= 0.5)),
                'cells_with_moderate_remodeling': int(np.sum((valid_remod > 0.5) & (valid_remod <= 1.0))),
                'cells_with_severe_remodeling': int(np.sum(valid_remod > 1.0))
            }
        
        # Spatial metrics
        if 'heterogeneity' in self.results:
            report['spatial_metrics'] = self.results['heterogeneity'].copy()
            report['spatial_metrics']['grid_squares_analyzed'] = self.results.get('n_grid_squares', 0)
        
        # Distance metrics
        valid_distances = self.connexins[self.connexins['distance_to_nucleus'] < np.inf]['distance_to_nucleus']
        if len(valid_distances) > 0:
            report['distance_metrics'] = {
                'mean_distance_to_nucleus_um': float(valid_distances.mean()),
                'median_distance_to_nucleus_um': float(valid_distances.median()),
                'std_distance_to_nucleus_um': float(valid_distances.std()),
                'min_distance_to_nucleus_um': float(valid_distances.min()),
                'max_distance_to_nucleus_um': float(valid_distances.max()),
                'connexins_within_5um': int(np.sum(valid_distances <= 5)),
                'connexins_within_10um': int(np.sum(valid_distances <= 10)),
                'connexins_within_15um': int(np.sum(valid_distances <= 15))
            }
        
        # Density metrics
        valid_density = self.nuclei[self.nuclei['connexin_density'] > 0]['connexin_density']
        if len(valid_density) > 0:
            report['density_metrics'] = {
                'mean_connexin_density_per_um2': float(valid_density.mean()),
                'median_connexin_density_per_um2': float(valid_density.median()),
                'std_connexin_density_per_um2': float(valid_density.std()),
                'max_connexin_density_per_um2': float(valid_density.max()),
                'cells_with_high_density': int(np.sum(valid_density > valid_density.mean() + valid_density.std())),
                'cells_with_low_density': int(np.sum(valid_density < valid_density.mean() - valid_density.std()))
            }
        
        # Save report
        report_filename = self.csv_file.replace('.csv', '_analysis_report.json')
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive report saved as '{report_filename}'")
        
        # Print key findings
        print("\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print("="*60)
        
        print(f"Analyzed {report['data_summary']['total_nuclei']} cardiomyocyte nuclei")
        print(f"Detected {report['data_summary']['total_connexins']} connexin objects")
        print(f"Association rate: {report['data_summary']['association_rate_percent']:.1f}%")
        
        if 'mean_plaque_size_um2' in report['plaque_size_metrics']:
            print(f"Mean plaque size: {report['plaque_size_metrics']['mean_plaque_size_um2']:.3f} μm²")
        
        if 'mean_lateralization_index' in report['lateralization_metrics']:
            lat_mean = report['lateralization_metrics']['mean_lateralization_index']
            high_lat = report['lateralization_metrics']['pathological_lateralization_threshold_25_percent']
            print(f"Mean lateralization: {lat_mean:.1f}%")
            print(f"Cells with pathological lateralization (>25%): {high_lat}")
        
        if 'mean_remodeling_score' in report['remodeling_metrics']:
            remod_mean = report['remodeling_metrics']['mean_remodeling_score']
            severe_remod = report['remodeling_metrics']['cells_with_severe_remodeling']
            print(f"Mean remodeling score: {remod_mean:.2f}")
            print(f"Cells with severe remodeling: {severe_remod}")
        
        return report
    
    def save_processed_data(self):
        """Save all processed data to CSV files"""
        print(f"\nSaving processed data...")
        
        # Save enhanced nuclei data
        nuclei_filename = self.csv_file.replace('.csv', '_processed_nuclei.csv')
        self.nuclei.to_csv(nuclei_filename, index=False)
        
        # Save enhanced connexin data
        connexin_filename = self.csv_file.replace('.csv', '_processed_connexins.csv')
        self.connexins.to_csv(connexin_filename, index=False)
        
        print(f"Saved processed data:")
        print(f"   Nuclei: {nuclei_filename}")
        print(f"   Connexins: {connexin_filename}")
        
        return nuclei_filename, connexin_filename

def main():
    """Main analysis workflow"""
    print("ADVANCED CONNEXIN ANALYSIS PIPELINE")
    print("="*60)
    
    # Get input file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter the path to your QuPath export CSV file: ")
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    try:
        # Initialize analyzer
        analyzer = AdvancedConnexinAnalyzer(csv_file)
        
        # Run complete analysis pipeline
        print("\nRunning complete analysis pipeline...")
        
        # Step 1: Associate connexins with nuclei
        association_rate = analyzer.associate_connexins_with_nuclei(max_distance=25.0)
        
        if association_rate < 50:
            print("Low association rate - consider increasing max_distance parameter")
        
        # Step 2: Calculate advanced lateralization
        analyzer.calculate_advanced_lateralization(
            cell_radius_multiplier=3.0,
            end_zone_fraction=0.15
        )
        
        # Step 3: Calculate spatial heterogeneity
        analyzer.calculate_spatial_heterogeneity(grid_size=50)
        
        # Step 4: Calculate remodeling scores
        analyzer.calculate_remodeling_scores()
        
        # Step 5: Create comprehensive visualizations
        analyzer.create_comprehensive_visualizations(save_plots=True)
        
        # Step 6: Generate comprehensive report
        analyzer.generate_comprehensive_report()
        
        # Step 7: Save processed data
        analyzer.save_processed_data()
        
        print("\nAdvanced analysis complete!")
        print("\nGenerated files:")
        print("  advanced_connexin_analysis.png - Comprehensive visualizations")
        print("  *_analysis_report.json - Detailed analysis report")
        print("  *_processed_nuclei.csv - Enhanced nuclei data")
        print("  *_processed_connexins.csv - Enhanced connexin data")
        
        print("\nInterpretation Guide:")
        print("  • Lateralization Index: <15% normal, 15-30% mild, >30% severe")
        print("  • Remodeling Score: <0.5 mild, 0.5-1.0 moderate, >1.0 severe")
        print("  • Plaque Size: 0.5-2.0 μm² typical range")
        print("  • Heterogeneity Index: <40% uniform, 40-80% moderate, >80% high")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
