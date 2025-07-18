#!/usr/bin/env python3
"""
Complete Connexin Analysis in Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import json
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ConnexinAnalyzer:
    def __init__(self, cell_file="cell_data.csv", connexin_file="connexin_data.csv", pixel_size=0.25):
        """
        Initialize the connexin analyzer
        
        Parameters:
        cell_file: Path to cell data CSV
        connexin_file: Path to connexin data CSV  
        pixel_size: Pixel size in micrometers (check from QuPath)
        """
        self.pixel_size = pixel_size
        self.cell_data = None
        self.connexin_data = None
        self.results = {}
        
        # Load data
        self.load_data(cell_file, connexin_file)
        
    def load_data(self, cell_file, connexin_file):
        """Load data from CSV files"""
        try:
            self.cell_data = pd.read_csv(cell_file)
            self.connexin_data = pd.read_csv(connexin_file)
            print(f"Loaded {len(self.cell_data)} cells and {len(self.connexin_data)} connexin objects")
            
            # Basic data info
            print(f"Cell area range: {self.cell_data['area_um2'].min():.1f} - {self.cell_data['area_um2'].max():.1f} μm²")
            print(f"Connexin area range: {self.connexin_data['area_um2'].min():.3f} - {self.connexin_data['area_um2'].max():.3f} μm²")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Make sure you've exported data from QuPath first!")
            raise
            
    def associate_connexins_with_cells(self, max_distance=15.0):
        """
        Associate each connexin with the nearest cell
        
        Parameters:
        max_distance: Maximum distance in micrometers for association
        """
        print(f"\nAssociating connexins with cells (max distance: {max_distance} μm)...")
        
        # Get coordinates
        cell_coords = self.cell_data[['centroid_x', 'centroid_y']].values
        connexin_coords = self.connexin_data[['centroid_x', 'centroid_y']].values
        
        # Calculate distances between all connexins and all cells
        distances = cdist(connexin_coords, cell_coords) * self.pixel_size
        
        # Find nearest cell for each connexin
        nearest_cell_idx = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        # Only associate if within max distance
        valid_associations = min_distances <= max_distance
        
        # Add association info to connexin data
        self.connexin_data['associated_cell'] = -1  # -1 means no association
        self.connexin_data['distance_to_cell'] = np.inf
        
        self.connexin_data.loc[valid_associations, 'associated_cell'] = nearest_cell_idx[valid_associations]
        self.connexin_data.loc[valid_associations, 'distance_to_cell'] = min_distances[valid_associations]
        
        # Calculate connexin counts per cell
        cell_connexin_counts = self.connexin_data[valid_associations].groupby('associated_cell').size()
        
        # Add connexin metrics to cell data
        self.cell_data['connexin_count'] = 0
        self.cell_data['total_connexin_area_um2'] = 0.0
        self.cell_data['avg_plaque_size_um2'] = 0.0
        self.cell_data['connexin_density_per_um2'] = 0.0
        
        for cell_idx, count in cell_connexin_counts.items():
            if cell_idx < len(self.cell_data):
                cell_connexins = self.connexin_data[self.connexin_data['associated_cell'] == cell_idx]
                total_area = cell_connexins['area_um2'].sum()
                avg_area = cell_connexins['area_um2'].mean()
                cell_area = self.cell_data.loc[cell_idx, 'area_um2']
                
                self.cell_data.loc[cell_idx, 'connexin_count'] = count
                self.cell_data.loc[cell_idx, 'total_connexin_area_um2'] = total_area
                self.cell_data.loc[cell_idx, 'avg_plaque_size_um2'] = avg_area
                self.cell_data.loc[cell_idx, 'connexin_density_per_um2'] = count / cell_area
        
        associated_count = np.sum(valid_associations)
        association_rate = associated_count / len(self.connexin_data) * 100
        
        print(f"Associated {associated_count}/{len(self.connexin_data)} connexins ({association_rate:.1f}%)")
        print(f"Cells with connexins: {np.sum(self.cell_data['connexin_count'] > 0)}/{len(self.cell_data)}")
        
        return association_rate
        
    def calculate_lateralization_index(self, end_zone_fraction=0.2):
        """
        Calculate lateralization index for each cell
        
        Parameters:
        end_zone_fraction: Fraction of cell length to consider as end zones (intercalated discs)
        """
        print(f"\nCalculating lateralization indices (end zone fraction: {end_zone_fraction})...")
        
        self.cell_data['lateralization_index'] = 0.0
        self.cell_data['end_zone_connexins'] = 0
        self.cell_data['lateral_connexins'] = 0
        
        for cell_idx, cell in self.cell_data.iterrows():
            # Get connexins for this cell
            cell_connexins = self.connexin_data[self.connexin_data['associated_cell'] == cell_idx]
            
            if len(cell_connexins) == 0:
                continue
                
            # Estimate cell orientation (simplified - assume elongated cells)
            # In reality, you might want to fit an ellipse to the cell boundary
            cell_x, cell_y = cell['centroid_x'], cell['centroid_y']
            
            # For now, assume cells are oriented along their major axis
            # Calculate distances from cell center for each connexin
            connexin_coords = cell_connexins[['centroid_x', 'centroid_y']].values
            
            if len(connexin_coords) == 0:
                continue
                
            # Calculate relative positions
            rel_x = connexin_coords[:, 0] - cell_x
            rel_y = connexin_coords[:, 1] - cell_y
            
            # Estimate cell length as range of connexin positions (simplified)
            x_range = np.max(rel_x) - np.min(rel_x)
            y_range = np.max(rel_y) - np.min(rel_y)
            
            # Determine major axis
            if x_range > y_range:
                # Cell is horizontally oriented
                distances_along_axis = rel_x
                cell_length = x_range
            else:
                # Cell is vertically oriented  
                distances_along_axis = rel_y
                cell_length = y_range
                
            if cell_length == 0:
                continue
                
            # Define end zones
            end_zone_size = cell_length * end_zone_fraction / 2
            min_pos = np.min(distances_along_axis)
            max_pos = np.max(distances_along_axis)
            
            # Classify connexins
            end_zone_connexins = 0
            lateral_connexins = 0
            
            for pos in distances_along_axis:
                if pos <= (min_pos + end_zone_size) or pos >= (max_pos - end_zone_size):
                    end_zone_connexins += 1
                else:
                    lateral_connexins += 1
                    
            total_connexins = end_zone_connexins + lateral_connexins
            lateralization_index = (lateral_connexins / total_connexins * 100) if total_connexins > 0 else 0
            
            self.cell_data.loc[cell_idx, 'lateralization_index'] = lateralization_index
            self.cell_data.loc[cell_idx, 'end_zone_connexins'] = end_zone_connexins
            self.cell_data.loc[cell_idx, 'lateral_connexins'] = lateral_connexins
            
        print(f"Calculated lateralization for {np.sum(self.cell_data['lateralization_index'] > 0)} cells")
        print(f"Mean lateralization index: {self.cell_data['lateralization_index'].mean():.1f}%")
        
    def calculate_spatial_heterogeneity(self, grid_size=100):
        """
        Calculate spatial heterogeneity using grid-based analysis
        
        Parameters:
        grid_size: Size of grid squares in pixels
        """
        print(f"\nCalculating spatial heterogeneity (grid size: {grid_size} pixels)...")
        
        # Get image bounds
        min_x = min(self.cell_data['centroid_x'].min(), self.connexin_data['centroid_x'].min())
        max_x = max(self.cell_data['centroid_x'].max(), self.connexin_data['centroid_x'].max())
        min_y = min(self.cell_data['centroid_y'].min(), self.connexin_data['centroid_y'].min())
        max_y = max(self.cell_data['centroid_y'].max(), self.connexin_data['centroid_y'].max())
        
        # Create grid
        x_bins = np.arange(min_x, max_x + grid_size, grid_size)
        y_bins = np.arange(min_y, max_y + grid_size, grid_size)
        
        # Assign grid coordinates
        connexin_grid_x = np.digitize(self.connexin_data['centroid_x'], x_bins)
        connexin_grid_y = np.digitize(self.connexin_data['centroid_y'], y_bins)
        
        # Count connexins per grid square
        grid_counts = {}
        for i, (gx, gy) in enumerate(zip(connexin_grid_x, connexin_grid_y)):
            key = (gx, gy)
            grid_counts[key] = grid_counts.get(key, 0) + 1
            
        # Calculate statistics
        counts = list(grid_counts.values())
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            heterogeneity_index = (std_count / mean_count * 100) if mean_count > 0 else 0
        else:
            heterogeneity_index = 0
            
        self.results['heterogeneity_index'] = heterogeneity_index
        self.results['grid_mean_count'] = mean_count if counts else 0
        self.results['grid_std_count'] = std_count if counts else 0
        self.results['n_grid_squares'] = len(counts)
        
        print(f"Heterogeneity index: {heterogeneity_index:.1f}%")
        print(f"Grid squares with connexins: {len(counts)}")
        
        return heterogeneity_index
        
    def calculate_remodeling_score(self):
        """Calculate composite remodeling score using PCA-like approach"""
        print(f"\nCalculating remodeling scores...")
        
        # Select metrics for remodeling score
        metrics = ['lateralization_index', 'avg_plaque_size_um2', 'connexin_density_per_um2']
        
        # Filter cells with all metrics available
        valid_cells = self.cell_data.dropna(subset=metrics)
        
        if len(valid_cells) < 3:
            print("Not enough cells with complete data for remodeling score")
            return
            
        # Standardize metrics (z-scores)
        standardized = {}
        for metric in metrics:
            values = valid_cells[metric]
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val > 0:
                standardized[metric] = (values - mean_val) / std_val
            else:
                standardized[metric] = np.zeros(len(values))
                
        # Calculate composite score
        # Higher lateralization = more remodeling (positive weight)
        # Smaller plaque size = more remodeling (negative weight)  
        # Lower density = more remodeling (negative weight)
        
        weights = {
            'lateralization_index': 0.4,
            'avg_plaque_size_um2': -0.3,  # Negative: smaller = more remodeling
            'connexin_density_per_um2': -0.3  # Negative: lower = more remodeling
        }
        
        remodeling_scores = np.zeros(len(valid_cells))
        for metric, weight in weights.items():
            remodeling_scores += standardized[metric].values * weight
            
        # Add remodeling scores to cell data
        self.cell_data['remodeling_score'] = np.nan
        self.cell_data.loc[valid_cells.index, 'remodeling_score'] = remodeling_scores
        
        print(f"Calculated remodeling scores for {len(valid_cells)} cells")
        print(f"Mean remodeling score: {remodeling_scores.mean():.2f} ± {remodeling_scores.std():.2f}")
        
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print(f"\nCreating visualizations...")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Connexin Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Plaque size distribution
        axes[0, 0].hist(self.connexin_data['area_um2'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Plaque Area (μm²)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Gap Junction Plaque Size Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Lateralization distribution
        valid_lat = self.cell_data['lateralization_index'].dropna()
        if len(valid_lat) > 0:
            axes[0, 1].hist(valid_lat, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('Lateralization Index (%)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Connexin Lateralization Distribution')
            axes[0, 1].grid(alpha=0.3)
        
        # 3. Spatial distribution
        scatter = axes[0, 2].scatter(self.connexin_data['centroid_x'], self.connexin_data['centroid_y'], 
                                   c=self.connexin_data['area_um2'], cmap='viridis', alpha=0.6, s=20)
        axes[0, 2].set_xlabel('X Coordinate (pixels)')
        axes[0, 2].set_ylabel('Y Coordinate (pixels)')
        axes[0, 2].set_title('Spatial Distribution of Connexins')
        plt.colorbar(scatter, ax=axes[0, 2], label='Plaque Area (μm²)')
        
        # 4. Connexin count per cell
        axes[1, 0].hist(self.cell_data['connexin_count'], bins=range(0, int(self.cell_data['connexin_count'].max()) + 2), 
                       alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Connexins per Cell')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Connexin Count Distribution')
        axes[1, 0].grid(alpha=0.3)
        
        # 5. Remodeling score distribution
        valid_remod = self.cell_data['remodeling_score'].dropna()
        if len(valid_remod) > 0:
            axes[1, 1].hist(valid_remod, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('Remodeling Score')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Remodeling Score Distribution')
            axes[1, 1].grid(alpha=0.3)
        
        # 6. Correlation plot
        if len(valid_lat) > 0 and len(valid_remod) > 0:
            common_cells = self.cell_data.dropna(subset=['lateralization_index', 'remodeling_score'])
            if len(common_cells) > 0:
                axes[1, 2].scatter(common_cells['lateralization_index'], common_cells['remodeling_score'], 
                                 alpha=0.6, color='purple')
                axes[1, 2].set_xlabel('Lateralization Index (%)')
                axes[1, 2].set_ylabel('Remodeling Score')
                axes[1, 2].set_title('Lateralization vs Remodeling')
                axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('connexin_analysis_results.png', dpi=300, bbox_inches='tight')
            print("Saved plots as 'connexin_analysis_results.png'")
            
        plt.show()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\nGenerating summary report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_cells': len(self.cell_data),
                'total_connexins': len(self.connexin_data),
                'cells_with_connexins': int(np.sum(self.cell_data['connexin_count'] > 0)),
                'association_rate': f"{np.sum(self.connexin_data['associated_cell'] >= 0) / len(self.connexin_data) * 100:.1f}%"
            },
            'plaque_metrics': {
                'mean_plaque_size_um2': float(self.connexin_data['area_um2'].mean()),
                'median_plaque_size_um2': float(self.connexin_data['area_um2'].median()),
                'std_plaque_size_um2': float(self.connexin_data['area_um2'].std()),
                'plaque_size_range_um2': [float(self.connexin_data['area_um2'].min()), 
                                        float(self.connexin_data['area_um2'].max())]
            },
            'cellular_metrics': {
                'mean_connexins_per_cell': float(self.cell_data['connexin_count'].mean()),
                'median_connexins_per_cell': float(self.cell_data['connexin_count'].median()),
                'mean_cell_area_um2': float(self.cell_data['area_um2'].mean()),
                'connexin_density_range': [float(self.cell_data['connexin_density_per_um2'].min()),
                                         float(self.cell_data['connexin_density_per_um2'].max())]
            }
        }
        
        # Add lateralization metrics if available
        valid_lat = self.cell_data['lateralization_index'].dropna()
        if len(valid_lat) > 0:
            report['lateralization_metrics'] = {
                'mean_lateralization_index': float(valid_lat.mean()),
                'median_lateralization_index': float(valid_lat.median()),
                'std_lateralization_index': float(valid_lat.std()),
                'cells_with_high_lateralization': int(np.sum(valid_lat > 25))  # >25% considered high
            }
        
        # Add spatial metrics if available
        if 'heterogeneity_index' in self.results:
            report['spatial_metrics'] = {
                'heterogeneity_index': self.results['heterogeneity_index'],
                'grid_squares_analyzed': self.results['n_grid_squares']
            }
        
        # Add remodeling metrics if available
        valid_remod = self.cell_data['remodeling_score'].dropna()
        if len(valid_remod) > 0:
            report['remodeling_metrics'] = {
                'mean_remodeling_score': float(valid_remod.mean()),
                'std_remodeling_score': float(valid_remod.std()),
                'cells_with_high_remodeling': int(np.sum(valid_remod > 1))  # >1 considered high
            }
        
        # Save report
        with open('connexin_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("Saved report as 'connexin_analysis_report.json'")
        
        # Print summary to console
        print("\n" + "="*60)
        print("CONNEXIN ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total cells: {report['data_summary']['total_cells']}")
        print(f"Total connexins: {report['data_summary']['total_connexins']}")
        print(f"Cells with connexins: {report['data_summary']['cells_with_connexins']}")
        print(f"Mean plaque size: {report['plaque_metrics']['mean_plaque_size_um2']:.3f} μm²")
        print(f"Mean connexins per cell: {report['cellular_metrics']['mean_connexins_per_cell']:.1f}")
        
        if 'lateralization_metrics' in report:
            print(f"Mean lateralization: {report['lateralization_metrics']['mean_lateralization_index']:.1f}%")
            
        if 'spatial_metrics' in report:
            print(f"Heterogeneity index: {report['spatial_metrics']['heterogeneity_index']:.1f}%")
            
        return report
        
    def save_processed_data(self):
        """Save all processed data to CSV files"""
        print(f"\nSaving processed data...")
        
        # Save enhanced cell data
        self.cell_data.to_csv('processed_cell_data.csv', index=False)
        
        # Save enhanced connexin data  
        self.connexin_data.to_csv('processed_connexin_data.csv', index=False)
        
        print("Saved processed data:")
        print("   - processed_cell_data.csv")
        print("   - processed_connexin_data.csv")

def main():
    """Main analysis workflow"""
    print("CONNEXIN ANALYSIS PIPELINE")
    print("="*50)
    
    flag = True
    while flag: 
    	cell_filepath = input("Enter Cell Filepath: ")
    	connexin_filepath = input("Enter Connexin Filepath: ")
    	pixel_size = input("Enter Pixel Size (Default = 0.1659): ")
    	if pixel_size = "":
    	    pixel_size = 0.1659
    	if os.path.exists(cell_filepath) & os.path.exists(connexin_filepath):
    	    flag = False   
    
    # Initialize analyzer
    analyzer = ConnexinAnalyzer(pixel_size=0.1659, cell_file=cell_filepath, connexin_file=connexin_filepath)  # Adjust pixel size as needed
    
    # Run complete analysis
    try:
        # Associate connexins with cells
        analyzer.associate_connexins_with_cells(max_distance=15.0)
        
        # Calculate lateralization
        analyzer.calculate_lateralization_index(end_zone_fraction=0.2)
        
        # Calculate spatial heterogeneity
        analyzer.calculate_spatial_heterogeneity(grid_size=100)
        
        # Calculate remodeling scores
        analyzer.calculate_remodeling_score()
        
        # Create visualizations
        analyzer.create_visualizations(save_plots=True)
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        # Save processed data
        analyzer.save_processed_data()
        
        print("\nAnalysis complete!")
        print("Check the generated files:")
        print("  - connexin_analysis_results.png")
        print("  - connexin_analysis_report.json")
        print("  - processed_cell_data.csv")
        print("  - processed_connexin_data.csv")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
