#!/usr/bin/env python3
"""
Connexin Data Analysis Module

"""

import logging
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnexinDataCompiler:
    """
    Handles compilation and analysis of connexin analysis results.
    """
    
    def __init__(self, settings_path: Optional[str] = None):
        """
        Initialize the data compiler.
        
        Args:
            settings_path: Path to settings JSON file
        """
        self.settings = self._load_settings(settings_path)
        self.results_data = []
        self.master_df = None
        self.compiled_df = None
        
    def _load_settings(self, settings_path: Optional[str] = None) -> Dict:
        """Load settings from JSON file."""
        if settings_path is None:
            # Look for settings file in same directory as script
            script_dir = Path(__file__).parent
            settings_path = script_dir / "IHC_settings.json"
        
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            logger.info(f"Loaded settings from {settings_path}")
            return settings
        except FileNotFoundError:
            logger.warning(f"Settings file not found: {settings_path}")
            logger.info("Using default settings")
            return self._get_default_settings()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing settings file: {e}")
            raise
    
    def _get_default_settings(self) -> Dict:
        """Return default settings."""
        return {
            "file_paths": {
                "master_filepath": "~/Desktop/IHC_Master_File.csv",
                "results_filepath": "~/Desktop/IHC_Master_File_Results.csv"
            },
            "output_settings": {
                "suppress_console_output": False
            }
        }
    
    def select_directory(self, title: str = "Select Analysis Results Directory") -> Optional[str]:
        """
        Open directory selection dialog.
        
        Args:
            title: Dialog title
            
        Returns:
            Selected directory path or None if cancelled
        """
        root = tk.Tk()
        root.withdraw()
        
        # Use default directory from settings if available
        initial_dir = self.settings.get("file_paths", {}).get("default_input_directory")
        if initial_dir:
            initial_dir = Path(initial_dir).expanduser()
        
        try:
            directory_path = filedialog.askdirectory(
                title=title,
                initialdir=initial_dir
            )
            
            if directory_path:
                logger.info(f"Selected directory: {directory_path}")
                return directory_path
            else:
                logger.info("No directory selected")
                return None
                
        except Exception as e:
            logger.error(f"Error in directory selection: {e}")
            return None
        finally:
            root.destroy()

    def select_merged_file(self):
        root = tk.Tk()
        root.withdraw()

        # Use default directory from settings if available
        initial_dir = self.settings.get("file_paths", {}).get("default_input_directory")
        if initial_dir:
            initial_dir = Path(initial_dir).expanduser()

        try:
            merged_file_path = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Select Pre-Existing Results File or Initial File"
            )
            if merged_file_path:
                logger.info(f"Selected Results File: {merged_file_path}")
                return merged_file_path
            else:
                logger.info("No directory selected")
                return None

        except Exception as e:
            logger.error(f"Error in directory selection: {e}")
            return None
        finally:
            root.destroy()
    
    def find_analysis_reports(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find all analysis report JSON files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of paths to analysis report files
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all analysis report files
        pattern = "*_analysis_report.json"
        files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(files)} analysis report files in {directory}")
        
        if not files:
            logger.warning(f"No analysis report files found matching pattern: {pattern}")
        
        return files
    
    def load_analysis_report(self, file_path: Path) -> Optional[Dict]:
        """
        Load a single analysis report JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data dictionary or None if error
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            if 'analysis_metadata' not in data:
                logger.warning(f"Missing analysis_metadata in {file_path}")
                return None
            
            if 'source_file' not in data['analysis_metadata']:
                logger.warning(f"Missing source_file in metadata for {file_path}")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def extract_slide_name(self, source_file: str) -> str:
        """
        Extract slide name from source file path.
        
        Args:
            source_file: Source file path from analysis metadata
            
        Returns:
            Extracted slide name
        """
        # Extract filename and remove -Detection.csv suffix
        filename = Path(source_file).name
        
        if filename.endswith('-Detection.csv'):
            # Extract slide name from filename
            # Expected format: something-Subject-SampleX-Detection.csv
            parts = filename[:-len('-Detection.csv')].split('-')
            if len(parts) >= 3:
                # Take the last 2 parts as Subject-Sample
                slide_name = '-'.join(parts[1:3])
            else:
                # Fallback: remove Detection.csv suffix
                slide_name = filename[:-len('-Detection.csv')]
        else:
            # Fallback: remove any .csv extension
            slide_name = filename.replace('.csv', '')
        
        return slide_name
    
    def flatten_analysis_data(self, data: Dict, slide_name: str) -> Dict:
        """
        Flatten nested analysis data into a single dictionary.
        
        Args:
            data: Analysis data dictionary
            slide_name: Slide identifier
            
        Returns:
            Flattened dictionary with slide_name as key
        """
        flattened = {'slide_name': slide_name}
        
        # Flatten all nested dictionaries
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    # Skip metadata fields that aren't numerical
                    if key in ['timestamp', 'source_file', 'software']:
                        continue
                    
                    # Use section prefix for clarity, maintaining total_ and subsample_ distinctions
                    if section_name != 'analysis_metadata':
                        if section_name == 'key_metrics':
                            # Keep key_metrics fields as-is (they already have total_ prefix)
                            flattened[key] = value
                        else:
                            # Add section prefix for other sections
                            flattened_key = f"{section_name}_{key}"
                            flattened[flattened_key] = value
                    else:
                        # Skip metadata except for important fields
                        if key not in ['timestamp', 'source_file', 'software']:
                            flattened[key] = value
        
        return flattened
    
    def load_all_results(self, directory: Union[str, Path]) -> List[Dict]:
        """
        Load all analysis results from directory.
        
        Args:
            directory: Directory containing analysis report files
            
        Returns:
            List of flattened result dictionaries
        """
        files = self.find_analysis_reports(directory)
        results = []
        
        for file_path in files:
            data = self.load_analysis_report(file_path)
            if data is None:
                continue
            
            slide_name = self.extract_slide_name(data['analysis_metadata']['source_file'])
            flattened = self.flatten_analysis_data(data, slide_name)
            results.append(flattened)
        
        logger.info(f"Successfully loaded {len(results)} analysis results")
        return results
    
    def create_results_dataframe(self, results_data: List[Dict]) -> pd.DataFrame:
        """
        Create DataFrame from results data.
        
        Args:
            results_data: List of flattened result dictionaries
            
        Returns:
            Compiled results DataFrame
        """
        if not results_data:
            logger.warning("No results data to compile")
            return pd.DataFrame()
        
        df = pd.DataFrame(results_data)
        
        # Remove any duplicate entries
        initial_count = len(df)
        df = df.drop_duplicates(subset=['slide_name'], keep='last')
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate entries")
        
        # Parse slide name components (Perform after have all data once at end!)
        #df = self._parse_slide_components(df)
        
        logger.info(f"Created results DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _parse_slide_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse slide name into components (subject, sample, section).
        
        Args:
            df: DataFrame with slide_name column
            
        Returns:
            DataFrame with additional parsed columns
        """
        if 'slide_name' not in df.columns:
            logger.warning("No slide_name column found for parsing")
            return df
        
        try:
            # Parse slide name components
            # Expected format: Subject-SampleX where X is a letter/number
            slide_parts = df['slide_name'].str.split('-')
            
            # Extract subject (first part)
            df['subject'] = slide_parts.str[0]
            
            # Extract sample (second part if exists)
            df['sample'] = slide_parts.str[1].fillna('')
            
            # Extract section (remove last character from sample if it exists)
            df['section'] =  df['sample'].str.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            
            # Convert subject to numeric where possible
            df['subject'] = pd.to_numeric(df['subject'])
            df['section'] = pd.to_numeric(df['section'])

            # Add analysis of location, spacing
            df['location'] = df.apply(lambda row: 'apex' if (row['study'] == 'sbrt' and int(row['section']) <= 4) else ('base' if (row['study'] == 'sbrt' and int(row['section']) >= 10) else ('mid' if (row['study'] == 'sbrt') else "")), axis=1)
            spacing_map = {'sbrt': 3, 'kpi': 5}
            df['spacing'] = df['study'].map(spacing_map).fillna('')

            logger.info("Successfully parsed slide name components")
            
        except Exception as e:
            logger.warning(f"Error parsing slide components: {e}")
            # Add empty columns if parsing fails
            if 'subject' not in df.columns:
                df['subject'] = ''
            if 'sample' not in df.columns:
                df['sample'] = ''
            if 'section' not in df.columns:
                df['section'] = ''
        
        return df
    
    def load_master_file(self) -> Optional[pd.DataFrame]:
        """
        Load master experiment file.
        
        Returns:
            Master DataFrame or None if not found
        """
        master_path = self.settings.get("file_paths", {}).get("master_filepath")
        
        if not master_path:
            logger.warning("No master file path specified in settings")
            return None
        
        master_path = Path(master_path).expanduser()
        
        try:
            df = pd.read_csv(master_path)
            logger.info(f"Loaded master file with {len(df)} rows from {master_path}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"Master file not found: {master_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading master file: {e}")
            return None
    
    def merge_with_master(self, results_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge results with master experiment data.
        
        Args:
            results_df: Results DataFrame
            master_df: Master experiment DataFrame
            
        Returns:
            Merged DataFrame
        """
        if results_df.empty or master_df.empty:
            logger.warning("Cannot merge: one or both DataFrames are empty")
            return results_df if not results_df.empty else master_df
        
        # Ensure both have slide_name column
        if 'slide_name' not in results_df.columns:
            logger.error("Results DataFrame missing slide_name column")
            return results_df
        
        if 'slide_name' not in master_df.columns:
            logger.warning("Master DataFrame missing slide_name column, performing outer join on all columns")
            return pd.concat([master_df, results_df], ignore_index=True, sort=False)
        
        # Perform merge
        merged_df = pd.merge(master_df, results_df, on='slide_name', how='outer')
        
        # Parse components for the merged DataFrame if not already done
        #if 'subject' not in merged_df.columns:
        #    merged_df = self._parse_slide_components(merged_df)
        
        logger.info(f"Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df.drop(columns=master_df.filter(regex='Unnamed').columns)
    
    def save_results(self, df: pd.DataFrame, output_path: Union[str, Path], 
                    description: str = "results") -> bool:
        """
        Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            description: Description for logging
            
        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"Cannot save empty DataFrame for {description}")
            return False
        
        output_path = Path(output_path).expanduser()
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp
            df = df.drop(columns=df.filter(regex='Unnamed').columns)
            df = df.drop_duplicates()
            df.to_csv(output_path, index=True)
            logger.info(f"Saved {description} to {output_path} ({len(df)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {description}: {e}")
            return False
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform statistical analysis on the compiled data.
        
        Args:
            df: Data DataFrame
            
        Returns:
            Dictionary containing analysis results
        """
        if df.empty:
            logger.warning("Cannot perform analysis on empty DataFrame")
            return {}
        
        logger.info("Performing statistical analysis...")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'descriptive_stats': {},
            'grouped_stats': {}
        }
        
        # Select numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns found for statistical analysis")
            return analysis_results
        
        # Overall descriptive statistics
        try:
            desc_stats = df[numeric_cols].describe()
            # Convert to dict, handling nested structure
            analysis_results['descriptive_stats'] = {}
            for col in desc_stats.columns:
                analysis_results['descriptive_stats'][col] = desc_stats[col].to_dict()
        except Exception as e:
            logger.error(f"Error in descriptive statistics: {e}")
        
        # Group by subject if available
        if 'subject' in df.columns and len(df['subject'].dropna()) > 0:
            try:
                grouped_stats = df.groupby('subject')[numeric_cols].describe()
                # Convert to nested dict structure
                analysis_results['grouped_stats']['by_subject'] = {}
                for subject in grouped_stats.index.get_level_values(0).unique():
                    analysis_results['grouped_stats']['by_subject'][str(subject)] = {}
                    for col in numeric_cols:
                        if (subject, col) in grouped_stats.index:
                            analysis_results['grouped_stats']['by_subject'][str(subject)][col] = grouped_stats.loc[subject, col].to_dict()
            except Exception as e:
                logger.error(f"Error in subject grouping: {e}")
        
        # Group by section if available
        if 'section' in df.columns and len(df['section'].dropna()) > 0:
            try:
                section_stats = df.groupby('section')[numeric_cols].describe()
                # Convert to nested dict structure
                analysis_results['grouped_stats']['by_section'] = {}
                for section in section_stats.index.get_level_values(0).unique():
                    if section and str(section).strip():  # Skip empty sections
                        analysis_results['grouped_stats']['by_section'][str(section)] = {}
                        for col in numeric_cols:
                            if (section, col) in section_stats.index:
                                analysis_results['grouped_stats']['by_section'][str(section)][col] = section_stats.loc[section, col].to_dict()
            except Exception as e:
                logger.error(f"Error in section grouping: {e}")
        
        logger.info("Statistical analysis completed")
        return analysis_results
    
    def print_summary(self, df: pd.DataFrame, stats: Dict) -> None:
        """
        Print analysis summary to console.
        
        Args:
            df: Analysis DataFrame
            stats: Statistical analysis results
        """
        if self.settings.get("output_settings", {}).get("suppress_console_output", False):
            return
        
        print("\n" + "="*80)
        print("CONNEXIN ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"Total samples analyzed: {len(df)}")
        
        if 'subject' in df.columns:
            subjects = df['subject'].nunique()
            print(f"Number of subjects: {subjects}")
        
        if 'section' in df.columns:
            sections = df['section'].nunique()
            print(f"Number of sections: {sections}")
        
        # Print key metrics if available (prioritizing total_ metrics)
        key_metrics = [
            'total_nuclei', 'total_connexins', 'total_mean_plaque_size',
            'total_connexin_count_per_cell', 'total_connexins_per_annotation_area'
        ]
        
        # Also check for subsample metrics as fallback
        subsample_metrics = [
            'subsample_mean_lateralization_index', 'subsample_mean_remodeling_score'
        ]
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        available_subsample = [m for m in subsample_metrics if m in df.columns]
        
        if available_metrics or available_subsample:
            print(f"\nKey Metrics Summary:")
            
            # Print total metrics first
            for metric in available_metrics:
                try:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")
                except:
                    continue
            
            # Then print subsample metrics
            for metric in available_subsample:
                try:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")
                except:
                    continue

    def check_results_file(self, merged_file_path: Path) -> None:
        """
        Check if analysis results file exists.
        :return:
        """
        #results_path = self.settings.get("file_paths", {}).get("results_filepath")
        if os.path.exists(merged_file_path):
            try:
                merged_df = pd.read_csv(merged_file_path)
                pattern = r'total'
                existing_cols = []
                for col in merged_df.columns:
                    if re.search(pattern, col):
                        existing_cols.append(col)
                if existing_cols:
                    return True
                else:
                    return False
            except Exception as e:
                return False

    def load_results_df(self):
        """
        Load analysis results dataframe.
        :return:
        """
        results_path = self.settings.get("file_paths", {}).get("results_filepath")
        results_df = pd.read_csv(results_path)
        return results_df

    def update_master(self, compiled_df, master_df):
        """
        Update analysis results dataframe.
        :param compiled_df:
        :param master_df:
        :return:
        """
        if compiled_df.empty or master_df.empty:
            logger.warning("Cannot Update: one or both DataFrames are empty")
            return compiled_df if not compiled_df.empty else master_df

        # Ensure both have slide_name column
        if 'slide_name' not in compiled_df.columns:
            logger.error("Results DataFrame missing slide_name column")
            return compiled_df

        if 'slide_name' not in master_df.columns:
            logger.warning("Master DataFrame missing slide_name column, performing outer join on all columns")
            return pd.concat([master_df, compiled_df], ignore_index=True, sort=False)

        # Perform update
        common_slides = set(master_df['slide_name']) & set(compiled_df['slide_name'])
        new_slides = set(compiled_df['slide_name']).difference(set(master_df['slide_name']))
        master_cols = set(master_df.columns)
        compiled_cols = set(compiled_df.columns)
        new_cols = compiled_cols - master_cols
        if new_cols:
            for col in new_cols:
                master_df[col] = np.nan
        for slide_name in common_slides:
            idx = master_df[master_df['slide_name'] == slide_name].index
            compiled_row = compiled_df[compiled_df['slide_name'] == slide_name]
            cols_to_update = [col for col in compiled_df.columns if col != 'slide_name']
            for col in cols_to_update:
                master_df.loc[idx, col] = compiled_row[col].iloc[0]
        for slide_name in new_slides:
            compiled_row = compiled_df[compiled_df['slide_name'] == slide_name]
            master_df = pd.concat([master_df, compiled_row])

        master_df = master_df.drop(columns=master_df.filter(regex='Unnamed').columns)
        master_df = master_df.drop_duplicates()

        # Parse components for the merged DataFrame if not already done
        if 'subject' not in master_df.columns:
            master_df = self._parse_slide_components(master_df)

        logger.info(f"Merged data: {len(master_df)} rows, {len(master_df.columns)} columns")
        return master_df
    
    def run_analysis(self, directory: Optional[str] = None) -> bool:
        """
        Run complete analysis pipeline.
        
        Args:
            directory: Input directory (will prompt if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get input directory
            if directory is None:
                directory = self.select_directory()
                if directory is None:
                    logger.info("Analysis cancelled by user")
                    return False
            
            # Load analysis results
            self.results_data = self.load_all_results(directory)
            if not self.results_data:
                logger.error("No analysis results found")
                return False
            
            # Create results DataFrame
            self.compiled_df = self.create_results_dataframe(self.results_data)
            if self.compiled_df.empty:
                logger.error("Failed to create results DataFrame")
                return False

            # Select Results File or Master File
            merged_file_path = self.select_merged_file()

            # Check if results file already exists and can update it:
            if self.check_results_file(merged_file_path):
                logging.info("Existing analysis results file selected. Updating Results.")
                self.master_df = pd.read_csv(merged_file_path) # Load results df
                #self.master_df = self.load_results_df()
                merged_df =  self.update_master(self.compiled_df, self.master_df) # Update Results df
            else:
                logging.info("Analysis results file not selected. Using Master File.")
                #self.master_df = self.load_master_file() # Load Master file
                self.master_df = pd.read_csv(merged_file_path)
                if self.master_df is not None:
                    merged_df = self.merge_with_master(self.compiled_df, self.master_df) # Merge with Master file
                else:
                    merged_df = self.compiled_df # Create new file from compiled if absent

            # Parse Slide Components
            merged_df = self._parse_slide_components(merged_df)

            # Save results
            results_dir = Path(directory)
            
            # Save compiled results
            compiled_path = results_dir / "compiled_connexin_analysis_results.csv"
            self.save_results(self.compiled_df, compiled_path, "compiled results")
            
            # Save merged results if master file was loaded
            #if self.master_df is not None:
            #    results_path = self.settings.get("file_paths", {}).get("results_filepath")
            #    if results_path:
            #        self.save_results(merged_df, results_path, "merged results")
            # Save Results
            if self.check_results_file(merged_file_path):
                self.save_results(merged_df, merged_file_path, "merged results")
            else:
                self.save_results(merged_df, results_dir / "IHC_Results.csv", "merged results")
            
            # Perform statistical analysis
            #stats = self.perform_statistical_analysis(merged_df)
            
            # Save statistics
            #stats_path = results_dir / "statistical_analysis.json"
            #try:
            #    with open(stats_path, 'w') as f:
            #        json.dump(stats, f, indent=2, default=str)
            #    logger.info(f"Saved statistical analysis to {stats_path}")
            #except Exception as e:
            #    logger.error(f"Error saving statistics: {e}")
            
            # Print summary
            #self.print_summary(merged_df, stats)
            
            logger.info("Analysis pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            return False


def main():
    """Main function for standalone execution."""
    print("CONNEXIN DATA ANALYSIS PIPELINE")
    print("="*50)
    
    try:
        compiler = ConnexinDataCompiler()
        success = compiler.run_analysis()
        
        if success:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()