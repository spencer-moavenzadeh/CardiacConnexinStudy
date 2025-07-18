#!/usr/bin/env python3
"""
Connexin Pipeline Control
"""

import logging
import json
import os
import sys
import traceback
import re
from pathlib import Path
from typing import List, Optional, Dict, Union
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our analysis modules
try:
    import advanced_connexin_analysis_batched
    import connexin_data_analysis
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Please ensure all required modules are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('connexin_analysis_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def analyze_single_file_wrapper(detection_file_path: str) -> Dict:
    """
    Wrapper function for multiprocessing analysis of a single file.
    
    Args:
        detection_file_path: Path to Detection.csv file as string
        
    Returns:
        Dictionary with analysis results
    """
    detection_file = Path(detection_file_path)
    result = {
        'file': detection_file,
        'success': False,
        'error': None,
        'start_time': datetime.now(),
        'end_time': None
    }
    
    try:
        logger.info(f"Starting analysis of {detection_file.name}")
        
        # Call the analysis function
        advanced_connexin_analysis_batched.main(str(detection_file))
        
        result['success'] = True
        logger.info(f"Successfully analyzed {detection_file.name}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to analyze {detection_file.name}: {e}")
        logger.debug(traceback.format_exc())
    
    finally:
        result['end_time'] = datetime.now()
        result['duration'] = result['end_time'] - result['start_time']
    
    return result


class ConnexinAnalysisPipeline:
    """
    Main pipeline controller for connexin analysis workflow.
    """
    
    def __init__(self, settings_path: Optional[str] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            settings_path: Path to settings JSON file
        """
        self.settings = self._load_settings(settings_path)
        self.input_directory = None
        self.detection_files = []
        self.failed_files = []
        self.successful_files = []
        
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
            
            # Validate settings structure
            self._validate_settings(settings)
            return settings
            
        except FileNotFoundError:
            logger.error(f"Settings file not found: {settings_path}")
            self._show_error_dialog(
                "Settings File Missing",
                f"Could not find settings file: {settings_path}\n\n"
                "Please ensure IHC_settings.json is in the same directory as this script."
            )
            raise
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing settings file: {e}")
            self._show_error_dialog(
                "Settings File Error",
                f"Error parsing settings file: {e}"
            )
            raise
    
    def _validate_settings(self, settings: Dict) -> None:
        """Validate settings structure."""
        required_sections = ['analysis_parameters', 'output_settings', 'file_paths']
        
        for section in required_sections:
            if section not in settings:
                logger.warning(f"Missing settings section: {section}")
    
    def _show_error_dialog(self, title: str, message: str) -> None:
        """Show error dialog to user."""
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(title, message)
            root.destroy()
        except:
            # Fallback to console if GUI not available
            print(f"ERROR - {title}: {message}")
    
    def _show_info_dialog(self, title: str, message: str) -> None:
        """Show info dialog to user."""
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
        except:
            # Fallback to console if GUI not available
            print(f"INFO - {title}: {message}")
    
    def select_input_directory(self) -> bool:
        """
        Prompt user to select input directory containing QuPath exports.
        
        Returns:
            True if directory selected, False if cancelled
        """
        root = tk.Tk()
        root.withdraw()
        
        # Get default directory from settings
        default_dir = self.settings.get("file_paths", {}).get("default_input_directory")
        if default_dir:
            default_dir = Path(default_dir).expanduser()
        
        try:
            directory = filedialog.askdirectory(
                title="Select Directory Containing QuPath Exports",
                initialdir=default_dir,
            )
            
            if directory:
                self.input_directory = Path(directory)
                logger.info(f"Selected input directory: {self.input_directory}")
                return True
            else:
                logger.info("No directory selected")
                return False
                
        except Exception as e:
            logger.error(f"Error in directory selection: {e}")
            self._show_error_dialog("Directory Selection Error", str(e))
            return False
        finally:
            root.destroy()
    
    def discover_detection_files(self) -> bool:
        """
        Find all Detection.csv files in the input directory.
        
        Returns:
            True if files found, False otherwise
        """
        if not self.input_directory or not self.input_directory.exists():
            logger.error("Invalid input directory")
            return False
        
        # Find Detection.csv files
        pattern = "*Detection.csv"
        self.detection_files = list(self.input_directory.glob(pattern))
        
        logger.info(f"Found {len(self.detection_files)} Detection.csv files")
        
        if not self.detection_files:
            message = f"No Detection.csv files found in {self.input_directory}"
            logger.warning(message)
            self._show_error_dialog("No Files Found", message)
            return False
        
        # Log file names
        for file_path in self.detection_files:
            logger.info(f"  - {file_path.name}")
        
        return True
    
    def validate_file_pairs(self) -> bool:
        """
        Validate that each Detection.csv has a corresponding Annotation.csv.
        
        Returns:
            True if all pairs exist, False otherwise
        """
        missing_annotations = []
        
        for detection_file in self.detection_files:
            # Expected annotation file name
            annotation_file = detection_file.parent / detection_file.name.replace(
                "Detection.csv", "Annotation.csv"
            )
            
            if not annotation_file.exists():
                missing_annotations.append(annotation_file.name)
                logger.warning(f"Missing annotation file: {annotation_file}")
        
        if missing_annotations:
            message = (
                f"Missing {len(missing_annotations)} annotation files:\n"
                f"{', '.join(missing_annotations)}\n\n"
                "Each Detection.csv file must have a corresponding Annotation.csv file."
            )
            self._show_error_dialog("Missing Annotation Files", message)
            return False
        
        logger.info("All Detection-Annotation file pairs validated")
        return True
    
    def analyze_single_file(self, detection_file: Path) -> Dict:
        """
        Analyze a single detection file.
        
        Args:
            detection_file: Path to Detection.csv file
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'file': detection_file,
            'success': False,
            'error': None,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        try:
            logger.info(f"Starting analysis of {detection_file.name}")
            
            # Call the analysis function
            advanced_connexin_analysis_batched.main(str(detection_file))
            
            result['success'] = True
            logger.info(f"Successfully analyzed {detection_file.name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to analyze {detection_file.name}: {e}")
            logger.debug(traceback.format_exc())
        
        finally:
            result['end_time'] = datetime.now()
            result['duration'] = result['end_time'] - result['start_time']
        
        return result
    
    def run_batch_analysis(self, parallel: bool = True, max_workers: Optional[int] = None) -> bool:
        """
        Run analysis on all detection files.
        
        Args:
            parallel: Whether to run analyses in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            True if at least one analysis succeeded, False otherwise
        """
        if not self.detection_files:
            logger.error("No detection files to analyze")
            return False
        
        logger.info(f"Starting batch analysis of {len(self.detection_files)} files")
        
        if parallel and len(self.detection_files) > 1:
            return self._run_parallel_analysis(max_workers)
        else:
            return self._run_sequential_analysis()
    
    def _run_sequential_analysis(self) -> bool:
        """Run analysis sequentially."""
        logger.info("Running sequential analysis")
        
        for i, detection_file in enumerate(self.detection_files, 1):
            logger.info(f"Processing file {i}/{len(self.detection_files)}: {detection_file.name}")
            
            result = self.analyze_single_file(detection_file)
            
            if result['success']:
                self.successful_files.append(result)
            else:
                self.failed_files.append(result)
        
        return len(self.successful_files) > 0
    
    def _run_parallel_analysis(self, max_workers: Optional[int] = None) -> bool:
        """Run analysis in parallel."""
        if max_workers is None:
            max_workers = min(len(self.detection_files), mp.cpu_count() - 1, 4)
        
        logger.info(f"Running parallel analysis with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks using string paths for multiprocessing
            future_to_file = {
                executor.submit(analyze_single_file_wrapper, str(file)): file 
                for file in self.detection_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        self.successful_files.append(result)
                        logger.info(f"✓ Completed: {file_path.name}")
                    else:
                        self.failed_files.append(result)
                        logger.error(f"✗ Failed: {file_path.name}")
                        
                except Exception as e:
                    error_result = {
                        'file': file_path,
                        'success': False,
                        'error': f"Execution error: {str(e)}",
                        'start_time': datetime.now(),
                        'end_time': datetime.now()
                    }
                    self.failed_files.append(error_result)
                    logger.error(f"✗ Exception in {file_path.name}: {e}")
        
        return len(self.successful_files) > 0
    
    def compile_results(self) -> bool:
        """
        Compile analysis results using the data analysis module.
        
        Returns:
            True if compilation succeeded, False otherwise
        """
        if not self.successful_files:
            logger.error("No successful analyses to compile")
            return False
        
        logger.info("Compiling analysis results...")
        
        try:
            # Initialize the data compiler
            compiler = connexin_data_analysis.ConnexinDataCompiler()
            
            # Run the compilation analysis
            success = compiler.run_analysis(str(self.input_directory))
            
            if success:
                logger.info("Results compilation completed successfully")
                return True
            else:
                logger.error("Results compilation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in results compilation: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report of the pipeline execution.
        
        Returns:
            Dictionary containing summary information
        """
        total_files = len(self.detection_files)
        successful_count = len(self.successful_files)
        failed_count = len(self.failed_files)
        
        # Calculate processing times
        if self.successful_files:
            durations = [r['duration'].total_seconds() for r in self.successful_files if 'duration' in r]
            avg_duration = sum(durations) / len(durations) if durations else 0
            total_duration = sum(durations) if durations else 0
        else:
            avg_duration = total_duration = 0
        
        summary = {
            'pipeline_metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(self.input_directory),
                'settings_file': 'IHC_settings.json'
            },
            'processing_summary': {
                'total_files': total_files,
                'successful_analyses': successful_count,
                'failed_analyses': failed_count,
                'success_rate_percent': (successful_count / total_files * 100) if total_files > 0 else 0,
                'average_processing_time_seconds': avg_duration,
                'total_processing_time_seconds': total_duration
            },
            'successful_files': [str(r['file'].name) for r in self.successful_files],
            'failed_files': [
                {
                    'file': str(r['file'].name),
                    'error': r['error']
                } for r in self.failed_files
            ]
        }
        
        return summary
    
    def save_summary_report(self, summary: Dict) -> bool:
        """
        Save summary report to JSON file.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            output_path = self.input_directory / "pipeline_summary_report.json"
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Saved pipeline summary to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
            return False
    
    def print_summary(self, summary: Dict) -> None:
        """
        Print pipeline summary to console.
        
        Args:
            summary: Summary dictionary
        """
        suppress_output = self.settings.get("output_settings", {}).get("suppress_console_output", False)
        
        if suppress_output:
            return
        
        print("\n" + "="*80)
        print("CONNEXIN ANALYSIS PIPELINE SUMMARY")
        print("="*80)
        
        proc_summary = summary['processing_summary']
        
        print(f"Input Directory: {summary['pipeline_metadata']['input_directory']}")
        print(f"Analysis Date: {summary['pipeline_metadata']['timestamp']}")
        print()
        print(f"Total Files: {proc_summary['total_files']}")
        print(f"Successful Analyses: {proc_summary['successful_analyses']}")
        print(f"Failed Analyses: {proc_summary['failed_analyses']}")
        print(f"Success Rate: {proc_summary['success_rate_percent']:.1f}%")
        print(f"Average Processing Time: {proc_summary['average_processing_time_seconds']:.1f} seconds")
        print(f"Total Processing Time: {proc_summary['total_processing_time_seconds']:.1f} seconds")
        
        if summary['failed_files']:
            print("\nFailed Files:")
            for failed in summary['failed_files']:
                print(f"  - {failed['file']}: {failed['error']}")
        
        print("\n" + "="*80)
    
    def show_completion_dialog(self, summary: Dict) -> None:
        """
        Show completion dialog to user.
        
        Args:
            summary: Summary dictionary
        """
        proc_summary = summary['processing_summary']
        
        if proc_summary['failed_analyses'] == 0:
            title = "Analysis Complete - Success!"
            message = (
                f"Successfully analyzed all {proc_summary['total_files']} files.\n\n"
                f"Average processing time: {proc_summary['average_processing_time_seconds']:.1f} seconds\n"
                f"Results compiled and saved to input directory."
            )
            self._show_info_dialog(title, message)
        else:
            title = "Analysis Complete - With Errors"
            message = (
                f"Completed analysis with {proc_summary['successful_analyses']}/{proc_summary['total_files']} successful.\n\n"
                f"Success Rate: {proc_summary['success_rate_percent']:.1f}%\n"
                f"Check logs for details on failed analyses."
            )
            self._show_error_dialog(title, message)
    
    def run_pipeline(self) -> bool:
        """
        Run the complete analysis pipeline.
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            logger.info("Starting Connexin Analysis Pipeline")
            
            # Step 1: Select input directory
            if not self.select_input_directory():
                logger.info("Pipeline cancelled by user")
                return False
            
            # Step 2: Discover detection files
            if not self.discover_detection_files():
                return False
            
            # Step 3: Validate file pairs
            if not self.validate_file_pairs():
                return False
            
            # Step 4: Run batch analysis
            analysis_success = self.run_batch_analysis(
                parallel=True,  # Can be made configurable
                max_workers=None
            )
            
            if not analysis_success:
                logger.error("All individual analyses failed")
                return False
            
            # Step 5: Compile results
            compilation_success = self.compile_results()
            
            # Step 6: Generate and save summary
            summary = self.generate_summary_report()
            self.save_summary_report(summary)
            self.print_summary(summary)
            
            # Step 7: Show completion dialog
            self.show_completion_dialog(summary)
            
            # Determine overall success
            overall_success = (
                len(self.successful_files) > 0 and 
                compilation_success
            )
            
            if overall_success:
                logger.info("Pipeline completed successfully")
            else:
                logger.warning("Pipeline completed with issues")
            
            return overall_success
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}")
            logger.debug(traceback.format_exc())
            self._show_error_dialog("Pipeline Error", str(e))
            return False


def setup_gui_error_handling():
    """Setup global error handling for GUI applications."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred:\n{exc_type.__name__}: {exc_value}\n\n"
                "Check the log file for more details."
            )
            root.destroy()
        except:
            pass
    
    sys.excepthook = handle_exception


def main():
    """Main entry point for the pipeline."""
    # Setup error handling
    setup_gui_error_handling()
    
    print("CONNEXIN ANALYSIS PIPELINE")
    print("="*50)
    print("Initializing...")
    
    try:
        # Create and run pipeline
        pipeline = ConnexinAnalysisPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline completed with errors!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
        