#!/usr/bin/env python3
"""
Automated data processing pipeline for PathWild.

This script orchestrates the complete data processing workflow:
1. Process raw presence data files → presence points CSV
2. Generate absence data → combined presence/absence CSV
3. Integrate environmental features → integrated CSV
4. Analyze integrated features
5. Assess training readiness
6. Prepare training features → feature datasets (excludes metadata)

Usage:
    python scripts/run_data_pipeline.py [--dataset NAME] [--skip-steps STEP1,STEP2] [--force] [--workers N]
    
Examples:
    # Process all datasets
    python scripts/run_data_pipeline.py
    
    # Process specific dataset
    python scripts/run_data_pipeline.py --dataset north_bighorn
    
    # Skip specific steps (e.g., if already done)
    python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
    
    # Force regeneration of all features (even if placeholders don't exist)
    python scripts/run_data_pipeline.py --force
    
    # Process with specific number of workers (for parallel steps)
    python scripts/run_data_pipeline.py --force --workers 4
    
    # Test mode: Process only first 50 rows (creates test files)
    python scripts/run_data_pipeline.py --dataset north_bighorn --limit 50
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
from datetime import datetime, timedelta

# Helper function to find all datasets (matches integrate_environmental_features.py logic)
def _find_all_datasets(processed_dir: Path, test_only: bool = False) -> list[Path]:
    """
    Find all combined presence/absence dataset files.
    
    Args:
        processed_dir: Directory to search for dataset files
        test_only: If True, only return test files. If False, only return regular files.
    
    Returns:
        List of paths to dataset CSV files
    """
    all_files = list(processed_dir.glob('combined_*_presence_absence.csv'))
    if test_only:
        dataset_files = [f for f in all_files if '_test' in f.stem]
    else:
        dataset_files = [f for f in all_files if '_test' not in f.stem]
    return sorted(dataset_files)

# Configure logging - will be set up in main() after determining log file path
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None) -> None:
    """
    Set up logging to both console and file.
    
    Args:
        log_file: Optional path to log file. If None, only logs to console.
    """
    # Clear any existing handlers
    logger.handlers.clear()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")
    
    root_logger.setLevel(logging.INFO)


def cleanup_old_logs(logs_dir: Path, max_age_days: int = 3) -> None:
    """
    Delete log files older than specified number of days.
    
    Args:
        logs_dir: Directory containing log files
        max_age_days: Maximum age in days (default: 3 = 1 week)
    """
    if not logs_dir.exists():
        return
    
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in logs_dir.glob('*.log'):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                deleted_count += 1
        except Exception as e:
            # Use print instead of logger since logging may not be set up yet
            print(f"Warning: Failed to delete old log file {log_file}: {e}", file=sys.stderr)
    
    if deleted_count > 0:
        # Use print instead of logger since logging may not be set up yet
        print(f"Cleaned up {deleted_count} log file(s) older than {max_age_days} days")


def get_log_file_path(data_dir: Path, dataset_name: Optional[str] = None) -> Path:
    """
    Generate log file path based on dataset and timestamp.
    
    Args:
        data_dir: Base data directory
        dataset_name: Optional dataset name. If None, uses "all_datasets"
    
    Returns:
        Path to log file
    """
    logs_dir = data_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate dataset identifier
    if dataset_name:
        dataset_id = dataset_name
    else:
        dataset_id = 'all_datasets'
    
    # Create log file name: pipeline_<dataset>_<timestamp>.log
    log_filename = f"pipeline_{dataset_id}_{timestamp}.log"
    
    return logs_dir / log_filename


class PipelineStep:
    """Represents a single step in the data pipeline."""
    
    def __init__(
        self,
        name: str,
        description: str,
        script_path: Path,
        command_args: List[str],
        required_input: Optional[Path] = None,
        expected_output: Optional[Path] = None,
        check_output_exists: bool = True,
        expected_outputs: Optional[List[Path]] = None  # For steps with multiple outputs
    ):
        self.name = name
        self.description = description
        self.script_path = script_path
        self.command_args = command_args
        self.required_input = required_input
        self.expected_output = expected_output
        self.expected_outputs = expected_outputs  # List of expected output files (for "all datasets" mode)
        self.check_output_exists = check_output_exists
    
    def should_skip(self, skip_steps: List[str]) -> bool:
        """Check if this step should be skipped."""
        return self.name in skip_steps
    
    def can_run(self) -> bool:
        """Check if this step can run (script exists, inputs available)."""
        if not self.script_path.exists():
            logger.warning(f"  Script not found: {self.script_path}")
            return False
        
        if self.required_input and not self.required_input.exists():
            logger.warning(f"  Required input not found: {self.required_input}")
            return False
        
        return True
    
    def is_complete(self) -> bool:
        """Check if this step has already been completed."""
        if not self.check_output_exists:
            # For steps without output files (like analysis steps), check if input hasn't changed
            # by comparing input file modification time to a marker file
            if self.required_input and self.required_input.exists():
                marker_file = self.required_input.parent / f".{self.required_input.stem}.{self.name}.complete"
                if marker_file.exists():
                    # Check if input file has been modified since marker was created
                    input_mtime = self.required_input.stat().st_mtime
                    marker_mtime = marker_file.stat().st_mtime
                    if marker_mtime >= input_mtime:
                        # Input hasn't changed since last successful run
                        return True
            # For steps without required_input (like assess_readiness), always run
            # as they provide current status information
            return False
        
        # Check single expected output
        if self.expected_output and self.expected_output.exists():
            return True
        
        # Check multiple expected outputs (for "all datasets" mode)
        if self.expected_outputs:
            # All expected outputs must exist for step to be considered complete
            all_exist = all(output.exists() for output in self.expected_outputs)
            if all_exist:
                return True
        
        return False
    
    def run(self, force: bool = False) -> bool:
        """Run this pipeline step."""
        if self.is_complete() and not force:
            if self.expected_output:
                logger.info(f"  ✓ Step already complete: {self.expected_output}")
            elif self.expected_outputs:
                existing = [str(f.name) for f in self.expected_outputs if f.exists()]
                logger.info(f"  ✓ Step already complete: {len(existing)}/{len(self.expected_outputs)} output files exist")
            else:
                logger.info(f"  ✓ Step already complete")
            return True
        
        if not self.can_run():
            return False
        
        logger.info(f"  Running: {self.script_path.name}")
        logger.info(f"  Command: python {self.script_path} {' '.join(self.command_args)}")
        
        start_time = time.time()
        
        try:
            # Use Popen to capture output in real-time and log it
            process = subprocess.Popen(
                [sys.executable, str(self.script_path)] + self.command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Read output line by line and log it
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Remove trailing newline and log the line
                    line = line.rstrip('\n\r')
                    if line:  # Only log non-empty lines
                        logger.info(line)
            
            # Wait for process to complete
            returncode = process.wait()
            
            elapsed = time.time() - start_time
            
            if returncode == 0:
                logger.info(f"  ✓ Completed in {elapsed:.1f}s")
                # Create marker file for steps without output files (analysis steps)
                if not self.check_output_exists and self.required_input and self.required_input.exists():
                    marker_file = self.required_input.parent / f".{self.required_input.stem}.{self.name}.complete"
                    marker_file.touch()  # Create/update marker file
                return True
            else:
                logger.error(f"  ✗ Failed with return code {returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Failed after {elapsed:.1f}s: {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Unexpected error after {elapsed:.1f}s: {e}")
            return False


class DataPipeline:
    """Orchestrates the complete data processing pipeline."""
    
    def __init__(
        self,
        data_dir: Path = Path('data'),
        dataset_name: Optional[str] = None,
        skip_steps: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        workers: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.skip_steps = skip_steps or []
        self.force = force
        self.limit = limit
        self.workers = workers
        
        self.raw_dir = data_dir / 'raw'
        self.processed_dir = data_dir / 'processed'
        self.scripts_dir = Path(__file__).parent
        
        # Build pipeline steps
        self.steps = self._build_pipeline_steps()
    
    def check_prerequisites(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check for required environmental data files before running pipeline.
        
        Returns:
            Tuple of (all_required_present, list_of_missing_files)
        """
        missing_required = []
        missing_optional = []
        
        # Required raster files (essential for feature integration)
        required_rasters = {
            'DEM': self.data_dir / 'dem' / 'wyoming_dem.tif',
            'Slope': self.data_dir / 'terrain' / 'slope.tif',
            'Aspect': self.data_dir / 'terrain' / 'aspect.tif',
            'Land Cover': self.data_dir / 'landcover' / 'nlcd.tif',
            'Canopy Cover': self.data_dir / 'canopy' / 'canopy_cover.tif',
        }
        
        # Required vector files (essential for feature integration)
        required_vectors = {
            'Water Sources': self.data_dir / 'hydrology' / 'water_sources.geojson',
        }
        
        # Optional vector files (have defaults but improve data quality)
        optional_vectors = {
            'Roads': self.data_dir / 'infrastructure' / 'roads.geojson',
            'Trails': self.data_dir / 'infrastructure' / 'trails.geojson',
            'Wolf Packs': self.data_dir / 'wildlife' / 'wolf_packs.geojson',
            'Bear Activity': self.data_dir / 'wildlife' / 'bear_activity.geojson',
            'Hunt Areas': self.data_dir / 'hunt_areas' / 'hunt_areas.geojson',
        }
        
        # Check required files
        for name, path in required_rasters.items():
            if not path.exists():
                missing_required.append(f"{name}: {path}")
        
        for name, path in required_vectors.items():
            if not path.exists():
                missing_required.append(f"{name}: {path}")
        
        # Check optional files
        for name, path in optional_vectors.items():
            if not path.exists():
                missing_optional.append(f"{name}: {path}")
        
        return len(missing_required) == 0, missing_required, missing_optional
    
    def _build_pipeline_steps(self) -> List[PipelineStep]:
        """Build the list of pipeline steps."""
        steps = []
        
        # Step 1: Process raw presence data
        if self.dataset_name:
            presence_output = self.processed_dir / f"{self.dataset_name}_points.csv"
            steps.append(PipelineStep(
                name='process_raw',
                description='Process raw presence data files into presence points',
                script_path=self.scripts_dir / 'process_raw_presence_data.py',
                command_args=[
                    '--dataset', self.dataset_name,
                    '--input-dir', str(self.raw_dir),
                    '--output-dir', str(self.processed_dir)
                ],
                required_input=self.raw_dir / f"elk_{self.dataset_name}",
                expected_output=presence_output
            ))
        else:
            # Process all datasets - check if all expected output files exist
            # Infer expected outputs from existing combined files
            combined_files = _find_all_datasets(self.processed_dir)
            if combined_files:
                # Extract dataset names from combined files and check for corresponding points files
                expected_outputs = []
                for combined_file in combined_files:
                    # Extract name: combined_north_bighorn_presence_absence.csv -> north_bighorn
                    name = combined_file.stem.replace('combined_', '').replace('_presence_absence', '')
                    points_file = self.processed_dir / f"{name}_points.csv"
                    expected_outputs.append(points_file)
            else:
                # No combined files yet - check for all points files
                all_points_files = list(self.processed_dir.glob('*_points.csv'))
                expected_outputs = [
                    f for f in all_points_files 
                    if not f.stem.endswith('_test') and '_points' in f.stem
                ]
            
            steps.append(PipelineStep(
                name='process_raw',
                description='Process raw presence data files into presence points',
                script_path=self.scripts_dir / 'process_raw_presence_data.py',
                command_args=[
                    '--input-dir', str(self.raw_dir),
                    '--output-dir', str(self.processed_dir)
                ],
                required_input=self.raw_dir,
                expected_output=None,  # Multiple outputs
                expected_outputs=expected_outputs,  # List of expected files
                check_output_exists=True  # Check if all expected outputs exist
            ))
        
        # Step 2: Generate absence data
        if self.dataset_name:
            # Process single dataset
            presence_file = self.processed_dir / f"{self.dataset_name}_points.csv"
            # Always create the regular combined file (full dataset)
            # integrate_features will limit it and create the test file when limit is set
            combined_output = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
            
            command_args = [
                '--presence-file', str(presence_file),
                '--output-file', str(combined_output),
                '--data-dir', str(self.data_dir)
            ]
            
            # In test mode (limit set), limit the presence data input for absence generation
            # This prevents generating 8K+ absence points when we only need 15
            # NOTE: Only apply limit if explicitly set - don't limit when force is used without limit
            if self.limit is not None:
                command_args.extend(['--limit', str(self.limit)])
                logger.info(f"  ⚠️  TEST MODE: Limiting absence generation to {self.limit} presence points")
            if self.workers is not None:
                command_args.extend(['--n-processes', str(self.workers)])
            
            steps.append(PipelineStep(
                name='generate_absence',
                description='Generate absence data and combine with presence',
                script_path=self.scripts_dir / 'generate_absence_data.py',
                command_args=command_args,
                required_input=presence_file,
                expected_output=combined_output
            ))
        else:
            # Process all datasets - need to find all presence point files
            # This step will be handled by processing each dataset individually
            # For now, we'll skip it when processing all datasets and rely on existing combined files
            # Or we could loop through datasets, but that's complex. Let's make it explicit.
            # Note: Combined files should already exist from previous runs
            # We could add a step that loops through all datasets, but for now we'll skip
            pass  # Skip generate_absence when processing all datasets
        
        # Step 3: Integrate environmental features
        if self.dataset_name:
            # Process single dataset
            combined_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
            # In test mode (limit set), output goes to _test file
            if self.limit is not None:
                integrated_output = self.processed_dir / f"combined_{self.dataset_name}_presence_absence_test.csv"
            else:
                integrated_output = combined_file  # Overwrites input
            
            command_args = [
                str(combined_file),
                '--data-dir', str(self.data_dir)
            ]
            if self.force:
                command_args.append('--force')
            if self.limit is not None:
                command_args.extend(['--limit', str(self.limit)])
            if self.workers is not None:
                command_args.extend(['--workers', str(self.workers)])
            
            steps.append(PipelineStep(
                name='integrate_features',
                description='Integrate environmental features (elevation, water, landcover, etc.)',
                script_path=self.scripts_dir / 'integrate_environmental_features.py',
                command_args=command_args,
                required_input=combined_file,
                expected_output=integrated_output
            ))
        else:
            # Process all datasets - integrate_environmental_features.py now supports this!
            # When no dataset path is provided, it processes all combined_*_presence_absence.csv files
            # Check if all expected combined files exist (and have been integrated)
            # Use the same logic as integrate_environmental_features.py to find datasets
            all_combined_files = _find_all_datasets(self.processed_dir)
            expected_outputs = all_combined_files if all_combined_files else []
            
            command_args = [
                '--data-dir', str(self.data_dir),
                '--processed-dir', str(self.processed_dir)
            ]
            if self.force:
                command_args.append('--force')
            if self.limit is not None:
                command_args.extend(['--limit', str(self.limit)])
            if self.workers is not None:
                command_args.extend(['--workers', str(self.workers)])
            
            steps.append(PipelineStep(
                name='integrate_features',
                description='Integrate environmental features for all datasets (elevation, water, landcover, etc.)',
                script_path=self.scripts_dir / 'integrate_environmental_features.py',
                command_args=command_args,
                required_input=self.processed_dir,  # Needs processed_dir to find combined files
                expected_output=None,  # Multiple outputs
                expected_outputs=expected_outputs,  # List of expected integrated files
                check_output_exists=True  # Check if all expected outputs exist
            ))
        
        # Step 4: Analyze integrated features
        if self.dataset_name:
            # Process single dataset
            # In test mode, analyze the test file
            if self.limit is not None:
                integrated_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence_test.csv"
            else:
                integrated_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
            
            steps.append(PipelineStep(
                name='analyze_features',
                description='Analyze integrated environmental features',
                script_path=self.scripts_dir / 'analyze_integrated_features.py',
                command_args=[str(integrated_file)],
                required_input=integrated_file,
                expected_output=None,  # Analysis output to stdout
                check_output_exists=False
            ))
        else:
            # Process all datasets - find all combined files and analyze each
            # For now, skip this step when processing all datasets
            # (analyze_integrated_features.py doesn't support processing all at once)
            # Users can run analyze_integrated_features.py individually for each dataset
            pass  # Skip analyze_features when processing all datasets
        
        # Step 5: Assess training readiness
        # In test mode (limit set), assess the test file created by integrate_features
        assess_args = []
        if self.limit is not None:
            # Test mode: use --test-mode flag to prefer test files
            assess_args.append('--test-mode')
            if self.dataset_name:
                # Also explicitly pass the test file for single dataset mode
                test_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence_test.csv"
                assess_args.extend(['--dataset', str(test_file)])
        
        steps.append(PipelineStep(
            name='assess_readiness',
            description='Assess training data readiness for model training',
            script_path=self.scripts_dir / 'assess_training_readiness.py',
            command_args=assess_args,
            required_input=None,  # Reads from processed_dir or uses provided dataset
            expected_output=None,  # Assessment output to stdout
            check_output_exists=False
        ))
        
        # Step 6: Prepare training features (only if dataset specified)
        if self.dataset_name:
            # In test mode, use test file and create test output
            if self.limit is not None:
                combined_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence_test.csv"
                features_dir = self.data_dir / 'features'
                features_output = features_dir / f"{self.dataset_name}_features_test.csv"
            else:
                combined_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
                features_dir = self.data_dir / 'features'
                features_output = features_dir / f"{self.dataset_name}_features.csv"
            
            steps.append(PipelineStep(
                name='prepare_features',
                description='Prepare training-ready features by excluding metadata columns',
                script_path=self.scripts_dir / 'prepare_training_features.py',
                command_args=[
                    str(combined_file),
                    str(features_output)
                ],
                required_input=combined_file,
                expected_output=features_output
            ))
        else:
            # Prepare features for all datasets
            features_dir = self.data_dir / 'features'
            # Check if all expected feature files exist
            combined_files = _find_all_datasets(self.processed_dir)
            if combined_files:
                # Infer expected feature outputs from combined files
                expected_outputs = []
                for combined_file in combined_files:
                    # Extract name: combined_north_bighorn_presence_absence.csv -> north_bighorn
                    name = combined_file.stem.replace('combined_', '').replace('_presence_absence', '')
                    if self.limit is not None:
                        features_output = features_dir / f"{name}_features_test.csv"
                    else:
                        features_output = features_dir / f"{name}_features.csv"
                    expected_outputs.append(features_output)
            else:
                expected_outputs = []
            
            command_args = [
                '--all-datasets',
                '--processed-dir', str(self.processed_dir),
                '--features-dir', str(features_dir)
            ]
            if self.limit is not None:
                command_args.extend(['--limit', str(self.limit)])
            
            steps.append(PipelineStep(
                name='prepare_features',
                description='Prepare training-ready features by excluding metadata columns',
                script_path=self.scripts_dir / 'prepare_training_features.py',
                command_args=command_args,
                required_input=self.processed_dir,
                expected_output=None,  # Multiple outputs
                expected_outputs=expected_outputs,  # List of expected feature files
                check_output_exists=True  # Check if all expected outputs exist
            ))
        
        return steps
    
    def run(self) -> bool:
        """Run the complete pipeline."""
        logger.info("=" * 70)
        logger.info("PATHWILD DATA PROCESSING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Data directory: {self.data_dir}")
        if self.dataset_name:
            logger.info(f"Dataset: {self.dataset_name}")
        else:
            logger.info(f"Dataset: All datasets")
        logger.info(f"Force mode: {self.force}")
        if self.limit is not None:
            logger.info(f"⚠️  TEST MODE: Processing only first {self.limit:,} rows (will create test files)")
        if self.workers is not None:
            logger.info(f"Workers: {self.workers}")
        else:
            logger.info(f"Workers: auto-detect")
        if self.skip_steps:
            logger.info(f"Skipping steps: {', '.join(self.skip_steps)}")
        logger.info("")
        
        # Check prerequisites before starting
        logger.info("Checking prerequisites...")
        all_present, missing_required, missing_optional = self.check_prerequisites()
        
        if not all_present:
            logger.error("=" * 70)
            logger.error("PREREQUISITE CHECK FAILED")
            logger.error("=" * 70)
            logger.error("Missing required environmental data files:")
            for missing in missing_required:
                logger.error(f"  ✗ {missing}")
            
            if missing_optional:
                logger.warning("\nMissing optional files (will use defaults):")
                for missing in missing_optional:
                    logger.warning(f"  ⚠ {missing}")
            
            logger.error("\n" + "=" * 70)
            logger.error("Please generate the required prerequisites before running the pipeline.")
            logger.error("See docs/environmental_data_prerequisites.md for detailed instructions.")
            logger.error("=" * 70)
            return False
        
        if missing_optional:
            logger.warning("Optional files missing (will use defaults):")
            for missing in missing_optional:
                logger.warning(f"  ⚠ {missing}")
        
        logger.info("✓ All required prerequisites present")
        logger.info("")
        
        pipeline_start = time.time()
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for i, step in enumerate(self.steps, 1):
            logger.info(f"\n[{i}/{len(self.steps)}] {step.name.upper()}: {step.description}")
            logger.info("-" * 70)
            
            if step.should_skip(self.skip_steps):
                logger.info(f"  ⊘ Skipped (requested)")
                skip_count += 1
                continue
            
            # Check if step is already complete (before running)
            was_already_complete = step.is_complete() and not self.force
            
            if step.run(force=self.force):
                if was_already_complete:
                    # Step was skipped because it was already complete
                    skip_count += 1
                else:
                    # Step ran and completed successfully
                    success_count += 1
            else:
                fail_count += 1
                logger.error(f"  ✗ Pipeline step failed: {step.name}")
                # Continue processing other steps even if one fails
                # (allows pipeline to complete what it can)
        
        # Summary
        pipeline_elapsed = time.time() - pipeline_start
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {pipeline_elapsed/60:.1f} minutes")
        logger.info(f"Steps completed: {success_count}/{len(self.steps)}")
        logger.info(f"Steps skipped: {skip_count}")
        logger.info(f"Steps failed: {fail_count}")
        
        if fail_count == 0:
            logger.info("\n✓ Pipeline completed successfully!")
            return True
        else:
            logger.error("\n✗ Pipeline completed with errors")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete PathWild data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  python scripts/run_data_pipeline.py
  
  # Process specific dataset
  python scripts/run_data_pipeline.py --dataset north_bighorn
  
  # Skip steps that are already complete
  python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
  
  # Force regeneration of all features
  python scripts/run_data_pipeline.py --force
        """
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Base data directory (default: data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Specific dataset to process (e.g., "north_bighorn"). If not specified, processes all datasets.'
    )
    parser.add_argument(
        '--skip-steps',
        type=str,
        default=None,
        help='Comma-separated list of steps to skip (e.g., "process_raw,generate_absence")'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of all features, even if placeholders don\'t exist'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing). Creates test files with _test suffix instead of overwriting originals.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers to use for parallelizable steps (generate_absence, integrate_features). Default: auto-detect based on hardware.'
    )
    
    args = parser.parse_args()
    
    # Set up logging to file and console
    # Clean up old logs first (before setting up logging to avoid logging the cleanup)
    logs_dir = args.data_dir / 'logs'
    cleanup_old_logs(logs_dir, max_age_days=7)
    
    # Generate log file path and set up logging
    log_file = get_log_file_path(args.data_dir, args.dataset)
    setup_logging(log_file)
    
    skip_steps = []
    if args.skip_steps:
        skip_steps = [s.strip() for s in args.skip_steps.split(',')]
    
    pipeline = DataPipeline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        skip_steps=skip_steps,
        force=args.force,
        limit=args.limit,
        workers=args.workers
    )
    
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

