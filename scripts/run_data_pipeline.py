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
    python scripts/run_data_pipeline.py [--dataset NAME] [--skip-steps STEP1,STEP2] [--force]
    
Examples:
    # Process all datasets
    python scripts/run_data_pipeline.py
    
    # Process specific dataset
    python scripts/run_data_pipeline.py --dataset north_bighorn
    
    # Skip specific steps (e.g., if already done)
    python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
    
    # Force regeneration of all features (even if placeholders don't exist)
    python scripts/run_data_pipeline.py --force
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        check_output_exists: bool = True
    ):
        self.name = name
        self.description = description
        self.script_path = script_path
        self.command_args = command_args
        self.required_input = required_input
        self.expected_output = expected_output
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
            return False
        
        if self.expected_output and self.expected_output.exists():
            return True
        
        return False
    
    def run(self, force: bool = False) -> bool:
        """Run this pipeline step."""
        if self.is_complete() and not force:
            logger.info(f"  ✓ Step already complete: {self.expected_output}")
            return True
        
        if not self.can_run():
            return False
        
        logger.info(f"  Running: {self.script_path.name}")
        logger.info(f"  Command: python {self.script_path} {' '.join(self.command_args)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.script_path)] + self.command_args,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"  ✓ Completed in {elapsed:.1f}s")
                return True
            else:
                logger.error(f"  ✗ Failed with return code {result.returncode}")
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
        force: bool = False
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.skip_steps = skip_steps or []
        self.force = force
        
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
            # Process all datasets
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
                check_output_exists=False
            ))
        
        # Step 2: Generate absence data (only if dataset specified)
        if self.dataset_name:
            presence_file = self.processed_dir / f"{self.dataset_name}_points.csv"
            combined_output = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
            
            steps.append(PipelineStep(
                name='generate_absence',
                description='Generate absence data and combine with presence',
                script_path=self.scripts_dir / 'generate_absence_data.py',
                command_args=[
                    '--presence-file', str(presence_file),
                    '--output-file', str(combined_output),
                    '--data-dir', str(self.data_dir)
                ],
                required_input=presence_file,
                expected_output=combined_output
            ))
        
        # Step 3: Integrate environmental features
        if self.dataset_name:
            combined_file = self.processed_dir / f"combined_{self.dataset_name}_presence_absence.csv"
            integrated_output = combined_file  # Overwrites input
            
            steps.append(PipelineStep(
                name='integrate_features',
                description='Integrate environmental features (elevation, water, landcover, etc.)',
                script_path=self.scripts_dir / 'integrate_environmental_features.py',
                command_args=[
                    str(combined_file),
                    '--data-dir', str(self.data_dir)
                ] + (['--force'] if self.force else []),
                required_input=combined_file,
                expected_output=integrated_output
            ))
        
        # Step 4: Analyze integrated features
        if self.dataset_name:
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
        
        # Step 5: Assess training readiness
        steps.append(PipelineStep(
            name='assess_readiness',
            description='Assess training data readiness for model training',
            script_path=self.scripts_dir / 'assess_training_readiness.py',
            command_args=[],
            required_input=None,  # Reads from processed_dir
            expected_output=None,  # Assessment output to stdout
            check_output_exists=False
        ))
        
        # Step 6: Prepare training features (only if dataset specified)
        if self.dataset_name:
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
            steps.append(PipelineStep(
                name='prepare_features',
                description='Prepare training-ready features by excluding metadata columns',
                script_path=self.scripts_dir / 'prepare_training_features.py',
                command_args=[
                    '--all-datasets',
                    '--processed-dir', str(self.processed_dir),
                    '--features-dir', str(features_dir)
                ],
                required_input=self.processed_dir,
                expected_output=None,  # Multiple outputs
                check_output_exists=False
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
            
            if step.run(force=self.force):
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
    
    args = parser.parse_args()
    
    skip_steps = []
    if args.skip_steps:
        skip_steps = [s.strip() for s in args.skip_steps.split(',')]
    
    pipeline = DataPipeline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        skip_steps=skip_steps,
        force=args.force
    )
    
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

