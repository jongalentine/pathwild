#!/usr/bin/env python3
"""
Add environmental features to existing presence/absence datasets.

This script updates existing CSV files with real environmental features
from DEM, water sources, land cover, etc. It replaces placeholder values
with actual data from environmental datasets.

Usage:
    # First, activate the conda environment:
    conda activate pathwild
    
    # Then run the script:
    python scripts/integrate_environmental_features.py [dataset_path] [--workers N] [--batch-size N] [--limit N]
    
Examples:
    # Process ALL datasets in data/processed/ (finds all combined_*_presence_absence.csv files)
    python scripts/integrate_environmental_features.py
    
    # Process specific dataset (auto-detects optimal workers and batch size based on hardware)
    python scripts/integrate_environmental_features.py data/processed/combined_northern_bighorn_presence_absence.csv
    
    # Process with specific number of workers
    python scripts/integrate_environmental_features.py data/processed/combined_national_refuge_presence_absence.csv --workers 4
    
    # Force sequential processing (no parallelization)
    python scripts/integrate_environmental_features.py data/processed/dataset.csv --workers 1
    
    # Test on first 100 rows (saves to *_test.csv, doesn't overwrite original)
    python scripts/integrate_environmental_features.py data/processed/combined_national_refuge_presence_absence.csv --limit 100
    
    # Process with custom batch size (auto-detects workers)
    python scripts/integrate_environmental_features.py data/processed/dataset.csv --batch-size 500
"""

import argparse
import logging
import sys
import os

# Check for required dependencies early with helpful error message
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.", file=sys.stderr)
    print("\nThis script requires the 'pathwild' conda environment.", file=sys.stderr)
    print("Please activate it first:", file=sys.stderr)
    print("  conda activate pathwild", file=sys.stderr)
    print("\nOr install pandas in your current environment:", file=sys.stderr)
    print("  pip install pandas", file=sys.stderr)
    sys.exit(1)

from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial

# Optional psutil for hardware detection
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress function
    def tqdm(iterable, desc=""):
        return iterable

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processors import DataContextBuilder

# Configure logging (must be before any logger usage)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            CONFIG = yaml.safe_load(f)
    else:
        CONFIG = {}
except ImportError:
    CONFIG = {}
    logger.warning("PyYAML not available, GEE config will not be loaded")


# Placeholder values that indicate data needs to be replaced
PLACEHOLDER_VALUES = {
    # Static features
    'elevation': 8500.0,
    'slope_degrees': 15.0,
    'aspect_degrees': 180.0,
    'canopy_cover_percent': 30.0,
    'water_distance_miles': 0.5,
    'water_reliability': 0.5,
    'land_cover_code': 0,
    'road_distance_miles': 10.0,
    'trail_distance_miles': 10.0,
    'security_habitat_percent': 0.5,
    # Temporal features (SNOTEL, weather, satellite)
    'snow_depth_inches': 0.0,  # Will be replaced with real SNOTEL data
    'snow_water_equiv_inches': 0.0,
    'snow_crust_detected': False,
    'temperature_f': 45.0,
    'precip_last_7_days_inches': 0.0,
    'cloud_cover_percent': 20,
    'ndvi': 0.5,
    'ndvi_age_days': 8,
    'irg': 0.0,
    'summer_integrated_ndvi': 0.0
}


def has_placeholder_values(row, env_columns, tolerance: float = 0.01) -> bool:
    """
    Check if a row contains placeholder values or missing data.
    
    Data quality tracking columns (snow_data_source, snow_station_name, etc.)
    are excluded from placeholder detection as they're metadata, not features.
    
    Args:
        row: pandas Series representing a row
        env_columns: List of environmental column names to check
        tolerance: Floating point tolerance for comparison
    
    Returns:
        True if row has placeholders or missing data, False otherwise
    """
    import numpy as np
    
    # Columns that are metadata/data quality tracking, not features with placeholders
    metadata_columns = {'snow_data_source', 'snow_station_name', 'snow_station_distance_km'}
    
    for col in env_columns:
        # Skip metadata columns - they don't have placeholder values
        if col in metadata_columns:
            continue
        
        # Check if column is missing
        if col not in row.index:
            return True
        
        # Check for NaN
        # Note: NaN can mean either "not processed yet" or "outside Wyoming bounds"
        # Both cases should be processed (outside bounds will get NaN again, which is correct)
        if pd.isna(row[col]):
            return True
        
        # Check for placeholder value
        if col in PLACEHOLDER_VALUES:
            placeholder = PLACEHOLDER_VALUES[col]
            value = row[col]
            
            # For numeric values, use tolerance
            if isinstance(value, (int, float)) and isinstance(placeholder, (int, float)):
                if abs(value - placeholder) <= tolerance:
                    return True
            elif value == placeholder:
                return True
    
    return False


def detect_optimal_workers(dataset_size: int = None) -> int:
    """
    Auto-detect optimal number of workers based on hardware.
    
    Args:
        dataset_size: Number of rows in dataset (optional, for size-based tuning)
    
    Returns:
        Optimal number of workers
    """
    try:
        # Get CPU count
        cpu_count = os.cpu_count() or 4
        
        if HAS_PSUTIL:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False) or logical_cores
            
            # Get available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
        else:
            # Fallback without psutil
            logical_cores = cpu_count
            physical_cores = cpu_count // 2 if cpu_count > 2 else cpu_count
            available_gb = 8.0  # Assume 8 GB available
            total_gb = 16.0  # Assume 16 GB total
        
        logger.info(f"Hardware detection:")
        logger.info(f"  Physical CPU cores: {physical_cores}")
        logger.info(f"  Logical CPU cores: {logical_cores}")
        logger.info(f"  Available memory: {available_gb:.1f} GB / {total_gb:.1f} GB")
        
        # Base calculation: use physical cores, but leave 1-2 cores free
        base_workers = max(1, physical_cores - 1)
        
        # Memory-based adjustment
        # Each worker loads ~1-2 GB of environmental data (rasters, GeoDataFrames)
        # Reserve 4 GB for system and main process
        memory_workers = max(1, int((available_gb - 4) / 1.5))
        
        # Dataset size adjustment (for very large datasets, can use more workers)
        if dataset_size:
            if dataset_size > 500000:
                # Very large dataset - can benefit from more workers
                size_factor = 1.2
            elif dataset_size > 100000:
                # Large dataset
                size_factor = 1.0
            elif dataset_size < 10000:
                # Small dataset - fewer workers to avoid overhead
                size_factor = 0.7
            else:
                size_factor = 1.0
        else:
            size_factor = 1.0
        
        # Take minimum of CPU-based and memory-based, apply size factor
        optimal = max(1, min(base_workers, memory_workers))
        optimal = max(1, int(optimal * size_factor))
        
        # Cap at reasonable maximum (8 workers for threading due to GIL)
        optimal = min(optimal, 8)
        
        # Minimum of 2 for small systems
        if optimal < 2 and physical_cores >= 2:
            optimal = 2
        
        logger.info(f"  Recommended workers: {optimal}")
        
        return optimal
        
    except Exception as e:
        logger.warning(f"Could not detect hardware, using default: {e}")
        return 4


def detect_optimal_batch_size(dataset_size: int, n_workers: int) -> int:
    """
    Auto-detect optimal batch size based on dataset size and workers.
    
    Args:
        dataset_size: Number of rows in dataset
        n_workers: Number of workers
    
    Returns:
        Optimal batch size
    """
    # Base batch size: save progress every 1000 rows
    base_batch = 1000
    
    # Adjust based on dataset size
    if dataset_size > 500000:
        # Very large dataset - save more frequently
        batch_size = 500
    elif dataset_size > 100000:
        # Large dataset - standard batch size
        batch_size = 1000
    elif dataset_size > 10000:
        # Medium dataset - can save less frequently
        batch_size = 2000
    else:
        # Small dataset - save at end
        batch_size = dataset_size
    
    # Adjust based on workers (more workers = smaller batches to avoid memory issues)
    if n_workers > 4:
        batch_size = int(batch_size * 0.75)
    
    # Ensure reasonable bounds
    batch_size = max(100, min(batch_size, 5000))
    
    return batch_size


def _process_sequential(df, builder, date_col, batch_size, limit, dataset_path, force: bool = False):
    """Process rows sequentially (original method)"""
    updated_count = 0
    error_count = 0
    skipped_count = 0
    
    # Define environmental columns (static + temporal)
    env_columns = [
        # Static features (terrain, infrastructure, etc.)
        'elevation', 'slope_degrees', 'aspect_degrees',
        'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
        'water_distance_miles', 'water_reliability',
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent',
        # Temporal features (SNOTEL, weather, satellite)
        'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
        'snow_data_source', 'snow_station_name', 'snow_station_distance_km',  # Data quality tracking
        'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
        'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
    ]
    
    # Filter rows if not forcing (only process rows with placeholders)
    if not force:
        rows_to_process = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            if has_placeholder_values(row, env_columns):
                rows_to_process.append(idx)
        skipped_count = len(df) - len(rows_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count:,} rows without placeholders (use --force to process all)")
    else:
        rows_to_process = list(range(len(df)))
        logger.info("Force mode: Processing all rows")
    
    iterator = rows_to_process
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Processing points", total=len(rows_to_process))
    else:
        logger.info(f"Processing {len(rows_to_process):,} points...")
    
    for idx in iterator:
        try:
            row = df.iloc[idx]
            lat, lon = row['latitude'], row['longitude']
            
            # Get date if available
            date_str = "2024-01-01"  # Default for feature enrichment (not for month assignment)
            used_default_date = True
            if date_col and pd.notna(row[date_col]):
                date_val = str(row[date_col]).strip()
                # Check if date string is valid (not empty, not 'nan', not 'none')
                if date_val and date_val.lower() not in ['nan', 'none', '']:
                    if ' ' in date_val:
                        date_val = date_val.split(' ')[0]
                    # Try to validate it's a parseable date
                    try:
                        from datetime import datetime
                        datetime.strptime(date_val, "%Y-%m-%d")  # Validate format
                        date_str = date_val
                        used_default_date = False  # We have an actual valid date
                    except (ValueError, TypeError):
                        # Invalid date format - use default for feature enrichment only
                        pass
            
            # Build context (use default date for feature enrichment even if date is missing)
            context = builder.build_context(
                location={"lat": lat, "lon": lon},
                date=date_str
            )
            
            # Extract year, month, and day_of_year from date_str ONLY if it's an actual date, not default
            # This ensures temporal metadata is populated from real dates, not defaults
            if not used_default_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    if 'year' in df.columns and pd.isna(row.get('year')):
                        df.at[idx, 'year'] = date_obj.year
                    if 'month' in df.columns and pd.isna(row.get('month')):
                        df.at[idx, 'month'] = date_obj.month
                    if 'day_of_year' in df.columns and pd.isna(row.get('day_of_year')):
                        df.at[idx, 'day_of_year'] = date_obj.timetuple().tm_yday
                except (ValueError, TypeError):
                    # If date parsing fails, leave year/month/day_of_year as is
                    pass
            
            # Update environmental columns (static + temporal)
            env_columns = [
                # Static features (terrain, infrastructure, etc.)
                'elevation', 'slope_degrees', 'aspect_degrees',
                'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
                'water_distance_miles', 'water_reliability',
                'road_distance_miles', 'trail_distance_miles',
                'security_habitat_percent',
                # Temporal features (SNOTEL, weather, satellite)
                'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
                'snow_data_source', 'snow_station_name', 'snow_station_distance_km',  # Data quality tracking
                'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
                'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
            ]
            
            for col in env_columns:
                if col in context:
                    if col not in df.columns:
                        df[col] = None
                    df.at[idx, col] = context[col]
            
            updated_count += 1
            
            # Save progress periodically
            if (idx + 1) % batch_size == 0:
                if limit is not None:
                    test_output = dataset_path.parent / f"{dataset_path.stem}_test{dataset_path.suffix}"
                    df.to_csv(test_output, index=False)
                else:
                    df.to_csv(dataset_path, index=False)
                logger.debug(f"Progress saved at row {idx + 1}")
        
        except Exception as e:
            error_count += 1
            logger.warning(f"Error processing row {idx}: {e}")
            if error_count > 100:
                logger.error("Too many errors, stopping")
                break
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count:,} rows without placeholders")
    return updated_count, error_count


def _process_parallel(df, data_dir, date_col, batch_size, limit, dataset_path, n_workers, force: bool = False):
    """Process rows in parallel using multiprocessing"""
    # Define environmental columns (static + temporal)
    env_columns = [
        # Static features (terrain, infrastructure, etc.)
        'elevation', 'slope_degrees', 'aspect_degrees',
        'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
        'water_distance_miles', 'water_reliability',
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent',
        # Temporal features (SNOTEL, weather, satellite)
        'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
        'snow_data_source', 'snow_station_name', 'snow_station_distance_km',  # Data quality tracking
        'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
        'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
    ]
    
    # Filter rows if not forcing (only process rows with placeholders)
    if not force:
        rows_to_process = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            if has_placeholder_values(row, env_columns):
                rows_to_process.append(idx)
        skipped_count = len(df) - len(rows_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count:,} rows without placeholders (use --force to process all)")
        # Filter dataframe to only rows with placeholders
        df = df.iloc[rows_to_process].copy()
    else:
        logger.info("Force mode: Processing all rows")
    
    # Prepare batches
    # Use batch_size parameter, but ensure reasonable chunking for parallel processing
    # Cap at 5000 rows per batch to avoid extremely long-running batches
    MAX_BATCH_SIZE = 5000
    effective_batch_size = min(batch_size, MAX_BATCH_SIZE)
    # Create enough batches for parallel processing (at least 2 batches per worker)
    min_batches = n_workers * 2
    chunk_size = max(100, min(effective_batch_size, len(df) // min_batches))
    batches = []
    
    # Convert data_dir to string for pickling (Path objects can sometimes cause issues)
    data_dir_str = str(data_dir)
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        # Convert rows to dictionaries to avoid any pandas-specific pickling issues
        batch_data = [(idx, row.to_dict()) for idx, row in chunk.iterrows()]
        # Include GEE config in batch args
        gee_config_dict = CONFIG.get('gee', {}) if CONFIG else {}
        batches.append((batch_data, data_dir_str, date_col, gee_config_dict))
    
    logger.info(f"Processing {len(batches)} batches with {n_workers} workers")
    
    updated_count = 0
    error_count = 0
    
    # Process batches in parallel
    env_columns = [
        # Static features (terrain, infrastructure, etc.)
        'elevation', 'slope_degrees', 'aspect_degrees',
        'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
        'water_distance_miles', 'water_reliability',
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent',
        # Temporal features (SNOTEL, weather, satellite)
        'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
        'snow_data_source', 'snow_station_name', 'snow_station_distance_km',  # Data quality tracking
        'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
        'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
    ]
    
    # Initialize columns if needed
    for col in env_columns:
        if col not in df.columns:
            df[col] = None
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
    # with rasterio dataset handles. ThreadPoolExecutor doesn't require pickling.
    # Note: Threads are limited by Python's GIL, but I/O operations (raster reading)
    # can still benefit from parallelization.
    logger.info("Using ThreadPoolExecutor (avoids rasterio pickling issues)")
    logger.info(f"Processing {len(batches)} batches with {n_workers} workers")
    logger.info(f"First batch has ~{len(batches[0][0]):,} rows - this may take 2-3 minutes")
    
    # Calculate dynamic timeout based on batch size (more rows = more time)
    # Base: 1 minute per 1000 rows, minimum 2 minutes, maximum 10 minutes
    base_rows = len(batches[0][0])
    timeout_seconds = max(120, min(600, int(base_rows / 1000 * 60)))
    logger.info(f"Using {timeout_seconds/60:.1f} minute timeout per batch (based on batch size)")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches
        logger.info("Submitting batches to workers...")
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        logger.info(f"All {len(future_to_batch)} batches submitted")
        
        # Start background thread to log periodic progress
        import threading
        progress_stop = threading.Event()
        progress_stats = {'batch_count': 0, 'start_time': time.time(), 'last_progress_time': time.time()}
        
        def log_periodic_progress():
            """Background thread to log progress every 2 minutes"""
            PROGRESS_INTERVAL = 120  # Log every 2 minutes
            while not progress_stop.is_set():
                time.sleep(PROGRESS_INTERVAL)
                if progress_stop.is_set():
                    break
                elapsed = time.time() - progress_stats['start_time']
                completed = progress_stats['batch_count']
                total = len(batches)
                logger.info(f"⏳ Processing in progress... ({completed}/{total} batches completed, "
                          f"{elapsed/60:.1f} min elapsed)")
                progress_stats['last_progress_time'] = time.time()
        
        progress_thread = threading.Thread(target=log_periodic_progress, daemon=True)
        progress_thread.start()
        
        try:
            # Process results as they complete
            iterator = as_completed(future_to_batch)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(batches), desc="Processing batches")
            
            batch_count = 0
            start_time = time.time()
            
            for future in iterator:
                try:
                    # Add timeout to detect stuck workers
                    # Use dynamic timeout based on batch size
                    batch_idx = future_to_batch[future]
                    batch_size = len(batches[batch_idx][0])
                    batch_timeout = max(120, min(600, int(batch_size / 1000 * 60)))
                    
                    logger.debug(f"Waiting for batch {batch_idx} (timeout: {batch_timeout}s)")
                    results = future.result(timeout=batch_timeout)
                    
                    for idx, context in results:
                        if context is None:
                            error_count += 1
                        else:
                            # Update year, month, and day_of_year if they were extracted from date
                            if '_year' in context and context['_year'] is not None:
                                if 'year' in df.columns and pd.isna(df.at[idx, 'year']):
                                    df.at[idx, 'year'] = context['_year']
                            if '_month' in context and context['_month'] is not None:
                                if 'month' in df.columns and pd.isna(df.at[idx, 'month']):
                                    df.at[idx, 'month'] = context['_month']
                            if '_day_of_year' in context and context['_day_of_year'] is not None:
                                if 'day_of_year' in df.columns and pd.isna(df.at[idx, 'day_of_year']):
                                    df.at[idx, 'day_of_year'] = context['_day_of_year']
                            
                            # Update environmental columns
                            for col in env_columns:
                                if col in context:
                                    df.at[idx, col] = context[col]
                            updated_count += 1
                    
                    # Log batch completion with timing
                    batch_count += 1
                    progress_stats['batch_count'] = batch_count  # Update for progress thread
                    elapsed = time.time() - start_time
                    avg_time_per_batch = elapsed / batch_count if batch_count > 0 else 0
                    remaining_batches = len(batches) - batch_count
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    
                    logger.info(f"✓ Batch {batch_count}/{len(batches)} completed "
                              f"({batch_count/len(batches)*100:.0f}%) - "
                              f"~{estimated_remaining/60:.1f} min remaining")
                              
                except TimeoutError:
                    batch_idx = future_to_batch[future]
                    batch_size_actual = len(batches[batch_idx][0])
                    error_count += batch_size_actual
                    logger.error(f"Batch {batch_idx} timed out after 5 minutes (likely stuck)")
                    logger.error(f"  This batch had {batch_size_actual} rows")
                    logger.error(f"  Consider reducing --workers or checking for issues")
                except Exception as e:
                    batch_idx = future_to_batch[future]
                    batch_size_actual = len(batches[batch_idx][0])
                    error_count += batch_size_actual
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    logger.error(f"  This batch had {batch_size_actual} rows")
                    import traceback
                    logger.debug(traceback.format_exc())
        finally:
            # Stop progress logging thread
            progress_stop.set()
            progress_thread.join(timeout=1)
    
    return updated_count, error_count


def process_batch(args):
    """Process a batch of rows - used for parallel processing"""
    batch_data, data_dir_str, date_col, gee_config_dict = args
    
    # Convert string back to Path and initialize DataContextBuilder for this worker
    # This ensures clean pickling (Path objects can be pickled, but we convert to string for safety)
    from pathlib import Path
    import logging
    worker_logger = logging.getLogger(f"{__name__}.worker")
    
    data_dir = Path(data_dir_str)
    
    # Initialize DataContextBuilder for this worker
    # Each worker loads its own copy of the rasters (avoids pickling issues)
    try:
        use_gee_ndvi = gee_config_dict.get('enabled', False) if gee_config_dict else False
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=use_gee_ndvi,
            gee_config=gee_config_dict or {}
        )
    except Exception as e:
        worker_logger.error(f"Failed to initialize DataContextBuilder: {e}")
        return [(idx, None) for idx, _ in batch_data]
    
    results = []
    processed = 0
    batch_start_time = time.time()
    last_log_time = batch_start_time  # Track last log time for time-based logging
    
    worker_logger.info(f"Worker starting batch with {len(batch_data)} rows")
    
    for row_data in batch_data:
        idx, row_dict = row_data
        try:
            # row_dict is a dictionary (converted from pandas Series)
            # Ensure we're working with basic Python types
            lat = float(row_dict['latitude'])
            lon = float(row_dict['longitude'])
            
            # Get date if available
            date_str = "2024-01-01"  # Default for feature enrichment (not for month assignment)
            used_default_date = True
            if date_col and date_col in row_dict:
                date_val = row_dict[date_col]
                # Handle None, NaN, and other non-string values
                if date_val is not None:
                    date_val_str = str(date_val).strip()
                    if date_val_str and date_val_str.lower() not in ['nan', 'none', '']:
                        if ' ' in date_val_str:
                            date_val_str = date_val_str.split(' ')[0]
                        # Try to validate it's a parseable date
                        try:
                            from datetime import datetime
                            datetime.strptime(date_val_str, "%Y-%m-%d")  # Validate format
                            date_str = date_val_str
                            used_default_date = False  # We have an actual valid date
                        except (ValueError, TypeError):
                            # Invalid date format - use default for feature enrichment only
                            pass
            
            # Build context (use default date for feature enrichment even if date is missing)
            context = builder.build_context(
                location={"lat": lat, "lon": lon},
                date=date_str
            )
            
            # Extract year, month, and day_of_year from date_str ONLY if it's an actual date, not default
            # This will be used to update year/month/day_of_year columns for absence rows
            # But we only set these if we have a real date, not a default
            if not used_default_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    context['_year'] = date_obj.year
                    context['_month'] = date_obj.month
                    context['_day_of_year'] = date_obj.timetuple().tm_yday
                except (ValueError, TypeError):
                    # If date parsing fails, leave year/month/day_of_year as is
                    context['_year'] = None
                    context['_month'] = None
                    context['_day_of_year'] = None
            else:
                # If we used default date, don't set temporal metadata (keep it None)
                context['_year'] = None
                context['_month'] = None
                context['_day_of_year'] = None
            
            results.append((idx, context))
            processed += 1
            
            # Log progress every 500 rows OR every 2 minutes (whichever comes first) to show worker is alive
            # This ensures progress is visible even during slow API calls
            current_time = time.time()
            elapsed = current_time - batch_start_time
            should_log_by_count = processed % 500 == 0
            should_log_by_time = (current_time - last_log_time) >= 120  # Log every 2 minutes
            
            if should_log_by_count or should_log_by_time:
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(batch_data) - processed
                eta = remaining / rate if rate > 0 else 0
                # Use .1f to show decimals (prevents "0 rows/sec" when rate is 0.4)
                worker_logger.info(f"Worker: {processed:,}/{len(batch_data):,} rows "
                                 f"({processed/len(batch_data)*100:.1f}%) - "
                                 f"{rate:.1f} rows/sec - ~{eta/60:.1f} min remaining")
                last_log_time = current_time
                
        except Exception as e:
            worker_logger.warning(f"Error processing row {idx}: {e}")
            results.append((idx, None))  # Error marker
    
    total_time = time.time() - batch_start_time
    worker_logger.info(f"Worker completed batch: {processed:,} rows in {total_time/60:.1f} minutes "
                      f"({processed/total_time:.1f} rows/sec)")
    
    return results


def update_dataset(dataset_path: Path, data_dir: Path, batch_size: int = 1000, limit: int = None, n_workers: int = None, force: bool = False):
    """
    Update dataset with real environmental features.
    
    Args:
        dataset_path: Path to CSV file to update
        data_dir: Path to data directory with environmental datasets
        batch_size: Number of rows to process before saving progress
        limit: Maximum number of rows to process (for testing). If None, processes all rows.
    """
    # Ensure absolute paths for multiprocessing compatibility
    data_dir = data_dir.resolve()
    dataset_path = dataset_path.resolve()

    logger.info(f"Loading dataset: {dataset_path}")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path)
    original_len = len(df)
    logger.info(f"Loaded {original_len:,} rows")
    
    # Limit rows for testing if specified
    if limit is not None:
        df = df.head(limit)
        logger.info(f"⚠️  TEST MODE: Processing only first {len(df):,} rows (limit={limit})")
    
    # Check required columns
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Initialize DataContextBuilder
    logger.info("Initializing DataContextBuilder...")
    try:
        # Load GEE config if available
        gee_config = CONFIG.get('gee', {}) if CONFIG else {}
        use_gee_ndvi = gee_config.get('enabled', False)
        
        if use_gee_ndvi:
            logger.info(f"GEE NDVI enabled (project: {gee_config.get('project', 'default')})")
        
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=use_gee_ndvi,
            gee_config=gee_config
        )
        # Note: NDVI data uses placeholders (integration deferred - see TODO in processors.py)
        # Weather data uses real APIs (PRISM/Open-Meteo) when available
        
        logger.info("✓ DataContextBuilder initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DataContextBuilder: {e}")
        return False
    
    # Check what environmental data is available
    logger.info("\nEnvironmental data availability:")
    logger.info(f"  DEM: {builder.dem is not None}")
    logger.info(f"  Slope: {builder.slope is not None}")
    logger.info(f"  Aspect: {builder.aspect is not None}")
    logger.info(f"  Land cover: {builder.landcover is not None}")
    logger.info(f"  Canopy: {builder.canopy is not None}")
    logger.info(f"  Water sources: {builder.water_sources is not None}")
    logger.info(f"  Roads: {builder.roads is not None}")
    logger.info(f"  Trails: {builder.trails is not None}")
    
    # Determine date column
    date_col = None
    for col in ['date', 'firstdate', 'Date_Time_MST']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        logger.info(f"Using date column: {date_col}")
        
        # Ensure temporal columns exist and populate them from timestamp if available
        # This ensures year, month, and day_of_year are available for all rows
        if 'year' not in df.columns:
            df['year'] = None
        if 'month' not in df.columns:
            df['month'] = None
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = None
        
        # Extract year, month, and day_of_year from timestamp if not already populated
        # This handles rows that have timestamps but missing temporal metadata
        try:
            pd_date = pd.to_datetime(df[date_col], errors='coerce')
            mask_notna = pd_date.notna()
            
            if df['year'].isna().any():
                df.loc[mask_notna & df['year'].isna(), 'year'] = pd_date[mask_notna & df['year'].isna()].dt.year
            if df['month'].isna().any():
                df.loc[mask_notna & df['month'].isna(), 'month'] = pd_date[mask_notna & df['month'].isna()].dt.month
            if df['day_of_year'].isna().any():
                df.loc[mask_notna & df['day_of_year'].isna(), 'day_of_year'] = pd_date[mask_notna & df['day_of_year'].isna()].dt.dayofyear
                
            logger.info(f"Extracted temporal columns from {date_col}: "
                       f"year={df['year'].notna().sum():,}, "
                       f"month={df['month'].notna().sum():,}, "
                       f"day_of_year={df['day_of_year'].notna().sum():,}")
        except Exception as e:
            logger.warning(f"Could not extract temporal columns from {date_col}: {e}")
    else:
        logger.warning("No date column found, using default date")
        # Still create the columns even if no date column (for consistency)
        if 'year' not in df.columns:
            df['year'] = None
        if 'month' not in df.columns:
            df['month'] = None
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = None
    
    # Add/update environmental features
    logger.info(f"\nAdding environmental features to {len(df):,} points...")
    
    # Auto-detect optimal workers if not specified
    if n_workers is None:
        n_workers = detect_optimal_workers(len(df))
        logger.info(f"Auto-detected optimal workers: {n_workers}")
    
    # Auto-detect optimal batch size if using default
    if batch_size == 1000:  # Default value
        optimal_batch = detect_optimal_batch_size(len(df), n_workers)
        if optimal_batch != batch_size:
            batch_size = optimal_batch
            logger.info(f"Auto-detected optimal batch size: {batch_size}")
    
    # Determine if we should use parallel processing
    # Standardized threshold: sequential for datasets < 5000 rows
    # This matches the logic in generate_absence_data.py
    # For small datasets, sequential is faster due to data loading overhead
    PARALLEL_THRESHOLD = 5000
    use_parallel = n_workers > 1 and len(df) >= PARALLEL_THRESHOLD
    
    if use_parallel:
        logger.info(f"Dataset size ({len(df):,}) >= {PARALLEL_THRESHOLD:,} rows - using parallel processing with {n_workers} workers")
        updated_count, error_count = _process_parallel(df, data_dir, date_col, batch_size, limit, dataset_path, n_workers, force)
    else:
        if len(df) < PARALLEL_THRESHOLD:
            logger.info(f"Dataset size ({len(df):,}) < {PARALLEL_THRESHOLD:,} rows - using sequential processing")
        else:
            logger.info("Using sequential processing")
        updated_count, error_count = _process_sequential(df, builder, date_col, batch_size, limit, dataset_path, force)
    
    # Final save
    logger.info(f"\nSaving updated dataset...")
    if limit is not None:
        # In test mode, save to a test file
        test_output = dataset_path.parent / f"{dataset_path.stem}_test{dataset_path.suffix}"
        df.to_csv(test_output, index=False)
        logger.info(f"⚠️  TEST MODE: Saved to {test_output} (not overwriting original)")
        logger.info(f"   Original file unchanged: {dataset_path}")
        output_path = test_output
    else:
        df.to_csv(dataset_path, index=False)
        output_path = dataset_path
    
    logger.info(f"\n✓ Update complete!")
    logger.info(f"  Updated: {updated_count:,} rows")
    logger.info(f"  Errors: {error_count:,} rows")
    if limit is not None:
        logger.info(f"  Test mode: Processed {len(df):,} of {original_len:,} total rows")
    logger.info(f"  Saved to: {output_path}")
    
    # Show sample of updated values
    logger.info("\nSample updated values:")
    sample_cols = ['elevation', 'slope_degrees', 'water_distance_miles']
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        logger.info(f"\n{df[available_cols].head().to_string()}")
    
    return True


def find_all_datasets(processed_dir: Path) -> list[Path]:
    """
    Find all combined presence/absence dataset files.
    
    Args:
        processed_dir: Directory to search for dataset files
    
    Returns:
        List of paths to dataset CSV files (excluding test files)
    """
    # Find all combined_*_presence_absence.csv files, excluding *_test.csv files
    all_files = list(processed_dir.glob('combined_*_presence_absence.csv'))
    # Filter out test files
    dataset_files = [f for f in all_files if '_test' not in f.stem]
    return sorted(dataset_files)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add environmental features to presence/absence datasets"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        default=None,
        help='Path to dataset CSV file to update. If not provided, processes all combined_*_presence_absence.csv files in the processed directory.'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--processed-dir',
        default='data/processed',
        help='Path to processed directory (default: data/processed). Used when processing all datasets.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of rows to process before saving progress. Default: 1000 (auto-adjusted based on dataset size)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing). If not specified, processes all rows.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers to use. Default: None (auto-detect based on hardware). Use 1 for sequential processing.'
    )
    parser.add_argument(
        '--auto-workers',
        action='store_true',
        help='Explicitly enable auto-detection of optimal workers (this is the default behavior)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of all features, even if placeholders don\'t exist'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()  # Resolve to absolute path for multiprocessing
    processed_dir = Path(args.processed_dir).resolve()
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Determine which dataset(s) to process
    if args.dataset is None:
        # Process all datasets
        if not processed_dir.exists():
            logger.error(f"Processed directory not found: {processed_dir}")
            return 1
        
        dataset_files = find_all_datasets(processed_dir)
        
        if not dataset_files:
            logger.warning(f"No combined_*_presence_absence.csv files found in {processed_dir}")
            logger.info("Expected files matching pattern: combined_*_presence_absence.csv")
            return 1
        
        logger.info(f"Found {len(dataset_files)} dataset file(s) to process:")
        for f in dataset_files:
            logger.info(f"  - {f.name}")
        logger.info("")
        
        # Process datasets in parallel (optimization 1)
        # Limit concurrent datasets to avoid memory exhaustion
        max_concurrent = min(len(dataset_files), 4)  # Limit to 4 concurrent datasets
        
        if max_concurrent > 1:
            logger.info(f"Processing {len(dataset_files)} dataset(s) in parallel (max {max_concurrent} concurrent)")
        else:
            logger.info(f"Processing {len(dataset_files)} dataset(s)")
        
        all_success = True
        failed_datasets = []
        
        # Use ThreadPoolExecutor for parallel dataset processing
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all datasets
            future_to_dataset = {
                executor.submit(
                    update_dataset,
                    dataset_path,
                    data_dir,
                    args.batch_size,
                    args.limit,
                    args.workers,
                    args.force
                ): dataset_path
                for dataset_path in dataset_files
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_dataset):
                dataset_path = future_to_dataset[future]
                completed += 1
                
                try:
                    success = future.result()
                    if success:
                        logger.info(f"✓ [{completed}/{len(dataset_files)}] Successfully processed {dataset_path.name}")
                    else:
                        logger.error(f"✗ [{completed}/{len(dataset_files)}] Failed to process {dataset_path.name}")
                        all_success = False
                        failed_datasets.append(dataset_path.name)
                except Exception as e:
                    logger.error(f"✗ [{completed}/{len(dataset_files)}] Error processing {dataset_path.name}: {e}")
                    all_success = False
                    failed_datasets.append(dataset_path.name)
        
        if all_success:
            logger.info(f"{'='*70}")
            logger.info(f"✓ All {len(dataset_files)} dataset(s) processed successfully!")
            logger.info(f"{'='*70}")
            return 0
        else:
            logger.error(f"{'='*70}")
            logger.error(f"✗ Some datasets failed to process")
            if failed_datasets:
                logger.error(f"Failed datasets: {', '.join(failed_datasets)}")
            logger.error(f"{'='*70}")
            return 1
    else:
        # Process single dataset
        dataset_path = Path(args.dataset)
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info(f"Usage: python {sys.argv[0]} [dataset_path]")
            logger.info(f"  Or omit dataset_path to process all combined_*_presence_absence.csv files")
            return 1
        
        success = update_dataset(
            dataset_path, data_dir, args.batch_size, args.limit, args.workers, args.force
        )
        
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

