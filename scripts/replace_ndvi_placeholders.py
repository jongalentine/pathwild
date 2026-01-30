#!/usr/bin/env python3
"""
Replace NDVI placeholder values with real AppEEARS data.

This script processes feature files to replace placeholder NDVI values
using AppEEARS async API. It:
1. Identifies rows with placeholder values
2. Submits AppEEARS requests asynchronously
3. Waits for task completion
4. Downloads and parses results
5. Updates feature file with real values

Usage:
    python scripts/replace_ndvi_placeholders.py \
        --input-file data/features/southern_gye_features.csv \
        [--output-file data/features/southern_gye_features.csv] \
        [--batch-size 100] \
        [--max-wait-minutes 30] \
        [--dry-run]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.appeears_client import AppEEARSClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Placeholder values
NDVI_PLACEHOLDERS = {0.3, 0.5, 0.55, 0.7}
SUMMER_NDVI_PLACEHOLDER = 60.0


def is_placeholder_ndvi(value: float) -> bool:
    """Check if NDVI value is a placeholder."""
    if pd.isna(value):
        return False
    try:
        return float(value) in NDVI_PLACEHOLDERS
    except (ValueError, TypeError):
        return False


def is_placeholder_summer_ndvi(value: float) -> bool:
    """Check if summer integrated NDVI value is a placeholder."""
    if pd.isna(value):
        return False
    try:
        return abs(float(value) - SUMMER_NDVI_PLACEHOLDER) < 0.01
    except (ValueError, TypeError):
        return False


def reconstruct_date_from_sin_cos(year: float, month: float, day_of_year_sin: float, day_of_year_cos: float) -> Optional[datetime]:
    """
    Reconstruct date from year, month, and day_of_year_sin/cos.
    
    The sin/cos encoding is: sin(2π * day_of_year / 365.25) and cos(2π * day_of_year / 365.25)
    To reverse: day_of_year = atan2(sin, cos) * 365.25 / (2π)
    
    Args:
        year: Year (e.g., 2010.0)
        month: Month (1-12)
        day_of_year_sin: sin(2π * day_of_year / 365.25)
        day_of_year_cos: cos(2π * day_of_year / 365.25)
        
    Returns:
        datetime object, or None if year/month are NaN
    """
    import math
    
    # Check for NaN values
    if pd.isna(year) or pd.isna(month):
        return None
    
    year_int = int(year)
    month_int = int(month)
    
    # Reconstruct day_of_year from sin/cos
    # atan2 returns angle in radians, convert to day of year
    angle = math.atan2(day_of_year_sin, day_of_year_cos)
    if angle < 0:
        angle += 2 * math.pi  # Normalize to [0, 2π]
    
    day_of_year = (angle / (2 * math.pi)) * 365.25
    day_of_year = int(round(day_of_year))
    
    # Normalize to valid range [1, 365/366]
    if day_of_year < 1:
        day_of_year = 1
    elif day_of_year > 366:
        day_of_year = 366
    
    # Create date from year and day of year
    date = datetime(year_int, 1, 1) + timedelta(days=day_of_year - 1)
    
    # Verify month matches (day_of_year should be consistent with month)
    if date.month != month_int:
        logger.debug(f"Month mismatch: reconstructed day_of_year={day_of_year} gives month={date.month}, but month={month_int}. Using month={month_int}.")
        # Use month instead - approximate to middle of month
        days_in_month = (datetime(year_int, month_int + 1, 1) - datetime(year_int, month_int, 1)).days if month_int < 12 else 31
        day = days_in_month // 2
        date = datetime(year_int, month_int, day)
    
    return date


def reconstruct_date(year: float, month: float, day_of_year: float = None, 
                     day_of_year_sin: float = None, day_of_year_cos: float = None) -> Optional[datetime]:
    """
    Reconstruct date from year, month, and either day_of_year or day_of_year_sin/cos.
    
    Args:
        year: Year (e.g., 2010.0)
        month: Month (1-12)
        day_of_year: Day of year (1-365/366) - if available
        day_of_year_sin: sin(2π * day_of_year / 365.25) - if day_of_year not available
        day_of_year_cos: cos(2π * day_of_year / 365.25) - if day_of_year not available
        
    Returns:
        datetime object, or None if year/month are NaN
    """
    # Check for NaN values in required fields
    if pd.isna(year) or pd.isna(month):
        return None
    
    if day_of_year is not None and not pd.isna(day_of_year):
        # Direct day_of_year available
        year_int = int(year)
        month_int = int(month)
        day_of_year_int = int(day_of_year)
        
        # Create date from year and day of year
        date = datetime(year_int, 1, 1) + timedelta(days=day_of_year_int - 1)
        
        # Verify month matches
        if date.month != month_int:
            logger.debug(f"Month mismatch: day_of_year={day_of_year_int} gives month={date.month}, but month={month_int}")
        
        return date
    elif day_of_year_sin is not None and day_of_year_cos is not None and not pd.isna(day_of_year_sin) and not pd.isna(day_of_year_cos):
        # Reconstruct from sin/cos
        return reconstruct_date_from_sin_cos(year, month, day_of_year_sin, day_of_year_cos)
    else:
        # Fallback: use middle of month
        year_int = int(year)
        month_int = int(month)
        days_in_month = (datetime(year_int, month_int + 1, 1) - datetime(year_int, month_int, 1)).days if month_int < 12 else 31
        day = days_in_month // 2
        return datetime(year_int, month_int, day)


def identify_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify rows with placeholder NDVI values.
    
    Returns:
        DataFrame with additional columns indicating placeholder status
    """
    df = df.copy()
    
    # Check for placeholder NDVI
    if 'ndvi' in df.columns:
        df['has_placeholder_ndvi'] = df['ndvi'].apply(is_placeholder_ndvi)
    else:
        df['has_placeholder_ndvi'] = False
        logger.warning("'ndvi' column not found in DataFrame")
    
    # Check for placeholder summer integrated NDVI
    if 'summer_integrated_ndvi' in df.columns:
        df['has_placeholder_summer_ndvi'] = df['summer_integrated_ndvi'].apply(is_placeholder_summer_ndvi)
    else:
        df['has_placeholder_summer_ndvi'] = False
        logger.warning("'summer_integrated_ndvi' column not found in DataFrame")
    
    return df


def group_by_date_range(rows: pd.DataFrame, batch_size: int = 100) -> List[Dict]:
    """
    Group rows by date ranges for efficient batching.
    
    Args:
        rows: DataFrame with placeholder rows
        batch_size: Maximum points per batch
        
    Returns:
        List of dicts with keys: start_date, end_date, rows
    """
    # Sort by date
    rows_sorted = rows.sort_values('date').copy()
    
    batches = []
    current_batch = []
    current_start = None
    current_end = None
    
    for idx, row in rows_sorted.iterrows():
        date = row['date']
        
        # Start new batch if needed
        if len(current_batch) >= batch_size or current_start is None:
            if current_batch:
                batches.append({
                    'start_date': current_start,
                    'end_date': current_end,
                    'rows': pd.DataFrame(current_batch)
                })
            current_batch = []
            current_start = date
            current_end = date
        
        current_batch.append(row)
        current_end = max(current_end, date) if current_end else date
    
    # Add final batch
    if current_batch:
        batches.append({
            'start_date': current_start,
            'end_date': current_end,
            'rows': pd.DataFrame(current_batch)
        })
    
    return batches


def calculate_summer_integrated_ndvi(ndvi_values: List[float]) -> float:
    """
    Calculate summer integrated NDVI from list of NDVI values.
    
    Args:
        ndvi_values: List of NDVI values from summer period
        
    Returns:
        Sum of NDVI values (integrated NDVI)
    """
    if not ndvi_values:
        return SUMMER_NDVI_PLACEHOLDER
    
    # Filter out invalid values
    valid_values = [v for v in ndvi_values if not pd.isna(v) and -1 <= v <= 1]
    
    if not valid_values:
        return SUMMER_NDVI_PLACEHOLDER
    
    return sum(valid_values)


def process_ndvi_placeholders(
    client: AppEEARSClient,
    placeholder_rows: pd.DataFrame,
    batch_size: int = 100,
    max_wait_minutes: int = 30
) -> Dict[int, float]:
    """
    Process NDVI placeholder rows using AppEEARS.
    
    Args:
        client: AppEEARS client
        placeholder_rows: DataFrame with rows needing NDVI replacement
        batch_size: Maximum points per AppEEARS request
        max_wait_minutes: Maximum time to wait for each task
        
    Returns:
        Dict mapping row index to NDVI value
    """
    results = {}
    
    # Group by date ranges
    batches = group_by_date_range(placeholder_rows, batch_size=batch_size)
    
    logger.info(f"Processing {len(placeholder_rows)} placeholder rows in {len(batches)} batches...")
    
    for batch_idx, batch in enumerate(batches, 1):
        batch_rows = batch['rows']
        start_date = batch['start_date']
        end_date = batch['end_date']
        
        logger.info(f"Batch {batch_idx}/{len(batches)}: {len(batch_rows)} points, "
                   f"date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Prepare points for AppEEARS
        points = []
        row_indices = []
        
        for idx, row in batch_rows.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            date = row['date']
            date_str = date.strftime("%Y-%m-%d")
            
            points.append((lat, lon, date_str))
            row_indices.append(idx)
        
        # Submit AppEEARS request (uses optimized batch submission and parallel polling)
        try:
            logger.info(f"  Submitting AppEEARS request (batch mode with parallel polling)...")
            result_df = client.get_ndvi_for_points(
                points,
                product="modis_ndvi",
                # date_buffer_days and max_points_per_batch use conservative optimization defaults
                # (10 days buffer, 200 points per batch) - can be overridden via environment variables
                max_wait_minutes=max_wait_minutes,
                use_batch=True,  # Use batch submission and parallel polling
                max_submit_workers=10,
                max_poll_workers=10
            )
            
            logger.info(f"  Received {len(result_df)} results")

            # Match results back to rows by index
            # get_ndvi_for_points() returns results in the same order as input points,
            # so we can directly pair them by position (result_df.iloc[i] corresponds to points[i])
            if len(result_df) != len(points):
                logger.warning(f"  Mismatch: {len(result_df)} results for {len(points)} points - some rows may not be matched")

            matched_count = 0
            none_count = 0
            invalid_count = 0

            # Iterate in parallel through points and results (same order guaranteed by client)
            for i, (point, idx) in enumerate(zip(points, row_indices)):
                point_lat, point_lon, point_date_str = point

                # Get result by position (results are in same order as input points)
                if i >= len(result_df):
                    logger.debug(f"  Row {idx}: No result at position {i} (result_df has {len(result_df)} rows)")
                    continue

                result_row = result_df.iloc[i]

                # Extract coordinates for logging (informational only - we trust index ordering)
                result_lat = result_row.get('latitude')
                result_lon = result_row.get('longitude')
                result_date = result_row.get('date')

                # Log coordinate info at debug level for diagnostics
                logger.debug(
                    f"  Row {idx} (pos {i}): input=({point_lat:.6f}, {point_lon:.6f}, {point_date_str}), "
                    f"result=({result_lat}, {result_lon}, {result_date})"
                )

                # Extract NDVI value - the client's _match_results_to_points already matched
                # results to input points by position, so we trust this ordering
                ndvi_value = result_row.get('ndvi')

                # Handle None NDVI (no data available for this point/date)
                if ndvi_value is None or pd.isna(ndvi_value):
                    none_count += 1
                    logger.debug(f"  Row {idx}: No NDVI data available (None)")
                    continue

                # Validate NDVI is in expected range
                try:
                    ndvi_float = float(ndvi_value)
                except (ValueError, TypeError):
                    invalid_count += 1
                    logger.warning(f"  Row {idx}: Cannot convert NDVI to float: {ndvi_value}")
                    continue

                if -1 <= ndvi_float <= 1:
                    results[idx] = ndvi_float
                    matched_count += 1
                    # Get original placeholder value for logging
                    original_ndvi = batch_rows.loc[idx].get('ndvi', 'N/A')
                    logger.debug(f"  Row {idx}: NDVI {original_ndvi} -> {ndvi_float:.4f}")
                else:
                    invalid_count += 1
                    logger.debug(f"  Row {idx}: Invalid NDVI value {ndvi_float} (outside range -1 to 1)")

            # Summary for this batch
            logger.info(f"  Batch result: {matched_count} valid, {none_count} no-data, {invalid_count} invalid")
                    
        except TimeoutError as e:
            logger.warning(f"  Batch {batch_idx} timed out: {e}")
            logger.warning(f"  Skipping {len(batch_rows)} rows in this batch")
        except Exception as e:
            logger.error(f"  Batch {batch_idx} failed: {e}")
            logger.error(f"  Skipping {len(batch_rows)} rows in this batch")
    
    logger.info(f"Successfully retrieved NDVI for {len(results)}/{len(placeholder_rows)} rows")
    return results


def process_summer_ndvi_placeholders(
    client: AppEEARSClient,
    placeholder_rows: pd.DataFrame,
    max_wait_minutes: int = 30
) -> Dict[int, float]:
    """
    Process summer integrated NDVI placeholder rows using AppEEARS.
    
    For each row with summer_integrated_ndvi == 60.0:
    - Extract year from date
    - Request NDVI for summer period (June 1 - September 30)
    - Sum NDVI values
    
    Optimized to process all locations in parallel instead of sequentially.
    
    Args:
        client: AppEEARS client
        placeholder_rows: DataFrame with rows needing summer NDVI replacement
        max_wait_minutes: Maximum time to wait for each task
        
    Returns:
        Dict mapping row index to summer integrated NDVI value
    """
    results = {}
    
    logger.info(f"Processing {len(placeholder_rows)} summer_integrated_ndvi placeholder rows...")
    
    # Group by year for efficiency
    by_year = defaultdict(list)
    for idx, row in placeholder_rows.iterrows():
        year = int(row['date'].year)
        by_year[year].append((idx, row))
    
    for year, year_rows in by_year.items():
        logger.info(f"Processing year {year}: {len(year_rows)} rows")
        
        # Summer period: June 1 - September 30
        summer_start = datetime(year, 6, 1)
        summer_end = datetime(year, 9, 30)
        
        # Collect all unique (lat, lon) pairs for this year
        locations = {}
        for idx, row in year_rows:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            loc_key = (round(lat, 6), round(lon, 6))  # Round to avoid floating point issues
            if loc_key not in locations:
                locations[loc_key] = []
            locations[loc_key].append(idx)
        
        logger.info(f"  {len(locations)} unique locations for year {year}")
        
        # OPTIMIZATION: Collect all points from all locations first, then process in parallel
        all_points = []
        location_to_points = {}  # Maps (lat, lon) to list of point tuples for tracking
        
        for (lat, lon), row_indices in locations.items():
            # Sample dates throughout summer (every 16 days for MODIS)
            points = []
            current_date = summer_start
            while current_date <= summer_end:
                date_str = current_date.strftime("%Y-%m-%d")
                points.append((lat, lon, date_str))
                current_date += timedelta(days=16)
            
            all_points.extend(points)
            location_to_points[(lat, lon)] = points
        
        logger.info(f"  Submitting {len(all_points)} total points ({len(locations)} locations) in parallel...")
        
        # Submit all points in a single batch request
        try:
            result_df = client.get_ndvi_for_points(
                all_points,
                product="modis_ndvi",
                date_buffer_days=5,  # Reduced from 7 to 5 for faster processing
                max_wait_minutes=max_wait_minutes,
                use_batch=True,  # Use batch submission and parallel polling
                max_submit_workers=10,
                max_poll_workers=10
            )
            
            logger.info(f"  Received {len(result_df)} results")

            # Ensure latitude/longitude columns are numeric (may be object dtype if mixed with None)
            if len(result_df) > 0:
                result_df['latitude'] = pd.to_numeric(result_df['latitude'], errors='coerce')
                result_df['longitude'] = pd.to_numeric(result_df['longitude'], errors='coerce')

            # Group results by location and calculate integrated NDVI for each
            for (lat, lon), row_indices in locations.items():
                # Find all results for this location
                location_results = result_df[
                    (result_df['latitude'].round(6) == round(lat, 6)) &
                    (result_df['longitude'].round(6) == round(lon, 6))
                ]
                
                # Calculate integrated NDVI (sum of all values)
                ndvi_values = []
                for _, result_row in location_results.iterrows():
                    ndvi = result_row.get('ndvi')
                    if pd.notna(ndvi) and -1 <= float(ndvi) <= 1:
                        ndvi_values.append(float(ndvi))
                
                integrated_ndvi = calculate_summer_integrated_ndvi(ndvi_values)
                
                # Apply to all rows with this location
                for idx in row_indices:
                    results[idx] = integrated_ndvi
                
                logger.debug(f"  Calculated summer_integrated_ndvi for ({lat:.4f}, {lon:.4f}): {integrated_ndvi:.2f} (from {len(ndvi_values)} values)")
                
        except TimeoutError as e:
            logger.warning(f"  Summer NDVI request timed out for year {year}: {e}")
            logger.warning(f"  This will affect {len(locations)} locations")
        except Exception as e:
            logger.error(f"  Summer NDVI request failed for year {year}: {e}")
            logger.error(f"  This will affect {len(locations)} locations")
    
    logger.info(f"Successfully calculated summer_integrated_ndvi for {len(results)}/{len(placeholder_rows)} rows")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Replace NDVI placeholder values with real AppEEARS data"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to input feature CSV file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to output feature CSV file (default: overwrites input file)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum points per AppEEARS batch (default: 100)"
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=30,
        help="Maximum time to wait for each AppEEARS task in minutes (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify placeholders but don't submit AppEEARS requests"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of placeholder rows to process (for testing). If not specified, processes all placeholder rows."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if no placeholders are found (for pipeline compatibility)"
    )
    
    args = parser.parse_args()
    
    # Check AppEEARS credentials
    username = os.getenv("APPEEARS_USERNAME")
    password = os.getenv("APPEEARS_PASSWORD")
    
    if not username or not password:
        logger.error("AppEEARS credentials not found. Set APPEEARS_USERNAME and APPEEARS_PASSWORD environment variables.")
        sys.exit(1)
    
    # Check if input file exists
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Set output file
    output_file = args.output_file or args.input_file
    
    logger.info("=" * 70)
    logger.info("AppEEARS NDVI Placeholder Replacement")
    logger.info("=" * 70)
    logger.info("")
    
    # Load feature file
    logger.info(f"Loading feature file: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"✗ Failed to load feature file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = ['year', 'month', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Identify placeholder rows
    logger.info("")
    logger.info("Identifying placeholder rows...")
    df = identify_placeholder_rows(df)
    
    # Reconstruct dates
    logger.info("Reconstructing dates...")
    
    # Check which date columns are available
    has_day_of_year = 'day_of_year' in df.columns
    has_day_of_year_sin = 'day_of_year_sin' in df.columns
    has_day_of_year_cos = 'day_of_year_cos' in df.columns
    
    if has_day_of_year:
        logger.info("  Using day_of_year column")
        df['date'] = df.apply(
            lambda row: reconstruct_date(row['year'], row['month'], day_of_year=row['day_of_year']),
            axis=1
        )
    elif has_day_of_year_sin and has_day_of_year_cos:
        logger.info("  Using day_of_year_sin and day_of_year_cos columns")
        df['date'] = df.apply(
            lambda row: reconstruct_date(
                row['year'], 
                row['month'], 
                day_of_year_sin=row['day_of_year_sin'],
                day_of_year_cos=row['day_of_year_cos']
            ),
            axis=1
        )
    else:
        logger.warning("  No day_of_year columns found, using middle of month as approximation")
        df['date'] = df.apply(
            lambda row: reconstruct_date(row['year'], row['month']),
            axis=1
        )
    
    # Filter out rows with missing dates (NaN year/month)
    rows_before = len(df)
    df = df[df['date'].notna()].copy()
    rows_after = len(df)
    if rows_before > rows_after:
        logger.warning(f"  Filtered out {rows_before - rows_after} rows with missing dates (NaN year/month)")
    
    placeholder_ndvi_rows = df[df['has_placeholder_ndvi']].copy()
    placeholder_summer_rows = df[df['has_placeholder_summer_ndvi']].copy()
    
    logger.info(f"  Rows with placeholder NDVI: {len(placeholder_ndvi_rows)}")
    logger.info(f"  Rows with placeholder summer_integrated_ndvi: {len(placeholder_summer_rows)}")
    
    # Apply limit if specified
    if args.limit is not None:
        original_ndvi_count = len(placeholder_ndvi_rows)
        original_summer_count = len(placeholder_summer_rows)
        
        # Limit NDVI placeholder rows
        if len(placeholder_ndvi_rows) > args.limit:
            placeholder_ndvi_rows = placeholder_ndvi_rows.head(args.limit)
            logger.info(f"  ⚠️  TEST MODE: Limiting NDVI placeholder processing to {args.limit} rows (from {original_ndvi_count})")
        
        # Limit summer NDVI placeholder rows (separate limit)
        if len(placeholder_summer_rows) > args.limit:
            placeholder_summer_rows = placeholder_summer_rows.head(args.limit)
            logger.info(f"  ⚠️  TEST MODE: Limiting summer_integrated_ndvi placeholder processing to {args.limit} rows (from {original_summer_count})")
    
    if len(placeholder_ndvi_rows) == 0 and len(placeholder_summer_rows) == 0:
        if args.force:
            logger.info("No placeholder values found, but --force specified. Nothing to replace.")
        else:
            logger.info("No placeholder values found. Nothing to replace.")
        return 0
    
    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN: Would process the following:")
        logger.info(f"  - {len(placeholder_ndvi_rows)} NDVI placeholders")
        logger.info(f"  - {len(placeholder_summer_rows)} summer_integrated_ndvi placeholders")
        return 0
    
    # Initialize AppEEARS client with caching enabled
    logger.info("")
    logger.info("Initializing AppEEARS client...")
    try:
        # Set up cache directory (use data/cache by default, relative to script location)
        script_dir = Path(__file__).parent.parent
        cache_dir = script_dir / 'data' / 'cache'
        client = AppEEARSClient(username, password, cache_dir=cache_dir, use_cache=True)
        logger.info("✓ AppEEARS client initialized")
        
        # Log cache statistics
        if client.cache:
            cache_stats = client.cache.get_stats()
            logger.info(f"  Cache: {cache_stats['total_entries']:,} entries ({cache_stats['total_size_mb']:.2f} MB)")
    except Exception as e:
        logger.error(f"✗ Failed to initialize AppEEARS client: {e}")
        sys.exit(1)
    
    # Process NDVI placeholders
    ndvi_replacements = {}
    if len(placeholder_ndvi_rows) > 0:
        logger.info("")
        logger.info("Processing NDVI placeholders...")
        ndvi_replacements = process_ndvi_placeholders(
            client,
            placeholder_ndvi_rows,
            batch_size=args.batch_size,
            max_wait_minutes=args.max_wait_minutes
        )
    
    # Process summer NDVI placeholders
    summer_replacements = {}
    if len(placeholder_summer_rows) > 0:
        logger.info("")
        logger.info("Processing summer_integrated_ndvi placeholders...")
        summer_replacements = process_summer_ndvi_placeholders(
            client,
            placeholder_summer_rows,
            max_wait_minutes=args.max_wait_minutes
        )
    
    # Apply replacements
    logger.info("")
    logger.info("Applying replacements...")
    
    replacements_applied = 0
    
    # Replace NDVI values
    for idx, ndvi_value in ndvi_replacements.items():
        old_value = df.at[idx, 'ndvi']
        df.at[idx, 'ndvi'] = ndvi_value
        logger.debug(f"  Row {idx}: NDVI {old_value} -> {ndvi_value:.4f}")
        replacements_applied += 1
    
    # Replace summer_integrated_ndvi values
    for idx, summer_value in summer_replacements.items():
        old_value = df.at[idx, 'summer_integrated_ndvi']
        df.at[idx, 'summer_integrated_ndvi'] = summer_value
        logger.debug(f"  Row {idx}: summer_integrated_ndvi {old_value} -> {summer_value:.2f}")
        replacements_applied += 1
    
    logger.info(f"✓ Applied {replacements_applied} replacements")
    
    # Remove temporary columns
    df = df.drop(columns=['has_placeholder_ndvi', 'has_placeholder_summer_ndvi', 'date'], errors='ignore')
    
    # Save updated file
    logger.info("")
    logger.info(f"Saving updated file: {output_file}")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved {len(df)} rows to {output_file}")
    except Exception as e:
        logger.error(f"✗ Failed to save file: {e}")
        sys.exit(1)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  NDVI replacements: {len(ndvi_replacements)}/{len(placeholder_ndvi_rows)}")
    logger.info(f"  Summer NDVI replacements: {len(summer_replacements)}/{len(placeholder_summer_rows)}")
    logger.info(f"  Total replacements: {replacements_applied}")
    logger.info("")
    logger.info("✓ Replacement complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
