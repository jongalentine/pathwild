#!/usr/bin/env python3
"""
Bulk download PRISM climate data for all dates needed by the training datasets.

This script pre-downloads all PRISM files needed for feature integration,
avoiding per-point API calls during pipeline execution.

Usage:
    python scripts/bulk_download_prism.py [--dry-run] [--workers N]

Examples:
    # Show what would be downloaded (dry run)
    python scripts/bulk_download_prism.py --dry-run

    # Download with 4 parallel workers
    python scripts/bulk_download_prism.py --workers 4

    # Download sequentially (safest, avoids rate limits)
    python scripts/bulk_download_prism.py --workers 1
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_needed_dates(processed_dir: Path) -> set:
    """
    Collect all unique dates needed across all datasets.

    Includes 7-day precipitation lookback window.
    """
    all_dates = set()

    for f in sorted(processed_dir.glob('combined_*_presence_absence.csv')):
        if '_test' in f.name:
            continue

        df = pd.read_csv(f)

        # Try to find date column
        date_col = None
        for col in ['firstdate', 'date', 'Date_Time_MST', 'DT']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            logger.info(f"Skipping {f.name}: no date column")
            continue

        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()

        # Add each date and 7-day lookback for precipitation
        for d in dates.dt.date:
            for i in range(8):  # target day + 7 days before
                all_dates.add(d - timedelta(days=i))

        logger.info(f"{f.name}: {len(dates)} rows with dates")

    return all_dates


def get_downloaded_dates(prism_dir: Path, variables: list) -> dict:
    """Get dates already downloaded for each variable."""
    downloaded = defaultdict(set)

    for var in variables:
        var_dir = prism_dir / var
        if not var_dir.exists():
            continue

        for f in var_dir.glob('*.tif'):
            parts = f.stem.split('_')
            if len(parts) >= 5:
                date_str = parts[-1]
                try:
                    d = datetime.strptime(date_str, '%Y%m%d').date()
                    downloaded[var].add(d)
                except ValueError:
                    pass

    return downloaded


def download_file(args: tuple) -> tuple:
    """
    Download a single PRISM file.

    Args:
        args: Tuple of (variable, date, prism_client)

    Returns:
        Tuple of (variable, date, success, error_message)
    """
    variable, date, prism_client = args

    try:
        # The extract_value method will download if needed
        result = prism_client.extract_value(variable, 44.0, -108.0, date)

        if result is not None:
            return (variable, date, True, None)
        else:
            return (variable, date, False, "No data available")

    except Exception as e:
        return (variable, date, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Bulk download PRISM climate data for training datasets"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel download workers (default: 2, use 1 to avoid rate limits)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Path to data directory (default: data)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / 'processed'
    prism_dir = data_dir / 'prism'

    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return 1

    # Collect needed dates
    logger.info("Scanning datasets for needed dates...")
    needed_dates = collect_needed_dates(processed_dir)
    logger.info(f"Total unique dates needed: {len(needed_dates)}")

    if not needed_dates:
        logger.warning("No dates found in datasets")
        return 0

    # Check what's already downloaded
    variables = ['tmean', 'tmin', 'tmax', 'ppt']
    downloaded = get_downloaded_dates(prism_dir, variables)

    # Calculate missing files
    missing_files = []
    for var in variables:
        for d in sorted(needed_dates):
            if d not in downloaded[var]:
                missing_files.append((var, d))

    logger.info(f"\n=== DOWNLOAD SUMMARY ===")
    for var in variables:
        var_missing = sum(1 for v, d in missing_files if v == var)
        var_have = len(downloaded[var] & needed_dates)
        var_need = len(needed_dates)
        logger.info(f"  {var}: {var_have}/{var_need} ({var_have/var_need*100:.0f}% complete), {var_missing} to download")

    logger.info(f"\nTotal files to download: {len(missing_files)}")

    if not missing_files:
        logger.info("All PRISM files already downloaded!")
        return 0

    if args.dry_run:
        logger.info("\n[DRY RUN] Would download the following files:")
        for var, d in missing_files[:20]:
            logger.info(f"  {var} {d}")
        if len(missing_files) > 20:
            logger.info(f"  ... and {len(missing_files) - 20} more")

        est_time = len(missing_files) * 2.5 / 3600
        logger.info(f"\nEstimated download time: {est_time:.1f} hours")
        return 0

    # Initialize PRISM client
    from src.data.prism_client import PRISMClient
    prism_client = PRISMClient(prism_dir)

    # Download missing files
    logger.info(f"\nStarting download with {args.workers} worker(s)...")

    start_time = time.time()
    success_count = 0
    error_count = 0
    rate_limit_count = 0

    # Prepare download tasks
    tasks = [(var, datetime.combine(d, datetime.min.time()), prism_client)
             for var, d in missing_files]

    if args.workers == 1:
        # Sequential download
        for i, task in enumerate(tasks):
            var, date, _ = task
            result = download_file(task)
            _, _, success, error = result

            if success:
                success_count += 1
            elif error and "rate limit" in error.lower():
                rate_limit_count += 1
                logger.warning(f"Rate limit hit for {var} {date.date()}")
            else:
                error_count += 1
                logger.warning(f"Failed to download {var} {date.date()}: {error}")

            # Progress every 50 files
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
                logger.info(f"Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%) "
                          f"- {rate:.1f} files/sec - ~{remaining/60:.1f} min remaining")
    else:
        # Parallel download
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_file, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                var, date, success, error = result

                if success:
                    success_count += 1
                elif error and "rate limit" in error.lower():
                    rate_limit_count += 1
                else:
                    error_count += 1

                # Progress every 50 files
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    logger.info(f"Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%) "
                              f"- {rate:.1f} files/sec - ~{remaining/60:.1f} min remaining")

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n=== DOWNLOAD COMPLETE ===")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Success: {success_count}")
    logger.info(f"Rate limited: {rate_limit_count}")
    logger.info(f"Errors: {error_count}")

    if rate_limit_count > 0:
        logger.warning(f"\n{rate_limit_count} files hit rate limit. Run again tomorrow to retry.")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
