#!/usr/bin/env python3
"""
Bulk download ERA5 cloud cover data for Wyoming.

This script downloads historical cloud cover data from the ECMWF Climate Data Store (CDS)
for the entire Wyoming region and date range needed by the training datasets.

ERA5 provides hourly reanalysis data at 0.25° resolution (~25km), which is appropriate
for cloud cover (a regional phenomenon). This replaces per-point OpenMeteo API calls
which are rate-limited and unsuitable for bulk historical data retrieval.

Prerequisites:
    1. Create a CDS account: https://cds.climate.copernicus.eu/user/register
    2. Accept the license: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
    3. Get your API key from: https://cds.climate.copernicus.eu/user (after login)
    4. Create ~/.cdsapirc file with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-uid>:<your-api-key>
    5. Install cdsapi: pip install cdsapi

Usage:
    python scripts/bulk_download_era5_cloud.py [--dry-run] [--year-start YEAR] [--year-end YEAR]

Examples:
    # Show what would be downloaded
    python scripts/bulk_download_era5_cloud.py --dry-run

    # Download all data (2006-2019)
    python scripts/bulk_download_era5_cloud.py

    # Download specific year range
    python scripts/bulk_download_era5_cloud.py --year-start 2006 --year-end 2010
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wyoming bounding box (with small buffer)
# Actual Wyoming: 41°N to 45°N, 111°W to 104°W
WYOMING_BBOX = {
    'north': 45.5,
    'south': 40.5,
    'west': -111.5,
    'east': -103.5
}


def check_cdsapi_setup():
    """Check if CDS API is properly configured."""
    try:
        import cdsapi
    except ImportError:
        logger.error("cdsapi not installed. Run: pip install cdsapi")
        return False

    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        logger.error(f"CDS API config not found at {cdsapirc}")
        logger.error("Please create this file with your CDS API credentials.")
        logger.error("See: https://cds.climate.copernicus.eu/how-to-api")
        return False

    return True


def get_date_range_from_datasets(data_dir: Path) -> tuple:
    """
    Determine the date range needed from all datasets.

    Returns:
        Tuple of (min_year, max_year)
    """
    import pandas as pd
    from datetime import timedelta

    processed_dir = data_dir / 'processed'
    all_dates = []

    for f in processed_dir.glob('combined_*_presence_absence.csv'):
        if '_test' in f.name:
            continue

        df = pd.read_csv(f)

        # Find date column
        date_col = None
        for col in ['date', 'firstdate', 'Date_Time_MST', 'DT']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            continue

        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if len(dates) > 0:
            all_dates.extend(dates.tolist())

    if not all_dates:
        # Default range if no data found
        return 2006, 2019

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Add buffer for 7-day precipitation lookback
    min_year = (min_date - timedelta(days=7)).year
    max_year = max_date.year

    return min_year, max_year


def download_era5_cloud_cover(
    output_dir: Path,
    year_start: int,
    year_end: int,
    dry_run: bool = False
):
    """
    Download ERA5 total cloud cover for Wyoming.

    Args:
        output_dir: Directory to save downloaded files
        year_start: First year to download
        year_end: Last year to download (inclusive)
        dry_run: If True, only show what would be downloaded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download one file per year to manage file sizes
    # Each year is roughly 50-100MB
    years = list(range(year_start, year_end + 1))

    logger.info(f"=== ERA5 Cloud Cover Download ===")
    logger.info(f"Years: {year_start} to {year_end} ({len(years)} years)")
    logger.info(f"Region: Wyoming ({WYOMING_BBOX})")
    logger.info(f"Resolution: 0.25° (~25km)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Estimated size: ~{len(years) * 75}MB total")

    if dry_run:
        logger.info("\n[DRY RUN] Would download the following files:")
        for year in years:
            output_file = output_dir / f'era5_cloud_cover_wyoming_{year}.nc'
            logger.info(f"  {output_file.name}")
        logger.info(f"\nTo download, run without --dry-run (requires cdsapi package)")
        return

    import cdsapi
    client = cdsapi.Client()

    for i, year in enumerate(years):
        output_file = output_dir / f'era5_cloud_cover_wyoming_{year}.nc'

        if output_file.exists():
            logger.info(f"[{i+1}/{len(years)}] {year}: Already downloaded, skipping")
            continue

        logger.info(f"[{i+1}/{len(years)}] Downloading {year}...")

        try:
            # Download daily data at noon (12:00 UTC = ~6:00 AM Mountain Time)
            # This captures overnight cloud conditions relevant for nocturnal activity
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': 'total_cloud_cover',
                    'year': str(year),
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],  # 4 times per day
                    'area': [
                        WYOMING_BBOX['north'],
                        WYOMING_BBOX['west'],
                        WYOMING_BBOX['south'],
                        WYOMING_BBOX['east']
                    ],
                    'format': 'netcdf',
                },
                str(output_file)
            )

            # Verify download
            if output_file.exists() and output_file.stat().st_size > 0:
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"  Downloaded: {output_file.name} ({size_mb:.1f} MB)")
            else:
                logger.error(f"  Download failed or empty file: {output_file}")

        except Exception as e:
            logger.error(f"  Error downloading {year}: {e}")
            # Continue with other years
            continue

    # Verify all files downloaded
    downloaded = list(output_dir.glob('era5_cloud_cover_wyoming_*.nc'))
    logger.info(f"\n=== Download Complete ===")
    logger.info(f"Files downloaded: {len(downloaded)}/{len(years)}")

    total_size = sum(f.stat().st_size for f in downloaded) / (1024 * 1024)
    logger.info(f"Total size: {total_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 cloud cover data for Wyoming"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without downloading'
    )
    parser.add_argument(
        '--year-start',
        type=int,
        default=None,
        help='First year to download (default: auto-detect from datasets)'
    )
    parser.add_argument(
        '--year-end',
        type=int,
        default=None,
        help='Last year to download (default: auto-detect from datasets)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Path to data directory (default: data)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = data_dir / 'era5'

    # Check CDS API setup
    if not args.dry_run and not check_cdsapi_setup():
        return 1

    # Determine date range
    if args.year_start is None or args.year_end is None:
        logger.info("Auto-detecting date range from datasets...")
        auto_start, auto_end = get_date_range_from_datasets(data_dir)
        year_start = args.year_start or auto_start
        year_end = args.year_end or auto_end
    else:
        year_start = args.year_start
        year_end = args.year_end

    logger.info(f"Date range: {year_start} to {year_end}")

    # Download
    download_era5_cloud_cover(
        output_dir=output_dir,
        year_start=year_start,
        year_end=year_end,
        dry_run=args.dry_run
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
