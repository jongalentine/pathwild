#!/usr/bin/env python3
"""
Bulk pre-cache SNOTEL snow data for all dates and stations needed by training datasets.

This script pre-fetches SNOTEL data from the AWDB API for all Wyoming stations
and date ranges needed by the training datasets, storing results in the SQLite cache.
This eliminates API calls during pipeline execution.

Usage:
    python scripts/bulk_precache_snotel.py [--dry-run] [--workers N]

Examples:
    # Show what would be fetched (dry run)
    python scripts/bulk_precache_snotel.py --dry-run

    # Fetch with 4 parallel workers
    python scripts/bulk_precache_snotel.py --workers 4

    # Fetch sequentially (safest, avoids rate limits)
    python scripts/bulk_precache_snotel.py --workers 1
"""

import argparse
import logging
import sys
import time
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWDB REST API configuration
AWDB_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
ELEMENTS = ['WTEQ', 'SNWD']  # Snow Water Equivalent, Snow Depth


def load_stations(cache_dir: Path) -> List[Dict]:
    """
    Load Wyoming SNOTEL stations from cached GeoJSON.

    Returns:
        List of station dictionaries with station_id, triplet, name, lat, lon
    """
    import geopandas as gpd

    stations_file = cache_dir / 'snotel_stations_wyoming.geojson'

    if not stations_file.exists():
        logger.error(f"Stations file not found: {stations_file}")
        logger.error("Run the pipeline once to generate the stations cache, or fetch from API")
        return []

    gdf = gpd.read_file(stations_file)

    stations = []
    for _, row in gdf.iterrows():
        station_id = row.get('awdb_station_id') or row.get('station_id')
        if station_id:
            stations.append({
                'station_id': str(station_id),
                'triplet': row.get('triplet', f"{station_id}:WY:SNTL"),
                'name': row.get('name', 'Unknown'),
                'lat': row.get('lat'),
                'lon': row.get('lon'),
            })

    return stations


def collect_date_range(processed_dir: Path) -> Tuple[datetime, datetime]:
    """
    Determine the full date range needed across all datasets.

    Returns:
        Tuple of (min_date, max_date)
    """
    all_dates = []

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

        if len(dates) > 0:
            all_dates.extend(dates.tolist())
            logger.info(f"{f.name}: {len(dates)} rows, range {dates.min().date()} to {dates.max().date()}")

    if not all_dates:
        logger.warning("No dates found in datasets")
        return None, None

    # Extend by 30 days on each side for caching buffer
    min_date = min(all_dates) - timedelta(days=30)
    max_date = max(all_dates) + timedelta(days=30)

    return min_date, max_date


def make_cache_key(station_id: str, begin_date: str, end_date: str) -> str:
    """Generate cache key matching SNOTELDataCache format."""
    key_data = f"{station_id}:{begin_date}:{end_date}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def get_cached_entries(cache_db: Path) -> Set[Tuple[str, str, str]]:
    """
    Get all existing cache entries.

    Returns:
        Set of (station_id, begin_date, end_date) tuples
    """
    if not cache_db.exists():
        return set()

    cached = set()
    conn = sqlite3.connect(str(cache_db), timeout=30.0)
    try:
        cursor = conn.execute("""
            SELECT station_id, begin_date, end_date
            FROM snotel_station_data
        """)
        for row in cursor.fetchall():
            cached.add((row[0], row[1], row[2]))
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        pass
    finally:
        conn.close()

    return cached


def fetch_station_data(
    station_id: str,
    triplet: str,
    begin_date: str,
    end_date: str,
    retry_count: int = 3
) -> Optional[pd.DataFrame]:
    """
    Fetch snow data for a station from AWDB API.

    Args:
        station_id: AWDB station ID
        triplet: Station triplet (e.g., "309:WY:SNTL")
        begin_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_count: Number of retries for failed requests

    Returns:
        DataFrame with columns: date, WTEQ, SNWD, or None if error
    """
    params = {
        'stationTriplets': triplet,
        'elements': ','.join(ELEMENTS),
        'beginDate': begin_date,
        'endDate': end_date,
        'duration': 'DAILY',
        'returnFlags': 'false',
        'returnOriginalValues': 'false',
        'returnSuspectData': 'false',
    }

    for attempt in range(retry_count):
        try:
            response = requests.get(
                f"{AWDB_BASE_URL}/data",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if not data or len(data) == 0:
                return None

            # Parse AWDB response format
            # Response is a list with one entry per station
            station_data = data[0]

            if 'data' not in station_data:
                return None

            # Build DataFrame from response
            records = []
            for element_data in station_data.get('data', []):
                element_code = element_data.get('stationElement', {}).get('elementCode')
                values = element_data.get('values', [])

                for val in values:
                    date_str = val.get('date', '')[:10]  # Trim to YYYY-MM-DD
                    value = val.get('value')

                    if date_str and value is not None:
                        # Find or create record for this date
                        record = next((r for r in records if r['date'] == date_str), None)
                        if record is None:
                            record = {'date': date_str}
                            records.append(record)
                        record[element_code] = value

            if not records:
                return None

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                # Server error, retry with backoff
                delay = 2 ** attempt
                logger.warning(f"Server error for station {station_id}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                logger.warning(f"HTTP error fetching station {station_id}: {e}")
                return None
        except requests.exceptions.RequestException as e:
            delay = 2 ** attempt
            logger.warning(f"Request error for station {station_id}: {e}, retrying in {delay}s...")
            time.sleep(delay)
            continue
        except Exception as e:
            logger.error(f"Error parsing data for station {station_id}: {e}")
            return None

    return None


def save_to_cache(
    cache_db: Path,
    station_id: str,
    begin_date: str,
    end_date: str,
    data: pd.DataFrame
):
    """Save station data to SQLite cache."""
    cache_key = make_cache_key(station_id, begin_date, end_date)

    # Convert DataFrame to JSON
    data_copy = data.copy()
    if 'date' in data_copy.columns:
        data_copy['date'] = data_copy['date'].dt.strftime('%Y-%m-%d')
    data_json = json.dumps(data_copy.to_dict('records'))

    conn = sqlite3.connect(str(cache_db), timeout=30.0)
    try:
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snotel_station_data (
                cache_key TEXT PRIMARY KEY,
                station_id TEXT NOT NULL,
                begin_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                data_json TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_station_dates
            ON snotel_station_data(station_id, begin_date, end_date)
        """)

        # Insert data
        conn.execute("""
            INSERT OR REPLACE INTO snotel_station_data
            (cache_key, station_id, begin_date, end_date, data_json)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, station_id, begin_date, end_date, data_json))
        conn.commit()
    finally:
        conn.close()


def process_station(
    station: Dict,
    begin_date: str,
    end_date: str,
    cache_db: Path,
    dry_run: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single station: fetch data and cache it.

    Returns:
        Tuple of (station_id, success, error_message)
    """
    station_id = station['station_id']
    triplet = station['triplet']
    name = station['name']

    if dry_run:
        return (station_id, True, None)

    # Rate limit: 100ms between requests
    time.sleep(0.1)

    try:
        df = fetch_station_data(station_id, triplet, begin_date, end_date)

        if df is None or len(df) == 0:
            return (station_id, False, "No data returned from API")

        save_to_cache(cache_db, station_id, begin_date, end_date, df)
        return (station_id, True, None)

    except Exception as e:
        return (station_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Bulk pre-cache SNOTEL data from AWDB API'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fetched without actually fetching'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel workers (default: 2, max recommended: 4)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Data directory (default: data)'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    processed_dir = data_dir / 'processed'
    cache_dir = data_dir / 'cache'
    cache_db = cache_dir / 'snotel_data_cache.db'

    # Ensure directories exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load stations
    logger.info("Loading Wyoming SNOTEL stations...")
    stations = load_stations(cache_dir)
    if not stations:
        logger.error("No stations found. Run the pipeline once to generate station cache.")
        return 1
    logger.info(f"Found {len(stations)} Wyoming SNOTEL stations")

    # Determine date range from datasets
    logger.info("Analyzing datasets for date range...")
    min_date, max_date = collect_date_range(processed_dir)
    if min_date is None:
        logger.error("No dates found in datasets")
        return 1

    begin_date = min_date.strftime('%Y-%m-%d')
    end_date = max_date.strftime('%Y-%m-%d')
    logger.info(f"Date range: {begin_date} to {end_date}")

    # Check existing cache entries
    logger.info("Checking existing cache entries...")
    cached_entries = get_cached_entries(cache_db)
    logger.info(f"Found {len(cached_entries)} existing cache entries")

    # Determine which stations need data for this date range
    stations_to_fetch = []
    stations_cached = 0

    for station in stations:
        station_id = station['station_id']
        # Check if we have a cache entry covering this date range
        # Note: We're using exact match here, could be smarter about overlapping ranges
        if (station_id, begin_date, end_date) in cached_entries:
            stations_cached += 1
        else:
            stations_to_fetch.append(station)

    logger.info(f"Stations already cached for this range: {stations_cached}")
    logger.info(f"Stations needing data: {len(stations_to_fetch)}")

    if not stations_to_fetch:
        logger.info("All stations already cached!")
        return 0

    if args.dry_run:
        logger.info("\n=== DRY RUN ===")
        logger.info(f"Would fetch data for {len(stations_to_fetch)} stations:")
        for station in stations_to_fetch[:10]:
            logger.info(f"  - {station['name']} ({station['station_id']})")
        if len(stations_to_fetch) > 10:
            logger.info(f"  ... and {len(stations_to_fetch) - 10} more")
        logger.info(f"\nDate range: {begin_date} to {end_date}")
        logger.info(f"Estimated API calls: {len(stations_to_fetch)}")
        logger.info(f"Estimated time: ~{len(stations_to_fetch) * 0.5:.0f} seconds (with rate limiting)")
        return 0

    # Fetch data for all stations
    logger.info(f"\n=== Starting SNOTEL Data Fetch ===")
    logger.info(f"Stations: {len(stations_to_fetch)}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Date range: {begin_date} to {end_date}")

    success_count = 0
    error_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_station,
                station,
                begin_date,
                end_date,
                cache_db,
                False
            ): station
            for station in stations_to_fetch
        }

        for i, future in enumerate(as_completed(futures), 1):
            station = futures[future]
            station_id, success, error = future.result()

            if success:
                success_count += 1
                logger.info(f"[{i}/{len(stations_to_fetch)}] Cached {station['name']} ({station_id})")
            else:
                error_count += 1
                logger.warning(f"[{i}/{len(stations_to_fetch)}] Failed {station['name']}: {error}")

    elapsed = time.time() - start_time

    logger.info("\n=== Fetch Complete ===")
    logger.info(f"Successful: {success_count}/{len(stations_to_fetch)}")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    # Show final cache stats
    final_entries = get_cached_entries(cache_db)
    logger.info(f"Total cache entries: {len(final_entries)}")

    if cache_db.exists():
        size_mb = cache_db.stat().st_size / (1024 * 1024)
        logger.info(f"Cache size: {size_mb:.2f} MB")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
