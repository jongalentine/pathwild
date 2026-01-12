"""
Open-Meteo Client for Weather Forecasts

Handles weather forecast data retrieval for inference pipeline.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging
import time
import random
import sqlite3
import threading
import hashlib
import json
import fcntl
import os

logger = logging.getLogger(__name__)

# Default rate limit settings
_DEFAULT_MIN_INTERVAL = 0.5  # 500ms between requests (more conservative)
_DEFAULT_RATE_LIMIT_FILE = Path('/tmp/openmeteo_rate_limit.json')


class CrossProcessRateLimiter:
    """
    Cross-process rate limiter using file-based locking.

    Ensures all worker processes coordinate their API requests to avoid
    overwhelming the OpenMeteo API and triggering rate limits.
    """

    def __init__(
        self,
        state_file: Path = _DEFAULT_RATE_LIMIT_FILE,
        min_interval: float = _DEFAULT_MIN_INTERVAL,
        max_interval: float = 10.0,
        backoff_multiplier: float = 1.5,
        recovery_rate: float = 0.95
    ):
        """
        Initialize cross-process rate limiter.

        Args:
            state_file: Path to shared state file for coordination
            min_interval: Minimum seconds between requests (default: 0.5s)
            max_interval: Maximum seconds between requests when backing off
            backoff_multiplier: How much to increase interval on 429 error
            recovery_rate: How fast to recover (multiply interval by this on success)
        """
        self.state_file = Path(state_file)
        self.lock_file = self.state_file.with_suffix('.lock')
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.backoff_multiplier = backoff_multiplier
        self.recovery_rate = recovery_rate

        # Ensure state file exists
        self._init_state()

    def _init_state(self):
        """Initialize or load rate limit state."""
        if not self.state_file.exists():
            self._write_state({
                'last_request_time': 0,
                'current_interval': self.min_interval,
                'consecutive_429s': 0
            })

    def _read_state(self) -> Dict:
        """Read current state from file."""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'last_request_time': 0,
                'current_interval': self.min_interval,
                'consecutive_429s': 0
            }

    def _write_state(self, state: Dict):
        """Write state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Blocks until enough time has passed since the last request.
        Returns the wait time in seconds.
        """
        # Create lock file if needed
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.lock_file, 'w') as lock_f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

            try:
                state = self._read_state()
                current_time = time.time()
                last_request = state.get('last_request_time', 0)
                interval = state.get('current_interval', self.min_interval)

                # Calculate required wait time
                elapsed = current_time - last_request
                wait_time = max(0, interval - elapsed)

                if wait_time > 0:
                    time.sleep(wait_time)

                # Update last request time
                state['last_request_time'] = time.time()
                self._write_state(state)

                return wait_time
            finally:
                # Release lock
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def report_success(self):
        """Report a successful request - gradually reduce interval."""
        with open(self.lock_file, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state()
                state['consecutive_429s'] = 0

                # Gradually reduce interval on success (but not below minimum)
                current = state.get('current_interval', self.min_interval)
                new_interval = max(self.min_interval, current * self.recovery_rate)
                state['current_interval'] = new_interval

                self._write_state(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def report_rate_limit(self):
        """Report a 429 rate limit - increase interval for all workers."""
        with open(self.lock_file, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state()
                state['consecutive_429s'] = state.get('consecutive_429s', 0) + 1

                # Increase interval (with exponential backoff for consecutive 429s)
                current = state.get('current_interval', self.min_interval)
                consecutive = state['consecutive_429s']

                # More aggressive backoff for repeated 429s
                multiplier = self.backoff_multiplier ** min(consecutive, 4)
                new_interval = min(self.max_interval, current * multiplier)
                state['current_interval'] = new_interval

                logger.warning(
                    f"Rate limit hit (#{consecutive}). "
                    f"Increasing global interval to {new_interval:.2f}s for all workers"
                )

                self._write_state(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def get_current_interval(self) -> float:
        """Get the current request interval."""
        state = self._read_state()
        return state.get('current_interval', self.min_interval)

    def reset(self):
        """Reset rate limiter to default state."""
        self._write_state({
            'last_request_time': 0,
            'current_interval': self.min_interval,
            'consecutive_429s': 0
        })


# Global rate limiter instance (shared via file system)
_rate_limiter: Optional[CrossProcessRateLimiter] = None


def get_rate_limiter() -> CrossProcessRateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = CrossProcessRateLimiter()
    return _rate_limiter


class OpenMeteoCache:
    """
    Persistent SQLite cache for OpenMeteo historical weather data.

    Uses SQLite for efficient storage and retrieval. Cache keys are based on:
    - Latitude (rounded to 2 decimal places, ~1.1km precision)
    - Longitude (rounded to 2 decimal places)
    - Date range (start_date, end_date)

    The 2-decimal precision matches OpenMeteo's native ~1km resolution.
    Finer precision would just increase API calls for identical data.

    Supports range-based lookups: if a cached entry contains the requested
    date range, it will be returned without making a new API call.

    Since historical weather data never changes, cached entries are permanent.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize OpenMeteo data cache.

        Args:
            cache_dir: Directory for cache database. Defaults to data/cache/
        """
        if cache_dir is None:
            cache_dir = Path('data/cache')
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = cache_dir / 'openmeteo_cache.db'
        self._lock = threading.Lock()

        # Initialize database
        self._init_db()

        logger.info(f"OpenMeteo cache initialized: {self.cache_db}")

    def _init_db(self):
        """Initialize cache database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS openmeteo_historical (
                        cache_key TEXT PRIMARY KEY,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_openmeteo_coords_dates
                    ON openmeteo_historical(lat, lon, start_date, end_date)
                """)
                conn.commit()
            finally:
                conn.close()

    def _make_cache_key(self, lat: float, lon: float, start_date: str, end_date: str) -> str:
        """Generate cache key from parameters."""
        # Round coordinates to 2 decimal places (~1.1km) to maximize cache hits.
        # OpenMeteo's native resolution is ~1km, so finer precision doesn't improve
        # weather accuracy - it just increases API calls for identical data.
        key_data = f"{lat:.2f}:{lon:.2f}:{start_date}:{end_date}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Retrieve historical weather data from cache.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with weather data, or None if not in cache
        """
        cache_key = self._make_cache_key(lat, lon, start_date, end_date)

        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("""
                    SELECT data_json
                    FROM openmeteo_historical
                    WHERE cache_key = ?
                """, (cache_key,))

                row = cursor.fetchone()
                if row is None:
                    return None

                return json.loads(row[0])
            finally:
                conn.close()

    def put(self, lat: float, lon: float, start_date: str, end_date: str, data: Dict):
        """
        Store historical weather data in cache.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: Dictionary with weather data
        """
        cache_key = self._make_cache_key(lat, lon, start_date, end_date)
        data_json = json.dumps(data)

        # Round for storage (match cache key precision)
        lat_rounded = round(lat, 2)
        lon_rounded = round(lon, 2)

        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO openmeteo_historical
                    (cache_key, lat, lon, start_date, end_date, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cache_key, lat_rounded, lon_rounded, start_date, end_date, data_json))
                conn.commit()
            finally:
                conn.close()

    def get_containing_range(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Find cached data that contains the requested date range.

        This enables date-range batching: if we've previously fetched a larger
        date range for this location, we can extract the needed dates without
        making a new API call.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date needed (YYYY-MM-DD)
            end_date: End date needed (YYYY-MM-DD)

        Returns:
            Dictionary with weather data for the FULL cached range (caller extracts needed dates),
            or None if no containing range exists
        """
        lat_rounded = round(lat, 2)
        lon_rounded = round(lon, 2)

        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                # Find any cached entry for this location that contains our date range
                cursor = conn.execute("""
                    SELECT data_json, start_date, end_date
                    FROM openmeteo_historical
                    WHERE lat = ? AND lon = ?
                      AND start_date <= ?
                      AND end_date >= ?
                    ORDER BY (julianday(end_date) - julianday(start_date)) DESC
                    LIMIT 1
                """, (lat_rounded, lon_rounded, start_date, end_date))

                row = cursor.fetchone()
                if row is None:
                    return None

                return json.loads(row[0])
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM openmeteo_historical")
                total_entries = cursor.fetchone()[0]

                db_size = self.cache_db.stat().st_size if self.cache_db.exists() else 0
                total_size_mb = db_size / (1024 * 1024)

                return {
                    'total_entries': total_entries,
                    'total_size_mb': round(total_size_mb, 2)
                }
            finally:
                conn.close()


class OpenMeteoClient:
    """
    Client for Open-Meteo weather API.

    IMPORTANT: This client should ONLY be used for:
        - Weather FORECASTS (future dates)
        - Current weather conditions
        - Real-time inference

    DO NOT use for bulk historical data retrieval - the API has aggressive rate
    limits (~10 requests/minute) that make it unsuitable for processing training
    datasets with thousands of points.

    For historical weather data, use:
        - PRISM: Temperature and precipitation (scripts/bulk_download_prism.py)
        - ERA5: Cloud cover (scripts/bulk_download_era5_cloud.py)

    The historical methods are retained for inference on recent dates not yet
    available in PRISM/ERA5, but should not be used in batch processing.
    """

    FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_BASE = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, forecast_days: int = 7, timeout: int = 30, cache_dir: Optional[Path] = None):
        """
        Initialize Open-Meteo client.

        Args:
            forecast_days: Number of forecast days to request (default: 7)
            timeout: Request timeout in seconds
            cache_dir: Directory for persistent cache database. Defaults to data/cache/
        """
        self.forecast_days = forecast_days
        self.timeout = timeout
        self.cache = {}  # In-memory cache for forecast data (changes frequently)
        self._persistent_cache = OpenMeteoCache(cache_dir)  # SQLite cache for historical data
    
    def _make_request(
        self,
        base_url: str,
        params: Dict,
        max_retries: int = 5
    ) -> Dict:
        """
        Make API request with retry logic and cross-process rate limit handling.

        Uses a file-based lock to coordinate rate limiting across all worker
        processes, preventing API rate limit errors (429s).

        Args:
            base_url: API base URL
            params: Request parameters
            max_retries: Maximum retry attempts

        Returns:
            JSON response as dictionary
        """
        rate_limiter = get_rate_limiter()

        for attempt in range(max_retries):
            # Acquire rate limit permission (blocks until allowed)
            # This coordinates across ALL worker processes
            rate_limiter.acquire()

            try:
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Success - report to rate limiter to gradually reduce interval
                rate_limiter.report_success()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    # Rate limited - report to global rate limiter
                    # This increases the interval for ALL workers
                    rate_limiter.report_rate_limit()

                    if attempt < max_retries - 1:
                        # Add additional jitter wait on top of the global backoff
                        extra_wait = random.uniform(1, 3)
                        logger.debug(
                            f"429 on attempt {attempt + 1}/{max_retries}. "
                            f"Global interval now {rate_limiter.get_current_interval():.2f}s. "
                            f"Extra wait: {extra_wait:.1f}s"
                        )
                        time.sleep(extra_wait)
                    else:
                        raise RuntimeError(
                            f"Rate limited by Open-Meteo after {max_retries} attempts. "
                            f"Current global interval: {rate_limiter.get_current_interval():.2f}s"
                        )
                else:
                    # Other HTTP errors - shorter backoff
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt + random.uniform(0, 1)
                        logger.warning(
                            f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Failed to get Open-Meteo data after {max_retries} attempts: {e}")

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to get Open-Meteo data after {max_retries} attempts: {e}")
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        timezone: str = "America/Denver"
    ) -> Dict:
        """
        Get weather forecast for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timezone: Timezone (default: America/Denver for Wyoming)
            
        Returns:
            Dictionary with forecast data
        """
        # Check cache
        cache_key = f"forecast_{lat:.4f}_{lon:.4f}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            # Cache for 1 hour
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_data
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,cloud_cover_mean",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "forecast_days": self.forecast_days,
            "timezone": timezone
        }
        
        data = self._make_request(self.FORECAST_BASE, params)
        
        # Cache result
        self.cache[cache_key] = (data, datetime.now())
        
        return data
    
    def get_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        timezone: str = "America/Denver",
        expand_range: bool = True
    ) -> Dict:
        """
        Get historical weather data for a location.

        Uses persistent SQLite cache with smart date-range batching:
        1. First checks for exact cache match
        2. Then checks for a cached range that contains the requested dates
        3. If fetching new data, expands to quarterly ranges to maximize cache hits

        Coordinates are rounded to 2 decimal places (~1.1km) since that matches
        OpenMeteo's native resolution.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timezone: Timezone
            expand_range: If True, fetch quarterly ranges to maximize cache hits

        Returns:
            Dictionary with historical data
        """
        # Check persistent cache first - exact match
        cached_data = self._persistent_cache.get(lat, lon, start_date, end_date)
        if cached_data is not None:
            logger.debug(f"Cache hit (exact): ({lat:.2f}, {lon:.2f}) {start_date} to {end_date}")
            return cached_data

        # Check for a cached range that contains our dates
        containing_data = self._persistent_cache.get_containing_range(lat, lon, start_date, end_date)
        if containing_data is not None:
            logger.debug(f"Cache hit (containing range): ({lat:.2f}, {lon:.2f}) {start_date} to {end_date}")
            return containing_data

        logger.debug(f"Cache miss: ({lat:.2f}, {lon:.2f}) {start_date} to {end_date}")

        # Determine fetch range - expand to quarterly boundaries for better cache reuse
        if expand_range:
            fetch_start, fetch_end = self._expand_to_quarter(start_date, end_date)
        else:
            fetch_start, fetch_end = start_date, end_date

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fetch_start,
            "end_date": fetch_end,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,cloud_cover_mean",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone
        }

        data = self._make_request(self.HISTORICAL_BASE, params)

        # Store in persistent cache with the expanded range
        self._persistent_cache.put(lat, lon, fetch_start, fetch_end, data)

        return data

    def _expand_to_quarter(self, start_date: str, end_date: str) -> tuple:
        """
        Expand date range to quarterly boundaries for better cache reuse.

        For example, a request for 2018-05-15 to 2018-05-22 becomes
        2018-04-01 to 2018-06-30 (Q2 2018).

        Args:
            start_date: Original start date (YYYY-MM-DD)
            end_date: Original end date (YYYY-MM-DD)

        Returns:
            Tuple of (expanded_start, expanded_end) as YYYY-MM-DD strings
        """
        from datetime import datetime as dt

        start = dt.strptime(start_date, "%Y-%m-%d")
        end = dt.strptime(end_date, "%Y-%m-%d")

        # Find quarter start (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
        quarter_start_month = ((start.month - 1) // 3) * 3 + 1
        expanded_start = dt(start.year, quarter_start_month, 1)

        # Find quarter end
        quarter_end_month = ((end.month - 1) // 3) * 3 + 3
        if quarter_end_month == 12:
            expanded_end = dt(end.year, 12, 31)
        else:
            # Last day of the quarter's final month
            next_quarter = dt(end.year, quarter_end_month + 1, 1)
            expanded_end = next_quarter - timedelta(days=1)

        return expanded_start.strftime("%Y-%m-%d"), expanded_end.strftime("%Y-%m-%d")
    
    def parse_forecast_response(self, data: Dict) -> List[Dict]:
        """
        Parse Open-Meteo forecast response into list of daily records.
        
        Args:
            data: Raw API response
            
        Returns:
            List of dictionaries with daily forecast data
        """
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        temp_mean = daily.get("temperature_2m_mean", [])
        precipitation = daily.get("precipitation_sum", [])
        cloud_cover = daily.get("cloud_cover_mean", [])

        results = []
        for i, date_str in enumerate(dates):
            result = {
                "date": date_str,
                "temp_max_f": temp_max[i] if i < len(temp_max) else None,
                "temp_min_f": temp_min[i] if i < len(temp_min) else None,
                "temp_mean_f": temp_mean[i] if i < len(temp_mean) else None,
                "precipitation_inches": precipitation[i] if i < len(precipitation) else None,
                "cloud_cover_percent": cloud_cover[i] if i < len(cloud_cover) else None
            }
            results.append(result)
        
        return results
    
    def get_forecast_for_location(
        self,
        lat: float,
        lon: float,
        timezone: str = "America/Denver"
    ) -> List[Dict]:
        """
        Get parsed forecast data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timezone: Timezone
            
        Returns:
            List of daily forecast dictionaries
        """
        data = self.get_forecast(lat, lon, timezone)
        return self.parse_forecast_response(data)
    
    def get_forecast_for_date(
        self,
        lat: float,
        lon: float,
        target_date: str,
        timezone: str = "America/Denver"
    ) -> Optional[Dict]:
        """
        Get forecast for a specific future date.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_date: Target date (YYYY-MM-DD)
            timezone: Timezone
            
        Returns:
            Forecast dictionary for the target date, or None if not in forecast range
        """
        forecasts = self.get_forecast_for_location(lat, lon, timezone)

        for forecast in forecasts:
            if forecast["date"] == target_date:
                return forecast

        return None

    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get statistics about the persistent historical data cache.

        Returns:
            Dictionary with 'total_entries' and 'total_size_mb'
        """
        return self._persistent_cache.get_stats()


def reset_rate_limiter():
    """
    Reset the global rate limiter to default state.

    Call this before starting a new pipeline run to ensure
    the rate limiter starts fresh without residual backoff.
    """
    rate_limiter = get_rate_limiter()
    rate_limiter.reset()
    logger.info("Rate limiter reset to default state")


def get_rate_limiter_status() -> Dict:
    """
    Get current status of the global rate limiter.

    Returns:
        Dictionary with current_interval, consecutive_429s, and last_request_time
    """
    rate_limiter = get_rate_limiter()
    state = rate_limiter._read_state()
    return {
        'current_interval_seconds': state.get('current_interval', _DEFAULT_MIN_INTERVAL),
        'consecutive_429s': state.get('consecutive_429s', 0),
        'last_request_time': state.get('last_request_time', 0),
        'min_interval': rate_limiter.min_interval,
        'max_interval': rate_limiter.max_interval
    }

