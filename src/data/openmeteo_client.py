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

logger = logging.getLogger(__name__)

# Rate limit tracking
_last_request_time = 0
_min_request_interval = 0.1  # Minimum 100ms between requests


class OpenMeteoCache:
    """
    Persistent SQLite cache for OpenMeteo historical weather data.

    Uses SQLite for efficient storage and retrieval. Cache keys are based on:
    - Latitude (rounded to 3 decimal places, ~100m precision)
    - Longitude (rounded to 3 decimal places)
    - Date range (start_date, end_date)

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
        # Round coordinates to 3 decimal places (~100m) to reduce cache misses
        key_data = f"{lat:.3f}:{lon:.3f}:{start_date}:{end_date}"
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

        # Round for storage
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)

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
    """Client for Open-Meteo weather forecast API"""

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
        Make API request with retry logic and rate limit handling.

        Args:
            base_url: API base URL
            params: Request parameters
            max_retries: Maximum retry attempts

        Returns:
            JSON response as dictionary
        """
        global _last_request_time

        for attempt in range(max_retries):
            # Rate limiting: ensure minimum interval between requests
            elapsed = time.time() - _last_request_time
            if elapsed < _min_request_interval:
                time.sleep(_min_request_interval - elapsed)

            try:
                _last_request_time = time.time()
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    # Rate limited - use longer exponential backoff with jitter
                    base_wait = min(60, 5 * (2 ** attempt))  # 5, 10, 20, 40, 60 seconds
                    jitter = random.uniform(0, base_wait * 0.5)  # Add up to 50% jitter
                    wait_time = base_wait + jitter

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limited (429) on attempt {attempt + 1}/{max_retries}. "
                            f"Backing off for {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Rate limited by Open-Meteo after {max_retries} attempts. "
                            f"Consider reducing request frequency or using caching."
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
        timezone: str = "America/Denver"
    ) -> Dict:
        """
        Get historical weather data for a location.

        Uses persistent SQLite cache - historical weather data never changes,
        so subsequent pipeline runs will be much faster.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timezone: Timezone

        Returns:
            Dictionary with historical data
        """
        # Check persistent cache first - historical data never changes
        cached_data = self._persistent_cache.get(lat, lon, start_date, end_date)
        if cached_data is not None:
            logger.debug(f"Cache hit for historical data: ({lat:.3f}, {lon:.3f}) {start_date} to {end_date}")
            return cached_data

        logger.debug(f"Cache miss for historical data: ({lat:.3f}, {lon:.3f}) {start_date} to {end_date}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,cloud_cover_mean",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone
        }

        data = self._make_request(self.HISTORICAL_BASE, params)

        # Store in persistent cache (historical data is immutable)
        self._persistent_cache.put(lat, lon, start_date, end_date, data)

        return data
    
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

