from typing import Dict, List, Tuple, Optional, Callable, TypeVar, Any
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
import requests
from functools import lru_cache, wraps
import logging
import warnings
import os
import threading
import time
import sqlite3
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Type variable for generic timeout wrapper
T = TypeVar('T')

logger = logging.getLogger(__name__)

# Lazy import rasterio to avoid import errors in sandboxed environments
try:
    import rasterio
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
    # Suppress NotGeoreferencedWarning (non-critical, rasterio handles it with identity transform)
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
except (ImportError, OSError) as e:
    rasterio = None
    mask = None
    RASTERIO_AVAILABLE = False
    logger.warning(f"rasterio not available: {e}. Some features may be limited.")

# Lazy import earthengine-api to avoid import errors if not installed
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    ee = None
    EE_AVAILABLE = False
    logger.debug("earthengine-api not available. GEE features will be disabled.")


# ============================================================================
# Google Earth Engine Utilities
# ============================================================================

class GEEInitializer:
    """Thread-safe singleton for GEE initialization"""
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _initialized_project = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, project: str = 'ee-jongalentine'):
        """Initialize GEE (thread-safe, idempotent)"""
        if not EE_AVAILABLE:
            raise ImportError("earthengine-api not installed. Install with: pip install earthengine-api")
        
        with self._lock:
            # If already initialized with the same project, skip
            if self._initialized and self._initialized_project == project:
                logger.debug(f"GEE already initialized with project: {project}")
                return
            
            # If initialized with different project, log warning but don't re-initialize
            # (GEE doesn't support changing projects after initialization)
            if self._initialized and self._initialized_project != project:
                logger.warning(
                    f"GEE already initialized with project '{self._initialized_project}'. "
                    f"Cannot switch to '{project}'. Using existing initialization."
                )
                return
            
            # Not initialized yet, try to initialize
            try:
                ee.Initialize(project=project)
                self._initialized = True
                self._initialized_project = project
                logger.info(f"GEE initialized: {project}")
            except Exception as e:
                logger.error(f"GEE init failed for project '{project}': {e}")
                # Don't set _initialized = True on failure, so we can retry
                raise
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def initialized_project(self) -> Optional[str]:
        """Get the project that GEE was initialized with"""
        return self._initialized_project


def retry_on_ee_exception(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator for GEE operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a "no images" error - don't retry these
                    error_str = str(e).lower()
                    if 'empty' in error_str or 'no images' in error_str or 'collection is empty' in error_str:
                        # No images available - don't retry, just return None
                        raise

                    # Check if it's an EE exception
                    if EE_AVAILABLE and hasattr(ee, 'EEException') and isinstance(e, ee.EEException):
                        if attempt == max_retries - 1:
                            raise
                        # Reduce delay for retries to speed up processing
                        retry_delay = delay * (1.5 ** attempt)  # Reduced from 2.0 to 1.5
                        logger.debug(
                            f"GEE operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay)
                    else:
                        # Not an EE exception, don't retry
                        raise
            return None
        return wrapper
    return decorator


class GEECircuitBreaker:
    """
    Circuit breaker pattern for Google Earth Engine API calls.

    Prevents cascade failures when GEE is experiencing issues (rate limiting,
    outages, network problems). After a threshold of consecutive failures,
    the circuit "opens" and fails fast for a cooldown period before retrying.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, fail fast without making requests
    - HALF_OPEN: Testing if service recovered, allow limited requests

    Thread-safe implementation using locks.
    """

    # Singleton instance
    _instance = None
    _lock = threading.Lock()

    # Circuit states
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 300.0,  # 5 minutes
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            reset_timeout: Seconds to wait before attempting recovery (half-open state)
            half_open_max_calls: Number of test calls allowed in half-open state
        """
        if self._initialized:
            return

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._state_lock = threading.Lock()
        self._initialized = True

        logger.info(
            f"GEE Circuit Breaker initialized: "
            f"failure_threshold={failure_threshold}, "
            f"reset_timeout={reset_timeout}s"
        )

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._state_lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current consecutive failure count."""
        with self._state_lock:
            return self._failure_count

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._state_lock:
            if self._state == self.CLOSED:
                return True

            if self._state == self.OPEN:
                # Check if we should transition to half-open
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.reset_timeout:
                        logger.info(
                            f"GEE Circuit Breaker: transitioning from OPEN to HALF_OPEN "
                            f"after {elapsed:.1f}s cooldown"
                        )
                        self._state = self.HALF_OPEN
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == self.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return True

    def record_success(self):
        """Record a successful request."""
        with self._state_lock:
            if self._state == self.HALF_OPEN:
                # Success in half-open means service recovered
                logger.info("GEE Circuit Breaker: service recovered, closing circuit")
                self._state = self.CLOSED

            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        """
        Record a failed request.

        Args:
            error: The exception that caused the failure (for logging)
        """
        with self._state_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Failure in half-open means service still down
                logger.warning(
                    f"GEE Circuit Breaker: failure in HALF_OPEN state, reopening circuit. "
                    f"Error: {error}"
                )
                self._state = self.OPEN
                self._half_open_calls = 0

            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"GEE Circuit Breaker: {self._failure_count} consecutive failures, "
                        f"opening circuit for {self.reset_timeout}s. Last error: {error}"
                    )
                    self._state = self.OPEN

    def reset(self):
        """Reset circuit breaker to initial state (for testing)."""
        with self._state_lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info("GEE Circuit Breaker: manually reset to CLOSED state")

    def get_status(self) -> Dict[str, any]:
        """Get circuit breaker status for monitoring."""
        with self._state_lock:
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self._last_failure_time,
                "reset_timeout": self.reset_timeout,
                "time_until_retry": (
                    max(0, self.reset_timeout - (time.time() - self._last_failure_time))
                    if self._last_failure_time and self._state == self.OPEN
                    else 0
                )
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request cannot proceed."""
    pass


class GEETimeoutError(Exception):
    """Raised when a GEE API call times out."""
    pass


# Default timeout for GEE API calls (in seconds)
GEE_API_TIMEOUT = 60  # 1 minute


def gee_with_timeout(
    func: Callable[[], T],
    timeout: float = GEE_API_TIMEOUT,
    operation_name: str = "GEE operation"
) -> T:
    """
    Execute a GEE operation with a timeout.

    This is a thread-safe wrapper that runs the operation in a daemon thread
    and enforces a timeout. Safe for use in parallel processing contexts.

    Args:
        func: Zero-argument callable that performs the GEE operation
        timeout: Maximum seconds to wait for completion (default: 120)
        operation_name: Description for logging purposes

    Returns:
        The result of func()

    Raises:
        GEETimeoutError: If the operation times out
        Exception: Any exception raised by func() is re-raised
    """
    # Use a daemon thread so timed-out GEE calls don't prevent process exit.
    # Daemon must be set before start(); setting it on an already-running thread raises RuntimeError.
    result_holder: List[Optional[T]] = [None]
    exc_holder: List[Optional[BaseException]] = [None]

    def run() -> None:
        try:
            result_holder[0] = func()
        except BaseException as e:
            exc_holder[0] = e

    thread = threading.Thread(target=run, daemon=True, name="gee_timeout")
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.error(
            f"{operation_name} timed out after {timeout}s. "
            f"The GEE server may be unresponsive."
        )
        raise GEETimeoutError(
            f"{operation_name} timed out after {timeout} seconds"
        )
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


# ============================================================================
# NDVI Cache for Google Earth Engine
# ============================================================================

class NDVICache:
    """
    Persistent cache for NDVI data retrieved from Google Earth Engine.
    
    Uses SQLite for efficient storage and retrieval. Cache keys are based on:
    - Location (lat, lon rounded to 5 decimal places ~1.1m precision)
    - Date (to the day)
    - Collection name
    - buffer_days
    - max_cloud_cover
    
    Since historical data never changes, cached entries are permanent.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize NDVI cache.
        
        Args:
            cache_dir: Directory for cache database. Defaults to data/cache/ndvi_cache.db
        """
        if cache_dir is None:
            cache_dir = Path('data/cache')
        else:
            cache_dir = Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = cache_dir / 'ndvi_cache.db'
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        logger.info(f"NDVI cache initialized: {self.cache_db}")
    
    def _init_db(self):
        """Initialize cache database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                # Create table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ndvi_cache (
                        cache_key TEXT PRIMARY KEY,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        date TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        buffer_days INTEGER NOT NULL,
                        max_cloud_cover REAL NOT NULL,
                        ndvi REAL,
                        cloud_cover REAL,
                        image_date TEXT,
                        ndvi_age_days REAL,
                        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Create index separately (SQLite doesn't support inline INDEX in CREATE TABLE)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_location_date 
                    ON ndvi_cache(lat, lon, date)
                """)
                conn.commit()
            finally:
                conn.close()
    
    def _make_cache_key(
        self,
        lat: float,
        lon: float,
        date: datetime,
        collection: str,
        buffer_days: int,
        max_cloud_cover: float
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            lat: Latitude (rounded to 5 decimal places)
            lon: Longitude (rounded to 5 decimal places)
            date: Observation date
            collection: Satellite collection name
            buffer_days: Buffer days parameter
            max_cloud_cover: Max cloud cover parameter
            
        Returns:
            Cache key string
        """
        # Round coordinates to 5 decimal places (~1.1m precision)
        lat_rounded = round(lat, 5)
        lon_rounded = round(lon, 5)
        date_str = date.strftime('%Y-%m-%d')
        
        # Create hash for cache key
        key_data = f"{lat_rounded:.5f},{lon_rounded:.5f},{date_str},{collection},{buffer_days},{max_cloud_cover:.1f}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(
        self,
        lat: float,
        lon: float,
        date: datetime,
        collection: str,
        buffer_days: int,
        max_cloud_cover: float
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Retrieve NDVI data from cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Observation date
            collection: Satellite collection name
            buffer_days: Buffer days parameter
            max_cloud_cover: Max cloud cover parameter
            
        Returns:
            Dict with 'ndvi', 'cloud_cover', 'image_date', 'ndvi_age_days' keys,
            or None if not in cache
        """
        cache_key = self._make_cache_key(lat, lon, date, collection, buffer_days, max_cloud_cover)
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("""
                    SELECT ndvi, cloud_cover, image_date, ndvi_age_days
                    FROM ndvi_cache
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                ndvi, cloud_cover, image_date_str, ndvi_age_days = row
                
                # Parse image_date if present
                image_date = None
                if image_date_str:
                    try:
                        image_date = datetime.fromisoformat(image_date_str)
                    except (ValueError, TypeError):
                        pass
                
                return {
                    'ndvi': float(ndvi) if ndvi is not None else None,
                    'cloud_cover': float(cloud_cover) if cloud_cover is not None else None,
                    'image_date': image_date,
                    'ndvi_age_days': float(ndvi_age_days) if ndvi_age_days is not None else None
                }
            finally:
                conn.close()
    
    def put(
        self,
        lat: float,
        lon: float,
        date: datetime,
        collection: str,
        buffer_days: int,
        max_cloud_cover: float,
        result: Dict[str, Optional[float]]
    ):
        """
        Store NDVI data in cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Observation date
            collection: Satellite collection name
            buffer_days: Buffer days parameter
            max_cloud_cover: Max cloud cover parameter
            result: Dict with 'ndvi', 'cloud_cover', 'image_date', 'ndvi_age_days' keys
        """
        cache_key = self._make_cache_key(lat, lon, date, collection, buffer_days, max_cloud_cover)
        
        # Round coordinates for storage
        lat_rounded = round(lat, 5)
        lon_rounded = round(lon, 5)
        date_str = date.strftime('%Y-%m-%d')
        
        # Serialize image_date
        image_date_str = None
        if result.get('image_date'):
            image_date_str = result['image_date'].isoformat()
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO ndvi_cache
                    (cache_key, lat, lon, date, collection, buffer_days, max_cloud_cover,
                     ndvi, cloud_cover, image_date, ndvi_age_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    lat_rounded,
                    lon_rounded,
                    date_str,
                    collection,
                    buffer_days,
                    max_cloud_cover,
                    result.get('ndvi'),
                    result.get('cloud_cover'),
                    image_date_str,
                    result.get('ndvi_age_days')
                ))
                conn.commit()
            finally:
                conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with 'total_entries', 'total_size_mb' keys
        """
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM ndvi_cache")
                total_entries = cursor.fetchone()[0]
                
                # Get database file size
                db_size = self.cache_db.stat().st_size if self.cache_db.exists() else 0
                total_size_mb = db_size / (1024 * 1024)
                
                return {
                    'total_entries': total_entries,
                    'total_size_mb': round(total_size_mb, 2)
                }
            finally:
                conn.close()
    
    def clear(self):
        """Clear all cached entries (use with caution)."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("DELETE FROM ndvi_cache")
                conn.commit()
                logger.info("NDVI cache cleared")
            finally:
                conn.close()


# Google Earth Engine NDVI Client
# ============================================================================

class GEENDVIClient:
    """
    Google Earth Engine NDVI client.
    Follows DataContextBuilder pattern for consistency with other data sources.
    """
    
    COLLECTIONS = {
        'landsat5': {
            'id': 'LANDSAT/LT05/C02/T1_L2',
            'red_band': 'SR_B3',
            'nir_band': 'SR_B4',
            'scale': 30,
            'start_date': '1984-03-01',
            'end_date': '2013-05-30'
        },
        'landsat8': {
            'id': 'LANDSAT/LC08/C02/T1_L2',
            'red_band': 'SR_B4',
            'nir_band': 'SR_B5',
            'scale': 30,
            'start_date': '2013-04-11'
        },
        'sentinel2': {
            'id': 'COPERNICUS/S2_SR_HARMONIZED',
            'red_band': 'B4',
            'nir_band': 'B8',
            'scale': 10,
            'start_date': '2017-03-28'
        }
    }
    
    def __init__(
        self,
        project: str = 'ee-jongalentine',
        collection: str = 'landsat8',
        batch_size: int = 100,
        max_workers: Optional[int] = 4,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize GEE NDVI client.
        
        Uses hybrid collection selection: automatically selects Landsat 5 for dates < 2013
        and Landsat 8 for dates >= 2013. This eliminates placeholder NDVI values for
        historical years (2006-2012).
        
        Args:
            project: GEE project ID
            collection: Default satellite collection ('landsat8' or 'sentinel2').
                       Note: If 'landsat8' (default), the client will automatically use
                       Landsat 5 for dates < 2013 and Landsat 8 for dates >= 2013.
            batch_size: Points per batch for processing
            max_workers: Thread pool size for parallel processing
            cache_dir: Directory for NDVI cache (defaults to data/cache)
            use_cache: If True, use persistent cache for historical data (default: True)
        """
        if not EE_AVAILABLE:
            raise ImportError("earthengine-api not installed. Install with: pip install earthengine-api")
        
        # Validate collection BEFORE initializing GEE (faster failure for invalid collections)
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}. Must be one of: {list(self.COLLECTIONS.keys())}")
        
        # Initialize GEE (singleton pattern)
        GEEInitializer().initialize(project)

        self.collection_config = self.COLLECTIONS[collection]
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.collection = collection
        self.use_cache = use_cache

        # Initialize circuit breaker (singleton pattern)
        # This prevents cascade failures when GEE is having issues
        self.circuit_breaker = GEECircuitBreaker(
            failure_threshold=5,      # Open circuit after 5 consecutive failures
            reset_timeout=300.0,      # Wait 5 minutes before retrying
            half_open_max_calls=3     # Allow 3 test calls in half-open state
        )

        # Initialize cache if enabled
        if self.use_cache:
            self.cache = NDVICache(cache_dir=cache_dir)
            cache_stats = self.cache.get_stats()
            logger.info(
                f"NDVI cache enabled: {cache_stats['total_entries']:,} entries "
                f"({cache_stats['total_size_mb']:.2f} MB)"
            )
        else:
            self.cache = None
            logger.info("NDVI cache disabled")

        logger.info(
            f"GEENDVIClient initialized: {collection}, "
            f"batch_size={batch_size}, workers={max_workers}, cache={'enabled' if use_cache else 'disabled'}"
        )
    
    def _get_collection_for_date(self, date: datetime) -> Dict:
        """
        Select collection based on date: Landsat 5 for <2013-02-11, Landsat 8 for >=2013-02-11.
        
        This enables hybrid collection selection to provide real NDVI data for
        historical years (2006-2012) using Landsat 5, eliminating placeholder values.
        
        Landsat 8 launched on 2013-02-11, so dates on or after this date use L8.
        
        Args:
            date: Date to determine collection for
            
        Returns:
            Dict with collection configuration (same structure as COLLECTIONS entries)
        """
        # Landsat 8 launched on 2013-02-11, so use L8 for dates on or after this date
        landsat8_launch_date = datetime(2013, 2, 11)
        if date < landsat8_launch_date:
            return self.COLLECTIONS['landsat5']
        else:
            return self.COLLECTIONS['landsat8']
    
    def _get_collection_name_for_date(self, date: datetime) -> str:
        """
        Get collection name (for cache keys) based on date.
        
        Landsat 8 launched on 2013-02-11, so dates on or after this date use L8.
        
        Args:
            date: Date to determine collection for
            
        Returns:
            Collection name string ('landsat5' or 'landsat8')
        """
        landsat8_launch_date = datetime(2013, 2, 11)
        if date < landsat8_launch_date:
            return 'landsat5'
        else:
            return 'landsat8'
    
    def check_availability(self) -> Dict[str, bool]:
        """Check GEE connection"""
        if not EE_AVAILABLE:
            return {
                'gee_available': False,
                'error': 'earthengine-api not installed'
            }
        
        try:
            test_point = ee.Geometry.Point([-107.25, 44.5])
            collection = ee.ImageCollection(self.collection_config['id']) \
                .filterBounds(test_point) \
                .limit(1)
            count = gee_with_timeout(
                lambda: collection.size().getInfo(),
                timeout=GEE_API_TIMEOUT,
                operation_name="GEE availability check"
            )

            return {
                'gee_initialized': True,
                'collection_accessible': True,
                'test_successful': count >= 0
            }
        except (GEETimeoutError, Exception) as e:
            logger.error(f"GEE availability check failed: {e}")
            return {
                'gee_initialized': GEEInitializer().is_initialized,
                'collection_accessible': False,
                'error': str(e)
            }
    
    @retry_on_ee_exception(max_retries=2, delay=1.0)  # Reduced retries and delay for faster processing
    def _get_ndvi_single_point(
        self,
        lat: float,
        lon: float,
        date: datetime,
        buffer_days: int,
        max_cloud_cover: float
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Get NDVI, cloud cover, and image age for single point (internal, with retry).
        
        Checks cache first, then queries GEE if not cached.
        Uses hybrid collection selection: Landsat 5 for <2013, Landsat 8 for >=2013.
        
        Returns:
            Dict with 'ndvi' (float), 'cloud_cover' (float or None), 
            'image_date' (datetime or None), and 'ndvi_age_days' (float or None) keys,
            or None if no data available
        """
        # Get collection config and name based on date (hybrid selection)
        collection_config = self._get_collection_for_date(date)
        collection_name = self._get_collection_name_for_date(date)
        
        # Use adaptive buffer_days: larger for Landsat 5 (16-day revisit) vs Landsat 8 (8-day revisit)
        # If buffer_days is provided, use it; otherwise use collection-specific defaults
        if collection_name == 'landsat5':
            # Landsat 5 has 16-day revisit, so use larger buffer to find images
            effective_buffer_days = buffer_days if buffer_days > 0 else 21  # Default: 21 days (covers ~1.3 revisit cycles)
        else:
            # Landsat 8 has 8-day revisit, smaller buffer is sufficient
            effective_buffer_days = buffer_days if buffer_days > 0 else 14  # Default: 14 days (covers ~1.75 revisit cycles)
        
        # DEBUG: Log collection selection
        logger.debug(f"NDVI query for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}: using {collection_name} (collection: {collection_config['id']}), buffer_days={effective_buffer_days}")
        
        # Check cache first (if enabled) using the correct collection name and effective buffer
        if self.use_cache and self.cache:
            cached_result = self.cache.get(
                lat=lat,
                lon=lon,
                date=date,
                collection=collection_name,
                buffer_days=effective_buffer_days,
                max_cloud_cover=max_cloud_cover
            )
            if cached_result is not None:
                logger.debug(f"Cache hit for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')} using {collection_name}")
                return cached_result

        # Check circuit breaker before making GEE API calls
        if not self.circuit_breaker.can_execute():
            status = self.circuit_breaker.get_status()
            logger.debug(
                f"Circuit breaker OPEN, skipping GEE call for ({lat:.5f}, {lon:.5f}). "
                f"Retry in {status['time_until_retry']:.0f}s"
            )
            raise CircuitBreakerOpenError(
                f"GEE circuit breaker is open after {status['failure_count']} failures. "
                f"Will retry in {status['time_until_retry']:.0f}s"
            )

        # Not in cache, query GEE
        point = ee.Geometry.Point([lon, lat])
        
        # Date range using effective buffer
        start_date = (date - timedelta(days=effective_buffer_days)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=effective_buffer_days)).strftime('%Y-%m-%d')
        
        # Filter collection using date-based collection config
        collection = ee.ImageCollection(collection_config['id']) \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
        
        # Get best image (lowest cloud cover) - skip size check to save API call
        # If no images exist, image.getInfo() will fail quickly
        image = collection.sort('CLOUD_COVER').first()

        # Check existence and get metadata
        # This will fail fast if no images exist, avoiding the extra collection.size() API call
        try:
            image_info = gee_with_timeout(
                lambda: image.getInfo(),
                timeout=GEE_API_TIMEOUT,
                operation_name=f"image.getInfo() for {collection_name} at ({lat:.5f}, {lon:.5f})"
            )
            if not image_info:
                logger.warning(f"image.getInfo() returned empty result for {collection_name} at ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}")
                return None
            
            properties = image_info.get('properties', {})
            
            # Extract cloud cover from image properties
            cloud_cover = properties.get('CLOUD_COVER', None)
            if cloud_cover is None:
                # Try alternative property names
                cloud_cover = properties.get('cloud_cover', None)
            
            # Extract image timestamp (system:time_start is in milliseconds since epoch)
            image_timestamp_ms = properties.get('system:time_start', None)
            image_date = None
            ndvi_age_days = None
            
            if image_timestamp_ms:
                # Convert milliseconds to datetime
                image_date = datetime.fromtimestamp(image_timestamp_ms / 1000.0)
                # Calculate age in days (positive = image is older than observation date)
                age_delta = date - image_date
                ndvi_age_days = age_delta.total_seconds() / 86400.0  # Convert to days
                # Handle images slightly after observation date (expected when using buffer_days)
                if ndvi_age_days < 0:
                    # Image is after observation date (expected when buffer_days allows future images)
                    logger.debug(f"Image date ({image_date}) is after observation date ({date}), "
                               f"age: {abs(ndvi_age_days):.2f} days (within {effective_buffer_days}-day buffer)")
                    ndvi_age_days = abs(ndvi_age_days)
            
        except Exception as e:
            # Check if this is a "no images" error - common and expected, don't log as warning
            error_str = str(e).lower()
            if 'empty' in error_str or 'no images' in error_str or 'collection is empty' in error_str:
                logger.debug(f"No images found in {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')} (date range: {start_date} to {end_date}, buffer: {effective_buffer_days} days, max_cloud_cover: {max_cloud_cover}%)")
                # "No images" is expected, not a GEE failure - don't count against circuit breaker
            else:
                logger.warning(f"Error extracting image metadata from {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}: {e}")
                # Record failure for circuit breaker (actual GEE error)
                self.circuit_breaker.record_failure(e)
            return None
        
        # Calculate NDVI using date-based collection config
        red_band = collection_config['red_band']
        nir_band = collection_config['nir_band']
        ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
        
        # Sample at point
        scale = collection_config['scale']
        value = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=scale
        ).get('NDVI')

        try:
            ndvi_value = gee_with_timeout(
                lambda: value.getInfo(),
                timeout=GEE_API_TIMEOUT,
                operation_name=f"NDVI value.getInfo() for ({lat:.5f}, {lon:.5f})"
            )
        except GEETimeoutError as e:
            logger.warning(f"Timeout getting NDVI value from {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}: {e}")
            self.circuit_breaker.record_failure(e)
            return None
        except Exception as e:
            # Check if this is a "no images" error - common and expected
            error_str = str(e).lower()
            if 'empty' in error_str or 'no images' in error_str or 'collection is empty' in error_str:
                logger.debug(f"No images found in {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}")
                # "No images" is expected, not a GEE failure - don't count against circuit breaker
            else:
                logger.warning(f"Error getting NDVI value from {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}: {e}")
                # Record failure for circuit breaker (actual GEE error)
                self.circuit_breaker.record_failure(e)
            return None
        
        if ndvi_value is None:
            logger.warning(f"NDVI value is None from {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}")
            result = None
        else:
            result = {
                'ndvi': float(ndvi_value),
                'cloud_cover': float(cloud_cover) if cloud_cover is not None else None,
                'image_date': image_date,
                'ndvi_age_days': float(ndvi_age_days) if ndvi_age_days is not None else None
            }
            # Record success for circuit breaker - GEE API call succeeded
            self.circuit_breaker.record_success()
            logger.debug(f"Successfully retrieved NDVI={ndvi_value:.3f} from {collection_name} for ({lat:.5f}, {lon:.5f}) on {date.strftime('%Y-%m-%d')}")
        
        # Store in cache (even if None, to avoid repeated failed queries) using correct collection name and effective buffer
        if self.use_cache and self.cache:
            self.cache.put(
                lat=lat,
                lon=lon,
                date=date,
                collection=collection_name,
                buffer_days=effective_buffer_days,
                max_cloud_cover=max_cloud_cover,
                result=result if result is not None else {
                    'ndvi': None,
                    'cloud_cover': None,
                    'image_date': None,
                    'ndvi_age_days': None
                }
            )
        
        return result
    
    def _calculate_irg(
        self,
        lat: float,
        lon: float,
        current_date: datetime,
        current_ndvi: float,
        buffer_days: int,
        max_cloud_cover: float,
        lookback_days: int = 21
    ) -> Optional[float]:
        """
        Calculate Instantaneous Rate of Green-up (IRG) by comparing current NDVI
        with NDVI from lookback_days ago.
        
        IRG = (current_ndvi - past_ndvi) / days_between
        
        Args:
            lat: Latitude
            lon: Longitude
            current_date: Current observation date
            current_ndvi: Current NDVI value
            buffer_days: Days to search around target date
            max_cloud_cover: Maximum cloud cover threshold
            lookback_days: Days to look back for past NDVI (default 21 = 3 weeks)
            
        Returns:
            IRG value (positive = greening, negative = browning), or None if can't calculate
        """
        # Calculate past date
        past_date = current_date - timedelta(days=lookback_days)
        
        # Fetch past NDVI
        past_result = self._get_ndvi_single_point(
            lat=lat,
            lon=lon,
            date=past_date,
            buffer_days=buffer_days,
            max_cloud_cover=max_cloud_cover
        )
        
        if past_result is None or past_result.get('ndvi') is None:
            # Can't calculate IRG without past data
            # Fall back to seasonal approximation
            month = current_date.month
            if month in [4, 5]:  # Spring greening
                return 0.01
            elif month in [9, 10]:  # Fall browning
                return -0.005
            else:
                return 0.0
        
        past_ndvi = past_result.get('ndvi')
        
        # Calculate IRG (rate of change per day)
        days_between = lookback_days
        if days_between <= 0:
            return 0.0
        
        irg = (current_ndvi - past_ndvi) / days_between
        
        return float(irg)
    
    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        date_column: str,
        lat_column: str,
        lon_column: str,
        buffer_days: int,
        max_cloud_cover: float
    ) -> pd.DataFrame:
        """Process batch using thread pool"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for idx, row in batch_df.iterrows():
                future = executor.submit(
                    self._get_ndvi_single_point,
                    lat=row[lat_column],
                    lon=row[lon_column],
                    date=pd.to_datetime(row[date_column]),
                    buffer_days=buffer_days,
                    max_cloud_cover=max_cloud_cover
                )
                futures.append((idx, future))
            
            # Collect results in order
            results = {}
            cloud_cover_results = {}
            age_days_results = {}
            for idx, future in futures:
                try:
                    result = future.result(timeout=60)
                    if result is not None:
                        results[idx] = result.get('ndvi')
                        cloud_cover_results[idx] = result.get('cloud_cover')
                        age_days_results[idx] = result.get('ndvi_age_days')
                    else:
                        results[idx] = None
                        cloud_cover_results[idx] = None
                        age_days_results[idx] = None
                except Exception as e:
                    logger.warning(f"NDVI retrieval failed for index {idx}: {e}")
                    results[idx] = None
                    cloud_cover_results[idx] = None
                    age_days_results[idx] = None
            
            ndvi_values = [results.get(idx) for idx in batch_df.index]
            cloud_cover_values = [cloud_cover_results.get(idx) for idx in batch_df.index]
            age_days_values = [age_days_results.get(idx) for idx in batch_df.index]
        
        batch_df = batch_df.copy()
        batch_df['ndvi'] = ndvi_values
        batch_df['cloud_cover_percent'] = cloud_cover_values
        batch_df['ndvi_age_days'] = age_days_values
        
        # Calculate IRG for points with valid NDVI
        # Use parallel processing for IRG as well to speed things up
        batch_df['irg'] = None
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            irg_futures = {}
            for idx, row in batch_df.iterrows():
                if pd.notna(row['ndvi']) and pd.notna(row[date_column]):
                    future = executor.submit(
                        self._calculate_irg,
                        lat=row[lat_column],
                        lon=row[lon_column],
                        current_date=pd.to_datetime(row[date_column]),
                        current_ndvi=row['ndvi'],
                        buffer_days=buffer_days,
                        max_cloud_cover=max_cloud_cover
                    )
                    irg_futures[idx] = future
            
            # Collect IRG results with timeout
            for idx, future in irg_futures.items():
                try:
                    irg = future.result(timeout=30)  # 30 second timeout per IRG calculation
                    batch_df.at[idx, 'irg'] = irg
                except TimeoutError:
                    logger.warning(f"IRG calculation timeout for index {idx}, using fallback")
                    # Use seasonal fallback
                    month = pd.to_datetime(batch_df.at[idx, date_column]).month
                    batch_df.at[idx, 'irg'] = 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0
                except Exception as e:
                    logger.warning(f"IRG calculation failed for index {idx}: {e}, using fallback")
                    # Use seasonal fallback
                    month = pd.to_datetime(batch_df.at[idx, date_column]).month
                    batch_df.at[idx, 'irg'] = 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0
        
        return batch_df
    
    def get_ndvi_for_points(
        self,
        points: pd.DataFrame,
        date_column: str = 'timestamp',
        lat_column: str = 'latitude',
        lon_column: str = 'longitude',
        buffer_days: int = 7,
        max_cloud_cover: float = 30.0
    ) -> pd.DataFrame:
        """
        Retrieve NDVI for all points.
        Thread-safe batch processing with parallel execution.
        
        Args:
            points: DataFrame with GPS observations
            date_column: Column name for date/timestamp
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            buffer_days: Days before/after target date to search for images
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            DataFrame with 'ndvi' column added
        """
        n_points = len(points)
        
        # Only log at INFO level for batches with multiple points to reduce log spam
        # Single-point calls (from build_context) will be logged at DEBUG level
        if n_points > 1:
            logger.info(f"Retrieving NDVI for {n_points} points using GEE")
        else:
            logger.debug(f"Retrieving NDVI for {n_points} point(s) using GEE")
        
        # Convert dates
        points = points.copy()
        if not pd.api.types.is_datetime64_any_dtype(points[date_column]):
            points[date_column] = pd.to_datetime(points[date_column])
        
        # Process in batches
        results = []
        n_batches = int(np.ceil(n_points / self.batch_size))
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_points)
            batch = points.iloc[start_idx:end_idx]
            
            logger.debug(f"Processing batch {i + 1}/{n_batches} ({len(batch)} points)")
            
            batch_result = self._process_batch(
                batch,
                date_column,
                lat_column,
                lon_column,
                buffer_days,
                max_cloud_cover
            )
            
            results.append(batch_result)
            
            # Log progress for larger batches (every 10 batches or at milestones)
            if n_batches > 10 and (i + 1) % 10 == 0:
                logger.info(f"NDVI progress: {i + 1}/{n_batches} batches processed ({end_idx}/{n_points} points)")
        
        # Combine
        final_df = pd.concat(results, ignore_index=False)
        
        # Statistics
        n_success = final_df['ndvi'].notna().sum()
        success_rate = n_success / len(final_df) * 100
        
        # Log cache statistics if enabled
        cache_info = ""
        if self.use_cache and self.cache:
            cache_stats = self.cache.get_stats()
            cache_info = f" (cache: {cache_stats['total_entries']:,} entries, {cache_stats['total_size_mb']:.2f} MB)"
        
        # Only log completion at INFO level for batches with multiple points
        # Single-point calls (from build_context) will be logged at DEBUG level
        if n_points > 1:
            logger.info(
                f"NDVI retrieval complete: {n_success}/{len(final_df)} "
                f"({success_rate:.1f}% success){cache_info}"
            )
        else:
            logger.debug(
                f"NDVI retrieval complete: {n_success}/{len(final_df)} "
                f"({success_rate:.1f}% success){cache_info}"
            )
        
        return final_df
    
    def get_summer_integrated_ndvi(
        self,
        lat: float,
        lon: float,
        year: int,
        buffer_days: int = 7,
        max_cloud_cover: float = 30.0
    ) -> Optional[float]:
        """
        Calculate summer integrated NDVI (sum of NDVI values from June-September).
        
        This is used for predicting pre-winter body condition and winterkill risk.
        Integrated NDVI = sum of NDVI values over the summer period.
        Uses hybrid collection selection: Landsat 5 for <2013, Landsat 8 for >=2013.
        
        Args:
            lat: Latitude
            lon: Longitude
            year: Year to calculate summer NDVI for
            buffer_days: Days to search around each target date
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            Integrated NDVI value (sum of NDVI over summer), or None if insufficient data
        """
        # Check circuit breaker before making GEE API calls
        if not self.circuit_breaker.can_execute():
            status = self.circuit_breaker.get_status()
            logger.debug(
                f"Circuit breaker OPEN, skipping summer NDVI for ({lat:.5f}, {lon:.5f}) year {year}. "
                f"Retry in {status['time_until_retry']:.0f}s"
            )
            raise CircuitBreakerOpenError(
                f"GEE circuit breaker is open. Will retry in {status['time_until_retry']:.0f}s"
            )

        point = ee.Geometry.Point([lon, lat])
        
        # Summer period: June 1 to September 30
        start_date = datetime(year, 6, 1)
        end_date = datetime(year, 9, 30)
        
        # Get collection config based on year (hybrid selection)
        collection_config = self._get_collection_for_date(start_date)
        collection_name = self._get_collection_name_for_date(start_date)
        
        # DEBUG: Log collection selection
        logger.debug(f"Summer integrated NDVI for ({lat:.5f}, {lon:.5f}) year {year}: using {collection_name} (collection: {collection_config['id']})")
        
        # Filter collection for summer period using date-based collection config
        collection = ee.ImageCollection(collection_config['id']) \
            .filterBounds(point) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)) \
            .sort('system:time_start')  # Sort by date for consistent sampling

        # Optimize: Limit to max 8 images to reduce API calls
        # This captures ~4 months of summer with Landsat 16-day revisit (typically 7-8 images)
        # For Sentinel-2 (5-day revisit), this still gives good coverage
        collection = collection.limit(8)
        
        # Calculate NDVI for all images in the collection using date-based collection config
        red_band = collection_config['red_band']
        nir_band = collection_config['nir_band']
        scale = collection_config['scale']
        
        # Map over collection to calculate NDVI for each image
        def calculate_ndvi(image):
            ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
            # Sample at point
            ndvi_value = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=scale
            ).get('NDVI')
            return image.set('ndvi_value', ndvi_value)
        
        # Apply NDVI calculation to all images
        ndvi_collection = collection.map(calculate_ndvi)
        
        # Get all NDVI values
        try:
            # Get list of NDVI values with timeout protection
            ndvi_values = gee_with_timeout(
                lambda: ndvi_collection.aggregate_array('ndvi_value').getInfo(),
                timeout=GEE_API_TIMEOUT,
                operation_name=f"summer NDVI aggregate for ({lat:.5f}, {lon:.5f}) year {year}"
            )

            # Filter out None values and calculate sum
            valid_ndvi = [v for v in ndvi_values if v is not None]

            if len(valid_ndvi) == 0:
                logger.debug(f"No valid NDVI data for summer {year} at ({lat}, {lon})")
                return None

            # Integrated NDVI = sum of all NDVI values
            integrated_ndvi = sum(valid_ndvi)

            # Log detailed information for debugging
            logger.debug(
                f"Summer integrated NDVI for {year} at ({lat:.5f}, {lon:.5f}): {integrated_ndvi:.2f} "
                f"(from {len(valid_ndvi)} images, NDVI range: {min(valid_ndvi):.3f}-{max(valid_ndvi):.3f})"
            )

            # Warn if value seems unusually high (suggests calculation issue or too many images)
            if integrated_ndvi > 10:
                logger.warning(
                    f"Summer integrated NDVI ({integrated_ndvi:.2f}) exceeds expected maximum (~8.0) "
                    f"for {year} at ({lat:.5f}, {lon:.5f}). "
                    f"Used {len(valid_ndvi)} images with NDVI values: {[f'{v:.3f}' for v in valid_ndvi[:5]]}"
                    f"{'...' if len(valid_ndvi) > 5 else ''}"
                )

            # Record success for circuit breaker
            self.circuit_breaker.record_success()
            return float(integrated_ndvi)

        except GEETimeoutError as e:
            logger.warning(f"Timeout calculating summer integrated NDVI for {year} at ({lat:.5f}, {lon:.5f}): {e}")
            self.circuit_breaker.record_failure(e)
            return None
        except Exception as e:
            # Check if this is a "no images" type error - expected, don't count against circuit breaker
            error_str = str(e).lower()
            if 'empty' in error_str or 'no images' in error_str or 'collection is empty' in error_str:
                logger.debug(f"No summer images found for {year} at ({lat:.5f}, {lon:.5f}): {e}")
            else:
                logger.warning(f"Failed to calculate summer integrated NDVI: {e}")
                # Record failure for circuit breaker (actual GEE error)
                self.circuit_breaker.record_failure(e)
            return None


class DataContextBuilder:
    """Builds comprehensive context for heuristic calculations"""
    
    def __init__(
        self, 
        data_dir: Path, 
        cache_dir: Optional[Path] = None,
        use_gee_ndvi: bool = False,
        gee_config: Optional[Dict] = None
    ):
        """
        Initialize DataContextBuilder.
        
        Args:
            data_dir: Path to data directory
            cache_dir: Optional cache directory (defaults to data_dir/cache)
            use_gee_ndvi: If True, use Google Earth Engine for NDVI (default: False)
            gee_config: Optional GEE configuration dict with keys:
                - project: GEE project ID (default: 'ee-jongalentine')
                - collection: 'landsat8' or 'sentinel2' (default: 'landsat8')
                - batch_size: Points per batch (default: 100)
                - max_workers: Thread pool size (default: 4)
                - use_cache: Enable NDVI caching (default: True)
                - cache_dir: Override cache directory (defaults to data_dir/cache)
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir or (self.data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load static layers
        self._load_static_layers()
        
        # Initialize data loaders
        self.snotel_client = AWDBClient(self.data_dir)
        self.weather_client = WeatherClient(data_dir=self.data_dir, use_real_data=True)
        
        # NDVI data source: GEE (if enabled) or SatelliteClient (AppEEARS/placeholder)
        self.use_gee_ndvi = use_gee_ndvi
        self.ndvi_client = None
        
        if use_gee_ndvi:
            if not EE_AVAILABLE:
                logger.warning("GEE requested but earthengine-api not available. Falling back to SatelliteClient.")
                self.use_gee_ndvi = False
                self.satellite_client = SatelliteClient(use_real_data=False)
            else:
                # Initialize GEE NDVI client
                gee_config = gee_config or {}
                # Use cache_dir from gee_config if provided, otherwise use DataContextBuilder's cache_dir
                ndvi_cache_dir = gee_config.get('cache_dir') or self.cache_dir
                use_cache = gee_config.get('use_cache', True)  # Default to True for historical data
                # Store buffer_days from config (0 = adaptive, >0 = fixed value)
                self.ndvi_buffer_days = gee_config.get('buffer_days', 0)  # 0 triggers adaptive mode
                # Store max_cloud_cover from config (default: 50.0%)
                self.ndvi_max_cloud_cover = gee_config.get('max_cloud_cover', 50.0)  # Use config value
                try:
                    self.ndvi_client = GEENDVIClient(
                        project=gee_config.get('project', 'ee-jongalentine'),
                        collection=gee_config.get('collection', 'landsat8'),
                        batch_size=gee_config.get('batch_size', 100),
                        max_workers=gee_config.get('max_workers', 4),
                        cache_dir=ndvi_cache_dir,
                        use_cache=use_cache
                    )
                    logger.info("GEE NDVI client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize GEE NDVI client: {e}. Falling back to SatelliteClient.")
                    self.use_gee_ndvi = False
                    self.satellite_client = SatelliteClient(use_real_data=False)
                else:
                    # GEE initialized successfully, still keep SatelliteClient for fallback
                    self.satellite_client = SatelliteClient(use_real_data=False)
        else:
            # TODO: NDVI data currently uses google earth engine. For production, implement pre-downloaded
            # raster files (similar to DEM/landcover) for training, and consider cloud-based
            # solutions for inference. AppEEARS async API is not suitable for inference (requires
            # minutes to process requests). See docs/dataset_gap_analysis.md for details.
            self.satellite_client = SatelliteClient(use_real_data=False)
    
    def _load_static_layers(self):
        """Load data that doesn't change over time"""
        
        if not RASTERIO_AVAILABLE:
            print("Loading static data layers...")
            print("  ⚠ rasterio not available, skipping raster data loading")
            self.dem = None
            self.slope = None
            self.aspect = None
            self.landcover = None
            self.canopy = None
            # Still load vector data which doesn't require rasterio
            import geopandas as gpd
            # Water sources
            water_path = self.data_dir / "hydrology" / "water_sources.geojson"
            if water_path.exists():
                self.water_sources = gpd.read_file(water_path)
                print(f"  ✓ Water sources loaded: {len(self.water_sources)} features")
            else:
                self.water_sources = None
            # Roads
            roads_path = self.data_dir / "infrastructure" / "roads.geojson"
            if roads_path.exists():
                self.roads = gpd.read_file(roads_path)
                print(f"  ✓ Roads loaded: {len(self.roads)} features")
            else:
                self.roads = None
            # Hunt areas
            hunt_areas_path = self.data_dir / "hunt_areas" / "hunt_areas.geojson"
            if hunt_areas_path.exists():
                self.hunt_areas = gpd.read_file(hunt_areas_path)
                print(f"  ✓ Hunt areas loaded: {len(self.hunt_areas)} features")
            else:
                self.hunt_areas = None
            # Trails
            trails_path = self.data_dir / "infrastructure" / "trails.geojson"
            if trails_path.exists():
                self.trails = gpd.read_file(trails_path)
                print(f"  ✓ Trails loaded: {len(self.trails)} features")
            else:
                self.trails = None
            # Wolf pack territories
            wolf_path = self.data_dir / "wildlife" / "wolf_packs.geojson"
            if wolf_path.exists():
                self.wolf_packs = gpd.read_file(wolf_path)
                print(f"  ✓ Wolf territories loaded: {len(self.wolf_packs)} packs")
            else:
                self.wolf_packs = None
            # Bear activity
            bear_path = self.data_dir / "wildlife" / "bear_activity.geojson"
            if bear_path.exists():
                self.bear_activity = gpd.read_file(bear_path)
                print(f"  ✓ Bear activity loaded: {len(self.bear_activity)} features")
            else:
                self.bear_activity = None
            return
        
        print("Loading static data layers...")
        
        # Digital Elevation Model
        dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
        if dem_path.exists():
            self.dem = rasterio.open(dem_path)
            print(f"  ✓ DEM loaded: {dem_path}")
        else:
            print(f"  ✗ DEM not found: {dem_path}")
            self.dem = None
        
        # Slope (derived from DEM)
        slope_path = self.data_dir / "terrain" / "slope.tif"
        if slope_path.exists():
            self.slope = rasterio.open(slope_path)
            print(f"  ✓ Slope loaded")
        else:
            self.slope = None
        
        # Aspect
        aspect_path = self.data_dir / "terrain" / "aspect.tif"
        if aspect_path.exists():
            self.aspect = rasterio.open(aspect_path)
            print(f"  ✓ Aspect loaded")
        else:
            self.aspect = None
        
        # Land cover
        landcover_path = self.data_dir / "landcover" / "nlcd.tif"
        if landcover_path.exists():
            self.landcover = rasterio.open(landcover_path)
            print(f"  ✓ Land cover loaded")
        else:
            self.landcover = None
        
        # Canopy cover
        canopy_path = self.data_dir / "canopy" / "canopy_cover.tif"
        if canopy_path.exists():
            self.canopy = rasterio.open(canopy_path)
            print(f"  ✓ Canopy cover loaded")
        else:
            self.canopy = None
        
        # Water sources (vector data)
        water_path = self.data_dir / "hydrology" / "water_sources.geojson"
        if water_path.exists():
            import geopandas as gpd
            self.water_sources = gpd.read_file(water_path)
            # Cache projected version for efficient distance calculations
            if self.water_sources.crs is None:
                self.water_sources.set_crs('EPSG:4326', inplace=True)
            # Convert to UTM Zone 12N (covers most of Wyoming) for accurate distance calculations
            self.water_sources_proj = self.water_sources.to_crs('EPSG:32612')
            print(f"  ✓ Water sources loaded: {len(self.water_sources)} features")
        else:
            print(f"  ✗ Water sources not found")
            self.water_sources = None
            self.water_sources_proj = None
        
        # Roads (vector data)
        roads_path = self.data_dir / "infrastructure" / "roads.geojson"
        if roads_path.exists():
            import geopandas as gpd
            self.roads = gpd.read_file(roads_path)
            print(f"  ✓ Roads loaded: {len(self.roads)} features")
        else:
            self.roads = None
        
        # Trails
        trails_path = self.data_dir / "infrastructure" / "trails.geojson"
        if trails_path.exists():
            import geopandas as gpd
            self.trails = gpd.read_file(trails_path)
            print(f"  ✓ Trails loaded: {len(self.trails)} features")
        else:
            self.trails = None
        
        # Wolf pack territories
        wolf_path = self.data_dir / "wildlife" / "wolf_packs.geojson"
        if wolf_path.exists():
            import geopandas as gpd
            self.wolf_packs = gpd.read_file(wolf_path)
            print(f"  ✓ Wolf territories loaded: {len(self.wolf_packs)} packs")
        else:
            self.wolf_packs = None
        
        # Bear activity centers
        bear_path = self.data_dir / "wildlife" / "bear_activity.geojson"
        if bear_path.exists():
            import geopandas as gpd
            self.bear_activity = gpd.read_file(bear_path)
            print(f"  ✓ Bear activity loaded: {len(self.bear_activity)} locations")
        else:
            self.bear_activity = None
        
        print("Static data loading complete.\n")
    
    def build_context(self, location: Optional[Dict] = None, date: Optional[str] = None,
                     lat: Optional[float] = None, lon: Optional[float] = None,
                     dt: Optional[datetime] = None,
                     buffer_km: float = 1.0) -> Dict:
        """
        Build complete context for a location and date
        
        Args:
            location: {"lat": float, "lon": float} (deprecated, use lat/lon instead)
            date: ISO format date string (deprecated, use dt instead)
            lat: Latitude (preferred)
            lon: Longitude (preferred)
            dt: datetime object (preferred over date string)
            buffer_km: Radius for neighborhood analysis
        
        Returns:
            Dictionary with all context data
        """
        # Support both old (location dict) and new (lat/lon) calling conventions
        if location is not None:
            # Old calling convention: location dict
            lat, lon = location["lat"], location["lon"]
            if date is not None:
                # Handle both string and datetime objects
                if isinstance(date, datetime):
                    dt = date
                else:
                    dt = pd.to_datetime(date)
        elif lat is not None and lon is not None:
            # New calling convention: separate lat/lon
            if dt is None:
                if date is not None:
                    # Handle both string and datetime objects
                    if isinstance(date, datetime):
                        dt = date
                    else:
                        dt = pd.to_datetime(date)
                else:
                    raise ValueError("Either 'dt' or 'date' must be provided")
        else:
            raise ValueError("Either 'location' dict or 'lat'/'lon' must be provided")
        point = Point(lon, lat)
        
        context = {}
        
        # --- STATIC TERRAIN DATA ---
        # Check if location is within Wyoming bounds before sampling
        # Wyoming boundaries: approximately 41-45°N, 104-111°W
        wyoming_bounds = {
            'north': 45.0,
            'south': 41.0,
            'east': -104.0,
            'west': -111.0
        }
        
        outside_bounds = (
            lat > wyoming_bounds['north'] or
            lat < wyoming_bounds['south'] or
            lon < wyoming_bounds['west'] or
            lon > wyoming_bounds['east']
        )
        
        if outside_bounds:
            # Location is outside Wyoming - use NaN to indicate "outside bounds" 
            # This is different from placeholder 8500.0, so we can distinguish
            # between "not processed yet" vs "outside bounds"
            context["elevation"] = np.nan
            logger.debug(f"Location ({lat:.6f}, {lon:.6f}) is outside Wyoming bounds - setting elevation to NaN")
        else:
            # Sample elevation from DEM
            elevation = self._sample_raster(self.dem, lon, lat, default=8500.0)
            context["elevation"] = elevation
            
            # Warn if using placeholder elevation (indicates DEM sampling failed)
            if elevation == 8500.0:
                # Check if near boundaries (within 0.1 degrees)
                tolerance = 0.1
                boundary_issues = []
                if lat > wyoming_bounds['north'] - tolerance:
                    boundary_issues.append(f"near northern boundary ({wyoming_bounds['north']}°N)")
                if lat < wyoming_bounds['south'] + tolerance:
                    boundary_issues.append(f"near southern boundary ({wyoming_bounds['south']}°N)")
                if lon < wyoming_bounds['west'] + tolerance:
                    boundary_issues.append(f"near western boundary ({wyoming_bounds['west']}°W)")
                if lon > wyoming_bounds['east'] - tolerance:
                    boundary_issues.append(f"near eastern boundary ({wyoming_bounds['east']}°W)")
                
                # Build warning message
                warning_msg = f"Using default elevation (8500 ft) for location ({lat:.6f}, {lon:.6f})"
                if boundary_issues:
                    warning_msg += f" - Location is {', '.join(boundary_issues)}"
                else:
                    warning_msg += " - DEM sampling failed (may be outside DEM bounds or DEM file issue)"
                
                logger.warning(warning_msg)
        
        if outside_bounds:
            # Location is outside Wyoming - set all terrain features to NaN
            context["slope_degrees"] = np.nan
            context["aspect_degrees"] = np.nan
            context["canopy_cover_percent"] = np.nan
            context["land_cover_code"] = np.nan
            context["land_cover_type"] = "outside_bounds"
        else:
            # Sample terrain features from rasters
            context["slope_degrees"] = self._sample_raster(self.slope, lon, lat, default=15.0)
            context["aspect_degrees"] = self._sample_raster(self.aspect, lon, lat, default=180.0)
            # Sample canopy cover and clamp to valid range (0-100%)
            canopy_value = self._sample_raster(
                self.canopy, lon, lat, default=30.0
            )
            # Clamp to valid percentage range (0-100%)
            context["canopy_cover_percent"] = max(0.0, min(100.0, canopy_value))
            
            # Land cover type
            landcover_code = self._sample_raster(self.landcover, lon, lat, default=0)
            context["land_cover_type"] = self._decode_landcover(landcover_code)
            context["land_cover_code"] = landcover_code
        
        # --- WATER DATA ---
        if self.water_sources is not None:
            water_dist, water_reliability = self._calculate_water_metrics(point)
            context["water_distance_miles"] = water_dist
            context["water_reliability"] = water_reliability
        else:
            context["water_distance_miles"] = 0.5  # Default
            context["water_reliability"] = 0.8
        
        # --- ROAD/TRAIL ACCESS ---
        if self.roads is not None:
            context["road_distance_miles"] = self._calculate_distance_to_nearest(
                point, self.roads
            )
        else:
            context["road_distance_miles"] = 2.0
        
        if self.trails is not None:
            context["trail_distance_miles"] = self._calculate_distance_to_nearest(
                point, self.trails
            )
        else:
            context["trail_distance_miles"] = 1.5
        
        # --- SECURITY HABITAT ---
        context["security_habitat_percent"] = self._calculate_security_habitat(
            point, buffer_km
        )
        
        # --- PREDATOR DATA ---
        if self.wolf_packs is not None:
            context["wolves_per_1000_elk"] = self._calculate_wolf_density(point)
            context["wolf_data_quality"] = 0.75
        else:
            context["wolves_per_1000_elk"] = 2.0  # Low default
            context["wolf_data_quality"] = 0.5
        
        if self.bear_activity is not None:
            context["bear_activity_distance_miles"] = \
                self._calculate_distance_to_nearest(point, self.bear_activity)
            context["bear_data_quality"] = 0.65
        else:
            context["bear_activity_distance_miles"] = 5.0
            context["bear_data_quality"] = 0.5
        
        # --- TEMPORAL DATA (depends on date) ---
        # dt is already set from the calling convention handling above
        
        # Snow data - pass elevation for better estimation if no station nearby
        location_elevation = context.get("elevation", None)
        snow_data = self.snotel_client.get_snow_data(lat, lon, dt, elevation_ft=location_elevation)
        context["snow_depth_inches"] = snow_data.get("depth", 0.0)
        context["snow_water_equiv_inches"] = snow_data.get("swe", 0.0)
        context["snow_crust_detected"] = snow_data.get("crust", False)
        
        # Data quality tracking: indicate whether data is real SNOTEL or estimate
        snow_station = snow_data.get("station")
        context["snow_data_source"] = "snotel" if snow_station else "estimate"
        context["snow_station_name"] = snow_station
        context["snow_station_distance_km"] = snow_data.get("station_distance_km", None)
        
        # Weather data
        weather = self.weather_client.get_weather(lat, lon, dt)
        context["temperature_f"] = weather.get("temp", 45.0)
        context["precip_last_7_days_inches"] = weather.get("precip_7d", 0.0)
        context["cloud_cover_percent"] = weather.get("cloud_cover", 20)

        # Lunar illumination features (for nocturnal activity modeling)
        # Research shows ungulates respond to moonlight conditions
        try:
            from .lunar_client import LunarCalculator
            lunar_calc = LunarCalculator()
            cloud_cover = context["cloud_cover_percent"]

            # Get comprehensive lunar features
            lunar_features = lunar_calc.get_all_lunar_features(lat, lon, dt, cloud_cover)

            context["moon_phase"] = lunar_features.get("moon_phase", 0.5)
            context["moon_altitude_midnight"] = lunar_features.get("moon_altitude_midnight", 0.0)
            context["effective_illumination"] = lunar_features.get("effective_illumination_midnight", 0.0)
            context["cloud_adjusted_illumination"] = lunar_features.get("cloud_adjusted_illumination", 0.0)
        except ImportError:
            # Fallback if lunar client not available
            logger.debug("Lunar client not available, using placeholder values")
            context["moon_phase"] = 0.5
            context["moon_altitude_midnight"] = 0.0
            context["effective_illumination"] = 0.0
            context["cloud_adjusted_illumination"] = 0.0
        except Exception as e:
            logger.debug(f"Error calculating lunar features: {e}")
            context["moon_phase"] = 0.5
            context["moon_altitude_midnight"] = 0.0
            context["effective_illumination"] = 0.0
            context["cloud_adjusted_illumination"] = 0.0

        # Vegetation data
        if self.use_gee_ndvi and self.ndvi_client:
            # Use GEE for NDVI (even for single points - wrap in DataFrame)
            try:
                # Create single-row DataFrame for GEE batch processing
                point_df = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lon],
                    'timestamp': [dt]
                })
                
                # Get NDVI from GEE (use config buffer_days and max_cloud_cover)
                ndvi_df = self.ndvi_client.get_ndvi_for_points(
                    point_df,
                    date_column='timestamp',
                    lat_column='latitude',
                    lon_column='longitude',
                    buffer_days=getattr(self, 'ndvi_buffer_days', 0),  # 0 = adaptive (21 days L5, 14 days L8)
                    max_cloud_cover=getattr(self, 'ndvi_max_cloud_cover', 50.0)  # Use config value (default: 50%)
                )
                
                # Extract NDVI value, cloud cover, age, and IRG
                ndvi_value = ndvi_df.iloc[0]['ndvi'] if len(ndvi_df) > 0 else None
                cloud_cover_value = ndvi_df.iloc[0].get('cloud_cover_percent') if len(ndvi_df) > 0 else None
                ndvi_age_days_value = ndvi_df.iloc[0].get('ndvi_age_days') if len(ndvi_df) > 0 else None
                irg_value = ndvi_df.iloc[0].get('irg') if len(ndvi_df) > 0 else None
                
                if pd.notna(ndvi_value) and ndvi_value is not None:
                    context["ndvi"] = float(ndvi_value)
                    
                    # Use real ndvi_age_days from GEE if available
                    if pd.notna(ndvi_age_days_value) and ndvi_age_days_value is not None:
                        context["ndvi_age_days"] = float(ndvi_age_days_value)
                    else:
                        context["ndvi_age_days"] = 8  # Fallback default
                    
                    # Use real IRG from GEE if available
                    if pd.notna(irg_value) and irg_value is not None:
                        context["irg"] = float(irg_value)
                    else:
                        # Fallback to seasonal approximation
                        month = dt.month
                        context["irg"] = 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0
                    
                    # Use cloud cover from GEE if available
                    if pd.notna(cloud_cover_value) and cloud_cover_value is not None:
                        context["cloud_cover_percent"] = float(cloud_cover_value)
                        logger.debug(f"Using cloud cover from GEE: {cloud_cover_value}%")
                else:
                    # GEE returned None, fall back to SatelliteClient
                    logger.debug(f"GEE returned None for NDVI at ({lat}, {lon}), using SatelliteClient fallback")
                    ndvi_data = self.satellite_client.get_ndvi(lat, lon, dt)
                    context["ndvi"] = ndvi_data.get("ndvi", 0.5)
                    context["ndvi_age_days"] = ndvi_data.get("age_days", 8)
                    # Try to calculate IRG even with fallback
                    month = dt.month
                    context["irg"] = ndvi_data.get("irg", 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0)
            except Exception as e:
                # GEE failed, fall back to SatelliteClient
                logger.warning(f"GEE NDVI retrieval failed: {e}, using SatelliteClient fallback")
                ndvi_data = self.satellite_client.get_ndvi(lat, lon, dt)
                context["ndvi"] = ndvi_data.get("ndvi", 0.5)
                context["ndvi_age_days"] = ndvi_data.get("age_days", 8)
                # Try to calculate IRG even with fallback
                month = dt.month
                context["irg"] = ndvi_data.get("irg", 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0)
        else:
            ndvi_data = self.satellite_client.get_ndvi(lat, lon, dt)
            context["ndvi"] = ndvi_data.get("ndvi", 0.5)
            context["ndvi_age_days"] = ndvi_data.get("age_days", 8)
            # Try to calculate IRG even with fallback
            month = dt.month
            context["irg"] = ndvi_data.get("irg", 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0)
        
        # Summer integrated NDVI (for nutritional condition)
        # Use current year if after September, otherwise use previous year
        summer_year = dt.year if dt.month >= 9 else dt.year - 1
        
        if self.use_gee_ndvi and self.ndvi_client:
            # Use GEE for summer integrated NDVI
            try:
                summer_ndvi = self.ndvi_client.get_summer_integrated_ndvi(
                    lat=lat,
                    lon=lon,
                    year=summer_year,
                    buffer_days=getattr(self, 'ndvi_buffer_days', 0),  # 0 = adaptive (21 days L5, 14 days L8)
                    max_cloud_cover=getattr(self, 'ndvi_max_cloud_cover', 50.0)  # Use config value (default: 50%)
                )
                if summer_ndvi is not None:
                    context["summer_integrated_ndvi"] = summer_ndvi
                else:
                    # Fallback: For historical data (before 2015), AppEEARS is not practical (async API, slow)
                    # For recent data, AppEEARS might work but is still async and slow for real-time processing
                    # Use placeholder 60.0 for all fallback cases to avoid pipeline slowdowns
                    if summer_year < 2015:
                        # Historical data: Skip AppEEARS (too slow, async API not suitable)
                        context["summer_integrated_ndvi"] = 60.0
                        logger.debug(f"GEE returned None for summer integrated NDVI for {summer_year} at ({lat:.4f}, {lon:.4f}), "
                                   f"using placeholder 60.0 (historical data, AppEEARS not suitable for async processing)")
                    else:
                        # Recent data: Try AppEEARS but with short timeout to avoid blocking
                        logger.debug(f"GEE returned None for summer integrated NDVI for {summer_year} at ({lat:.4f}, {lon:.4f}), "
                                   f"trying AppEEARS fallback (may timeout and return placeholder)")
                        summer_start = datetime(summer_year, 6, 1)
                        summer_end = datetime(summer_year, 9, 1)
                        fallback_ndvi = self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
                        context["summer_integrated_ndvi"] = fallback_ndvi
                        if fallback_ndvi == 60.0:
                            logger.debug(f"Summer integrated NDVI fallback returned placeholder 60.0 for {summer_year} "
                                       f"at ({lat:.4f}, {lon:.4f}) (AppEEARS unavailable or timed out)")
            except Exception as e:
                # GEE failed, fall back to placeholder (AppEEARS too slow for real-time processing)
                if summer_year < 2015:
                    context["summer_integrated_ndvi"] = 60.0
                    logger.debug(f"GEE summer integrated NDVI failed for {summer_year} at ({lat:.4f}, {lon:.4f}): {e}, "
                               f"using placeholder 60.0 (historical data)")
                else:
                    logger.debug(f"GEE summer integrated NDVI failed for {summer_year} at ({lat:.4f}, {lon:.4f}): {e}, "
                               f"trying AppEEARS fallback (may timeout)")
                    summer_start = datetime(summer_year, 6, 1)
                    summer_end = datetime(summer_year, 9, 1)
                    fallback_ndvi = self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
                    context["summer_integrated_ndvi"] = fallback_ndvi
                    if fallback_ndvi == 60.0:
                        logger.debug(f"Summer integrated NDVI fallback returned placeholder 60.0 for {summer_year} "
                                   f"at ({lat:.4f}, {lon:.4f}) (AppEEARS unavailable or timed out)")
        else:
            # Use SatelliteClient
            summer_start = datetime(summer_year, 6, 1)
            summer_end = datetime(summer_year, 9, 1)
            context["summer_integrated_ndvi"] = \
                self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
        
        # Population/herd data (from state agencies)
        context["pregnancy_rate"] = 0.90  # Typical
        
        # Grid references for spatial operations
        context["dem_grid"] = self.dem
        context["water_sources"] = self.water_sources
        
        return context
    
    def _sample_raster(self, raster, lon: float, lat: float, 
                      default: float = 0.0) -> float:
        """Sample value from raster at point"""
        if raster is None or not RASTERIO_AVAILABLE:
            return default
        
        try:
            # Handle CRS transformation if needed
            # If raster is not in WGS84, transform coordinates
            if raster.crs and not raster.crs.is_geographic:
                # Raster is in projected CRS, need to transform lat/lon
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
                # Ensure scalar values to avoid deprecation warning
                x, y = transformer.transform(float(lon), float(lat))
                row, col = raster.index(x, y)
            else:
                # Raster is in geographic CRS (WGS84), use directly
                row, col = raster.index(lon, lat)
            
            # Read value
            window = rasterio.windows.Window(col, row, 1, 1)
            data = raster.read(1, window=window)
            
            value = float(data[0, 0])
            
            # Check for nodata
            if raster.nodata is not None and (value == raster.nodata or np.isnan(value)):
                return default
            
            return value
        except Exception as e:
            logger.debug(f"Error sampling raster at ({lon}, {lat}): {e}")
            return default
    
    def _calculate_water_metrics(self, point: Point) -> Tuple[float, float]:
        """Calculate distance to water and reliability using spatial index for efficiency"""
        import geopandas as gpd
        import pandas as pd
        
        # Ensure water_sources has a CRS (should be EPSG:4326)
        if self.water_sources.crs is None:
            self.water_sources.set_crs('EPSG:4326', inplace=True)
        
        # Use cached projected version for efficient distance calculations
        utm_crs = 'EPSG:32612'
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
        point_gdf_proj = point_gdf.to_crs(utm_crs)
        point_proj = point_gdf_proj.geometry.iloc[0]
        
        try:
            # Use sjoin_nearest in projected CRS (no warnings, accurate distances)
            nearest = gpd.sjoin_nearest(
                point_gdf_proj,
                self.water_sources_proj,
                how='left',
                max_distance=100000,  # 100 km max in meters
                distance_col='distance_m'
            )
            
            if len(nearest) == 0 or pd.isna(nearest.iloc[0].get('index_right')):
                # No water source found within max_distance
                return 10.0, 0.5  # Default: 10 miles, medium reliability
            
            # Get the nearest water source (use original index from projected dataframe)
            nearest_idx = int(nearest.iloc[0]['index_right'])
            nearest_feature = self.water_sources.iloc[nearest_idx]
            
            # Distance is already in meters from sjoin_nearest
            distance_m = nearest.iloc[0].get('distance_m', point_proj.distance(self.water_sources_proj.iloc[nearest_idx].geometry))
            distance_mi = distance_m / 1609.34
            
        except (AttributeError, TypeError, KeyError):
            # Fallback: use spatial index directly
            if not hasattr(self.water_sources_proj, 'sindex') or self.water_sources_proj.sindex is None:
                self.water_sources_proj.sindex  # Build spatial index if needed
            
            # Find nearest using spatial index
            possible_matches_index = list(self.water_sources_proj.sindex.nearest(
                point_proj.bounds, num_results=1
            ))
            
            if len(possible_matches_index) == 0:
                return 10.0, 0.5  # Default
            
            # Get the nearest feature
            nearest_idx = possible_matches_index[0]
            nearest_feature = self.water_sources.iloc[nearest_idx]
            nearest_geom_proj = self.water_sources_proj.iloc[nearest_idx].geometry
            
            # Calculate distance in projected CRS (meters)
            distance_m = point_proj.distance(nearest_geom_proj)
            distance_mi = distance_m / 1609.34
        
        # Get water source attributes
        water_type = nearest_feature.get("water_type", nearest_feature.get("type", "stream"))
        if pd.notna(water_type):
            water_type = str(water_type).lower()
        else:
            water_type = "stream"
        
        reliability_map = {
            "spring": 1.0,
            "lake": 1.0,
            "pond": 0.9,
            "stream": 0.7,
            "creek": 0.7,
            "ephemeral": 0.4
        }
        reliability = reliability_map.get(water_type, 0.7)
        
        return distance_mi, reliability
    
    def _calculate_distance_to_nearest(self, point: Point, features) -> float:
        """Calculate distance to nearest feature in miles using spatial index"""
        import geopandas as gpd
        import pandas as pd
        
        # Ensure features have a CRS
        if features.crs is None:
            features.set_crs('EPSG:4326', inplace=True)
        
        # Convert to projected CRS (UTM Zone 12N for Wyoming) for accurate distance calculations
        utm_crs = 'EPSG:32612'
        features_proj = features.to_crs(utm_crs)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
        point_gdf_proj = point_gdf.to_crs(utm_crs)
        point_proj = point_gdf_proj.geometry.iloc[0]
        
        try:
            # Use sjoin_nearest in projected CRS (no warnings, accurate distances)
            nearest = gpd.sjoin_nearest(
                point_gdf_proj,
                features_proj,
                how='left',
                max_distance=100000,  # 100 km max in meters
                distance_col='distance_m'
            )
            
            if len(nearest) == 0 or pd.isna(nearest.iloc[0].get('index_right')):
                # No feature found within max_distance
                return 10.0  # Default: 10 miles
            
            # Get the nearest feature
            nearest_idx = int(nearest.iloc[0]['index_right'])
            
            # Distance is already in meters from sjoin_nearest
            distance_m = nearest.iloc[0].get('distance_m', point_proj.distance(features_proj.iloc[nearest_idx].geometry))
            distance_mi = distance_m / 1609.34
            
        except (AttributeError, TypeError, KeyError):
            # Fallback: use spatial index directly
            if not hasattr(features_proj, 'sindex') or features_proj.sindex is None:
                features_proj.sindex  # Build spatial index if needed
            
            # Find nearest using spatial index
            possible_matches_index = list(features_proj.sindex.nearest(
                point_proj.bounds, num_results=1
            ))
            
            if len(possible_matches_index) == 0:
                return 10.0  # Default
            
            # Get the nearest feature
            nearest_idx = possible_matches_index[0]
            nearest_geom_proj = features_proj.iloc[nearest_idx].geometry
            
            # Calculate distance in projected CRS (meters)
            distance_m = point_proj.distance(nearest_geom_proj)
            distance_mi = distance_m / 1609.34
        
        return distance_mi
    
    def _calculate_security_habitat(self, point: Point, buffer_km: float) -> float:
        """Calculate % of security habitat in buffer around point"""
        
        # Create buffer (in degrees, approximate)
        buffer_deg = buffer_km / 111.0  # 1 degree ≈ 111 km
        buffered = point.buffer(buffer_deg)
        
        # Sample terrain within buffer
        # Security criteria: slope > 40° OR canopy > 70% OR remote
        
        if self.slope is None:
            return 35.0  # Default moderate security
        
        if not RASTERIO_AVAILABLE or self.slope is None:
            return 35.0  # Default moderate security
        
        try:
            # Extract slope within buffer
            from rasterio.mask import mask as rio_mask
            
            geom = [buffered.__geo_interface__]
            # Suppress NotGeoreferencedWarning for this operation (non-critical)
            with warnings.catch_warnings():
                if RASTERIO_AVAILABLE:
                    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
                slope_data, _ = rio_mask(self.slope, geom, crop=True)
            
            # Calculate % meeting security criteria
            steep_pixels = np.sum(slope_data > 40)
            total_pixels = slope_data.size
            
            security_pct = (steep_pixels / total_pixels) * 100
            
            # TODO: Also check canopy cover and remoteness
            # For now, just use slope
            
            return float(security_pct)
        except Exception as e:
            return 35.0
    
    def _calculate_wolf_density(self, point: Point) -> float:
        """Calculate wolves per 1000 elk in area"""
        
        # Check if point is within any wolf pack territory
        for idx, pack in self.wolf_packs.iterrows():
            if pack.geometry.contains(point):
                pack_size = pack.get("pack_size", 6)
                territory_area_sqmi = pack.geometry.area * 111.0**2  # deg² to km² to mi²
                elk_in_territory = pack.get("elk_count", 2000)
                
                wolves_per_1000 = (pack_size / elk_in_territory) * 1000
                
                return wolves_per_1000
        
        # Not in any pack territory
        return 0.5  # Background level
    
    def _decode_landcover(self, code) -> str:
        """Decode NLCD land cover code to description"""
        import pandas as pd
        
        # Handle NaN/None values
        if pd.isna(code) or code is None:
            return "outside_bounds"
        
        landcover_map = {
            41: "deciduous_forest",
            42: "evergreen_forest",
            43: "mixed_forest",
            52: "shrub",
            71: "grassland",
            81: "pasture",
            90: "wetland",
            95: "emergent_wetland",
            # ... add more as needed
        }
        return landcover_map.get(int(code), "unknown")
    
    def add_ndvi(
        self,
        points: pd.DataFrame,
        date_column: str = 'timestamp',
        lat_column: str = 'latitude',
        lon_column: str = 'longitude',
        buffer_days: int = 7,
        max_cloud_cover: float = 30.0
    ) -> pd.DataFrame:
        """
        Add NDVI data to points using Google Earth Engine (if enabled) or SatelliteClient.
        
        Args:
            points: DataFrame with GPS observations
            date_column: Column name for date/timestamp (default: 'timestamp')
            lat_column: Column name for latitude (default: 'latitude')
            lon_column: Column name for longitude (default: 'longitude')
            buffer_days: Days before/after target date to search for images (default: 7)
            max_cloud_cover: Maximum cloud cover percentage 0-100 (default: 30.0)
            
        Returns:
            DataFrame with 'ndvi' column added
        """
        if self.use_gee_ndvi and self.ndvi_client:
            # Use GEE for batch NDVI retrieval
            return self.ndvi_client.get_ndvi_for_points(
                points,
                date_column=date_column,
                lat_column=lat_column,
                lon_column=lon_column,
                buffer_days=buffer_days,
                max_cloud_cover=max_cloud_cover
            )
        else:
            # Fall back to SatelliteClient (AppEEARS or placeholder)
            logger.info("Using SatelliteClient for NDVI (GEE not enabled)")
            # Convert DataFrame to list of tuples for batch processing
            result_df = points.copy()
            ndvi_values = []
            
            for idx, row in points.iterrows():
                date = pd.to_datetime(row[date_column])
                lat = row[lat_column]
                lon = row[lon_column]
                
                ndvi_data = self.satellite_client.get_ndvi(lat, lon, date)
                ndvi_values.append(ndvi_data.get("ndvi", 0.5))
            
            result_df['ndvi'] = ndvi_values
            return result_df


class SNOTELDataCache:
    """
    Persistent cache for SNOTEL station data retrieved from AWDB API.
    
    Uses SQLite for efficient storage and retrieval. Cache keys are based on:
    - Station ID
    - Date range (begin_date, end_date)
    
    Since historical data never changes, cached entries are permanent.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize SNOTEL data cache.
        
        Args:
            cache_dir: Directory for cache database. Defaults to data/cache/snotel_cache.db
        """
        if cache_dir is None:
            cache_dir = Path('data/cache')
        else:
            cache_dir = Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = cache_dir / 'snotel_data_cache.db'
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        logger.info(f"SNOTEL data cache initialized: {self.cache_db}")
    
    def _init_db(self):
        """Initialize cache database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                # Create table for station data
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
                # Create index separately (SQLite doesn't support inline INDEX in CREATE TABLE)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_station_dates 
                    ON snotel_station_data(station_id, begin_date, end_date)
                """)
                conn.commit()
            finally:
                conn.close()
    
    def _make_cache_key(self, station_id: str, begin_date: str, end_date: str) -> str:
        """Generate cache key from parameters."""
        key_data = f"{station_id}:{begin_date}:{end_date}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, station_id: str, begin_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieve station data from cache.
        
        Args:
            station_id: AWDB station ID
            begin_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with station data, or None if not in cache
        """
        cache_key = self._make_cache_key(station_id, begin_date, end_date)
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("""
                    SELECT data_json
                    FROM snotel_station_data
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                # Deserialize JSON to DataFrame
                data_json = row[0]
                import json
                data_dict = json.loads(data_json)
                df = pd.DataFrame(data_dict)
                df['date'] = pd.to_datetime(df['date'])
                return df
            finally:
                conn.close()
    
    def put(self, station_id: str, begin_date: str, end_date: str, data: pd.DataFrame):
        """
        Store station data in cache.
        
        Args:
            station_id: AWDB station ID
            begin_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: DataFrame with station data
        """
        cache_key = self._make_cache_key(station_id, begin_date, end_date)
        
        # Serialize DataFrame to JSON
        import json
        # Convert date column to string for JSON serialization
        data_copy = data.copy()
        if 'date' in data_copy.columns:
            data_copy['date'] = data_copy['date'].dt.strftime('%Y-%m-%d')
        data_json = json.dumps(data_copy.to_dict('records'))
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO snotel_station_data
                    (cache_key, station_id, begin_date, end_date, data_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (cache_key, station_id, begin_date, end_date, data_json))
                conn.commit()
            finally:
                conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM snotel_station_data")
                total_entries = cursor.fetchone()[0]
                
                db_size = self.cache_db.stat().st_size if self.cache_db.exists() else 0
                total_size_mb = db_size / (1024 * 1024)
                
                return {
                    'total_entries': total_entries,
                    'total_size_mb': round(total_size_mb, 2)
                }
            finally:
                conn.close()


class AWDBClient:
    """Client for accessing SNOTEL snow data using AWDB REST API"""
    
    # AWDB REST API base URL
    AWDB_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
    
    # Class-level sets to track warnings across all instances (shared by all workers)
    # This prevents duplicate warnings when multiple workers process the same stations
    _warned_stations = set()  # Set of (station_id, warning_type) tuples
    _warned_api_failures = set()  # Set of station_ids that have failed API calls
    _warning_lock = None  # Will be initialized on first use for thread safety
    
    def __init__(self, data_dir: Optional[Path] = None, use_cache: bool = True, cache_historical_only: bool = True):
        """
        Initialize AWDB client.
        
        Args:
            data_dir: Data directory (defaults to "data")
            use_cache: If True, use persistent cache for historical data (default: True)
            cache_historical_only: If True, only cache data older than 30 days (default: True)
                                   This allows live data for inference while caching historical training data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.station_cache_path = self.cache_dir / "snotel_stations_wyoming.geojson"
        self._stations_gdf = None
        
        self.use_cache = use_cache
        self.cache_historical_only = cache_historical_only
        
        # Initialize persistent cache if enabled
        if self.use_cache:
            self.data_cache = SNOTELDataCache(cache_dir=self.cache_dir)
            cache_stats = self.data_cache.get_stats()
            logger.info(
                f"SNOTEL data cache enabled: {cache_stats['total_entries']:,} entries "
                f"({cache_stats['total_size_mb']:.2f} MB)"
            )
        else:
            self.data_cache = None
            logger.info("SNOTEL data cache disabled")
        
        # Two-level caching:
        # 1. Station data cache: station_id -> DataFrame with date range data
        #    This avoids re-downloading the same station data for multiple locations/dates
        #    We cache by station ID and date range to minimize API calls
        # 2. Request cache: (lat, lon, date) -> final result dict
        #    This provides fast lookup for exact same location/date queries
        self.station_data_cache = {}  # In-memory cache: {station_id: DataFrame}
        self.request_cache = {}  # Cache final results: {(lat, lon, date_str): result_dict}
        
        # Cache size limits to prevent unbounded memory growth
        self.MAX_STATION_CACHE_SIZE = 100  # Keep only 100 most recent stations
        self.MAX_REQUEST_CACHE_SIZE = 10000  # Keep only 10k most recent requests
        
        # Initialize lock for thread-safe warning tracking (lazy initialization)
        if AWDBClient._warning_lock is None:
            import threading
            AWDBClient._warning_lock = threading.Lock()
        
        # Load stations (from local cache or AWDB API)
        self._load_stations()
    
    def _trim_caches(self):
        """
        Trim caches to prevent unbounded memory growth.
        Uses LRU-style eviction (removes oldest entries first).
        """
        # Trim station data cache
        if len(self.station_data_cache) > self.MAX_STATION_CACHE_SIZE:
            excess = len(self.station_data_cache) - self.MAX_STATION_CACHE_SIZE
            # Remove oldest entries (simple: remove first N keys)
            keys_to_remove = list(self.station_data_cache.keys())[:excess]
            for key in keys_to_remove:
                del self.station_data_cache[key]
            logger.debug(f"Trimmed station cache: removed {excess} entries, {len(self.station_data_cache)} remaining")
        
        # Trim request cache
        if len(self.request_cache) > self.MAX_REQUEST_CACHE_SIZE:
            excess = len(self.request_cache) - self.MAX_REQUEST_CACHE_SIZE
            # Remove oldest entries (simple: remove first N keys)
            keys_to_remove = list(self.request_cache.keys())[:excess]
            for key in keys_to_remove:
                del self.request_cache[key]
            logger.debug(f"Trimmed request cache: removed {excess} entries, {len(self.request_cache)} remaining")
    
    def _load_stations_from_awdb(self) -> bool:
        """
        Load all Wyoming SNOTEL stations from AWDB API.
        
        Returns:
            True if successful, False otherwise
        """
        import traceback
        import os
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Skip API call in test environment to avoid slow tests
            if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING'):
                logger.debug("Skipping AWDB API call in test environment")
                return False
            
            # Query AWDB API for all Wyoming SNOTEL stations
            params = {
                'stationTriplets': '*:WY:SNTL',
                'returnForecastPointMetadata': 'false',
                'returnReservoirMetadata': 'false',
                'returnStationElements': 'false',
                'activeOnly': 'true',
            }
            
            # Retry logic for connection errors
            max_retries = 3
            base_delay = 2.0
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(f"{self.AWDB_BASE_URL}/stations", params=params, timeout=30)
                    response.raise_for_status()
                    stations_data = response.json()
                    break  # Success, exit retry loop
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                        logger.warning(f"Connection error loading stations (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        # Final attempt failed
                        logger.error(f"Failed to load stations after {max_retries + 1} attempts: {e}")
                        raise
                except requests.exceptions.HTTPError as e:
                    # Non-retryable HTTP errors (4xx, etc.) - raise immediately
                    raise
            
            if not stations_data:
                logger.warning("No Wyoming SNOTEL stations found in AWDB API")
                return False
            
            # Convert to GeoDataFrame
            features = []
            for station in stations_data:
                station_id = str(station.get('stationId', ''))
                name = station.get('name', 'Unknown')
                lat = station.get('latitude')
                lon = station.get('longitude')
                elevation_ft = station.get('elevation', 0)  # AWDB returns in feet
                triplet = station.get('stationTriplet', f"{station_id}:WY:SNTL")
                
                if lat is None or lon is None:
                    logger.debug(f"Skipping station {name} (missing coordinates)")
                    continue
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "station_id": station_id,
                        "triplet": triplet,
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "elevation_ft": elevation_ft,
                        "state": "WY",
                        "awdb_station_id": station_id,  # AWDB station ID (same as snotelr_site_id)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                }
                features.append(feature)
            
            if not features:
                logger.warning("No valid Wyoming SNOTEL stations found in AWDB API response")
                return False
            
            # Create GeoDataFrame
            self._stations_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            
            # Save to cache
            self._stations_gdf.to_file(self.station_cache_path, driver="GeoJSON")
            logger.info(f"Loaded {len(self._stations_gdf)} Wyoming SNOTEL stations from AWDB API")
            return True
            
        except Exception as e:
            logger.warning(f"Error loading stations from AWDB API: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_stations(self):
        """
        Load SNOTEL station locations from AWDB REST API.
        
        Uses local cache if available. For historical data processing, the cache
        never expires (station locations don't change). For inference, we may want
        to refresh to get newly activated stations, but for training data this is
        not necessary.
        """
        # Always try to load from cache first (for historical data, cache is permanent)
        if self.station_cache_path.exists():
            try:
                import geopandas as gpd
                logger.debug(f"Loading Wyoming SNOTEL stations from cache ({self.station_cache_path.name})...")
                self._stations_gdf = gpd.read_file(self.station_cache_path)
                logger.debug(f"Loaded {len(self._stations_gdf)} Wyoming SNOTEL stations from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load stations from cache: {e}, fetching from API")
                # Fall through to API fetch
        
        # Cache doesn't exist, fetch from API (with retry for connection errors)
        logger.info("Loading Wyoming SNOTEL stations from AWDB API...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._load_stations_from_awdb():
                    return  # Success
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    delay = 2.0 * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(f"Connection error loading stations (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to load stations after {max_retries} attempts: {e}")
        
        # All retries failed
        logger.warning("Failed to load stations from AWDB API, falling back to elevation estimates")
        self._stations_gdf = None
    
    def _find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 100.0):
        """
        Find nearest SNOTEL station to location.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum distance to search (default: 100 km)
        
        Returns:
            Dictionary with station info, or None if no station within range
        """
        if self._stations_gdf is None:
            return None
        
        from shapely.geometry import Point
        import geopandas as gpd
        import pandas as pd
        
        point = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        
        # Convert to UTM for distance calculation (Wyoming is in UTM Zone 12N)
        stations_utm = self._stations_gdf.to_crs("EPSG:32612").copy()
        point_utm = point_gdf.to_crs("EPSG:32612")
        
        # Calculate distances to all stations
        stations_utm['distance_m'] = stations_utm.geometry.distance(point_utm.geometry.iloc[0])
        stations_utm['distance_km'] = stations_utm['distance_m'] / 1000.0
        
        # Filter to stations within max distance
        within_range = stations_utm[stations_utm['distance_km'] <= max_distance_km].copy()
        
        if len(within_range) == 0:
            return None
        
        # Prioritize mapped stations (those with awdb_station_id) over unmapped ones
        # Get station ID column (prefer awdb_station_id, fall back to snotelr_site_id for backward compatibility)
        if 'awdb_station_id' in self._stations_gdf.columns:
            station_id_col = 'awdb_station_id'
        elif 'snotelr_site_id' in self._stations_gdf.columns:
            station_id_col = 'snotelr_site_id'
        else:
            # No station ID column, use all stations
            station_id_col = None
        
        if station_id_col:
            # Filter to mapped stations (those with non-null station ID)
            station_ids = self._stations_gdf.iloc[within_range.index][station_id_col]
            mapped_mask = station_ids.notna()
            mapped_stations = within_range[mapped_mask]
        else:
            mapped_stations = within_range
        
        if len(mapped_stations) > 0:
            # Use nearest mapped station
            nearest_mapped = mapped_stations.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest_mapped.name]  # Use original index
            distance_km = nearest_mapped['distance_km']
        else:
            # No mapped stations available, use nearest unmapped station
            nearest = within_range.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest.name]  # Use original index
            distance_km = nearest['distance_km']
            logger.debug(f"Using unmapped station {station['name']} (no mapped stations within {max_distance_km} km)")
        
        # Get station ID (prefer awdb_station_id, fall back to snotelr_site_id for backward compatibility)
        station_id = station.get("awdb_station_id") or station.get("snotelr_site_id")
        
        return {
            "triplet": station.get("triplet", f"{station_id}:WY:SNTL" if station_id else None),
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "elevation_ft": station.get("elevation_ft", 0),
            "distance_km": distance_km,
            "awdb_station_id": station_id,  # AWDB station ID (same format as snotelr_site_id)
            "AWDB_site_id": station_id  # Alias for backward compatibility with tests
        }
    
    def get_snow_data(self, lat: float, lon: float, date: datetime, elevation_ft: Optional[float] = None) -> Dict:
        """
        Get snow data for location and date using AWDB REST API.
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date for snow data
            elevation_ft: Optional elevation in feet (from DEM). If provided, used for
                         better elevation-based estimation when no station is nearby.
        
        Returns:
            Dictionary with snow data: depth, swe, crust, station, station_distance_km
        """
        import pandas as pd
        import traceback
        
        # Check request cache first (fast path for exact same location/date)
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.request_cache:
            logger.debug(f"Cache hit for location ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}")
            return self.request_cache[cache_key]
        
        # Find nearest SNOTEL station
        station = self._find_nearest_station(lat, lon)
        
        if station is None:
            # No station nearby, use elevation-based estimate
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft, station_distance_km=None)
            self.request_cache[cache_key] = result
            self._trim_caches()  # Prevent unbounded growth
            return result
        
        try:
            # Get AWDB station ID
            station_id = station.get("awdb_station_id")
            
            if station_id is None or pd.isna(station_id):
                # Station not mapped to AWDB station ID
                logger.debug(f"Station {station['name']} has no AWDB station ID, using elevation estimate")
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft, station_distance_km=station.get("distance_km"))
                self.request_cache[cache_key] = result
                self._trim_caches()  # Prevent unbounded growth
                return result
            
            station_id = str(station_id)
            date_str = date.strftime("%Y-%m-%d")
            
            # Check station data cache first
            # We cache by station ID and date range to minimize API calls
            # For efficiency, we fetch a date range around the requested date
            station_cache_key = station_id
            
            # Determine date range for caching (fetch ±30 days to cache more data)
            begin_date = (date - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = (date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Check if this is historical data (older than 30 days) for caching
            is_historical = (datetime.now() - date).days > 30
            
            # Check persistent cache first (if enabled and historical)
            df = None
            if self.use_cache and self.data_cache and is_historical:
                df = self.data_cache.get(station_id, begin_date, end_date)
                if df is not None:
                    logger.debug(f"Using persistent cache for station {station['name']} (station_id {station_id})")
                    # Also store in in-memory cache for faster access
                    self.station_data_cache[station_cache_key] = df
            
            # Check in-memory cache if persistent cache didn't have it
            if df is None and station_cache_key in self.station_data_cache:
                # Check if cached data covers the requested date
                df = self.station_data_cache[station_cache_key].copy()  # Make a copy to avoid modifying cached data
                # Ensure date column is datetime
                df['date'] = pd.to_datetime(df['date'])
                df_dates = df['date']
                if df_dates.min() <= pd.Timestamp(date) <= df_dates.max():
                    # Use cached data
                    logger.debug(f"Using in-memory cache for station {station['name']} (station_id {station_id})")
                else:
                    # Cached data doesn't cover this date, fetch new data
                    logger.debug(f"Cached data for station {station['name']} doesn't cover {date_str}, fetching from API")
                    df = None  # Will fetch below
            
            # Fetch from API if not in cache
            if df is None:
                logger.debug(f"Fetching data for station {station['name']} (station_id {station_id}) from AWDB API")
                df = self._fetch_station_data_from_awdb(station_id, begin_date, end_date)
                if df is not None:
                    # Ensure date column is datetime before caching
                    df['date'] = pd.to_datetime(df['date'])
                    # Store in in-memory cache
                    self.station_data_cache[station_cache_key] = df
                    self._trim_caches()  # Prevent unbounded growth
                    
                    # Store in persistent cache if enabled and historical
                    if self.use_cache and self.data_cache and is_historical:
                        self.data_cache.put(station_id, begin_date, end_date, df)
                        logger.debug(f"Cached station {station['name']} data for future use")
            
            if df is None or len(df) == 0:
                # Only warn once per station to avoid log spam (shared across all instances)
                warning_key = (station_id, 'no_data')
                with AWDBClient._warning_lock:
                    if warning_key not in AWDBClient._warned_stations:
                        logger.warning(f"No AWDB data available for station {station['name']} (station_id {station_id})")
                        AWDBClient._warned_stations.add(warning_key)
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft, station_distance_km=station.get("distance_km"))
                self.request_cache[cache_key] = result
                return result
            
            # Ensure date column is datetime (should already be, but ensure it for safety)
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Filter to the specific date
            date_data = df[df['date'].dt.date == date.date()]
            
            if len(date_data) == 0:
                # No data for this exact date, try to get closest date within ±7 days
                logger.debug(f"No data for {date_str}, using closest available date")
                date_range = pd.date_range(date - timedelta(days=7), date + timedelta(days=7))
                date_data = df[df['date'].isin(date_range)]
                if len(date_data) > 0:
                    # Use closest date
                    closest_date_label = (date_data['date'] - pd.Timestamp(date)).abs().idxmin()
                    date_data = date_data.loc[[closest_date_label]]
            
            if len(date_data) == 0:
                # Only warn once per station to avoid log spam (shared across all instances)
                warning_key = (station_id, 'no_date_data')
                with AWDBClient._warning_lock:
                    if warning_key not in AWDBClient._warned_stations:
                        logger.warning(f"No AWDB data available for station {station['name']} (station_id {station_id}) near {date_str}")
                        AWDBClient._warned_stations.add(warning_key)
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft, station_distance_km=station.get("distance_km"))
                self.request_cache[cache_key] = result
                return result
            
            # Extract snow depth and SWE from AWDB response
            # AWDB returns data in inches (not mm!)
            row = date_data.iloc[0]
            
            # AWDB element codes: WTEQ (Snow Water Equivalent), SNWD (Snow Depth)
            swe = row.get('WTEQ', 0)
            if pd.isna(swe):
                swe = 0
            swe = float(swe)  # Already in inches!
            
            # Snow depth (if available)
            snow_depth = row.get('SNWD', None)
            if snow_depth is not None and not pd.isna(snow_depth):
                snow_depth = float(snow_depth)  # Already in inches!
            else:
                # Estimate snow depth from SWE (typical density ~0.25)
                snow_depth = swe / 0.25 if swe > 0 else 0.0
            
            # Detect crusting (SWE high relative to depth)
            density = swe / snow_depth if snow_depth > 0 else 0
            crust_detected = density > 0.35  # High density = crust
            
            result = {
                "depth": float(snow_depth),
                "swe": float(swe),
                "crust": crust_detected,
                "station": station["name"],
                "station_distance_km": station["distance_km"]
            }
            
            # Cache result for this specific location/date (fast lookup)
            self.request_cache[cache_key] = result
            self._trim_caches()  # Prevent unbounded growth
            return result
        
        except Exception as e:
            logger.warning(f"Error retrieving SNOTEL data via AWDB API: {e}, using elevation estimate")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Pass elevation if available for better estimation
            # Include station distance if station was found (station variable may exist in outer scope)
            station_dist = station.get("distance_km") if station is not None else None
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft, station_distance_km=station_dist)
            self.request_cache[cache_key] = result
            self._trim_caches()  # Prevent unbounded growth
            return result
    
    def _fetch_station_data_from_awdb(self, station_id: str, begin_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch snow data for a station from AWDB API.
        
        Args:
            station_id: AWDB station ID
            begin_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: date, WTEQ, SNWD, or None if error
        """
        import pandas as pd
        import traceback
        import os
        
        # Skip API call in test environment to avoid slow tests
        if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING'):
            logger.debug(f"Skipping AWDB API call in test environment for station {station_id}")
            return None
        
        try:
            # Construct station triplet
            triplet = f"{station_id}:WY:SNTL"
            
            # Query AWDB API with retry logic for 5xx errors
            params = {
                'stationTriplets': triplet,
                'elements': 'WTEQ,SNWD',
                'beginDate': begin_date,
                'endDate': end_date,
                'duration': 'DAILY',
                'returnFlags': 'false',
                'returnOriginalValues': 'false',
                'returnSuspectData': 'false',
            }
            
            # Retry logic with exponential backoff for 5xx errors, timeouts, and connection errors
            max_retries = 3
            base_delay = 1.0  # Start with 1 second
            import time
            import threading
            
            # Rate limiting: ensure minimum 100ms between API requests (shared across all instances)
            if not hasattr(AWDBClient, '_rate_limit_lock'):
                AWDBClient._rate_limit_lock = threading.Lock()
                AWDBClient._last_request_time = 0
                AWDBClient.MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests (10 req/sec max)
            
            for attempt in range(max_retries + 1):
                try:
                    # Rate limiting: ensure minimum interval between requests
                    # CRITICAL: Only hold lock for time check, not during API call!
                    with AWDBClient._rate_limit_lock:
                        elapsed = time.time() - AWDBClient._last_request_time
                        if elapsed < AWDBClient.MIN_REQUEST_INTERVAL:
                            sleep_time = AWDBClient.MIN_REQUEST_INTERVAL - elapsed
                            time.sleep(sleep_time)
                        # Update last request time BEFORE making the call
                        AWDBClient._last_request_time = time.time()
                    
                    # Make API call OUTSIDE the lock to avoid serializing all workers
                    # Reduced timeout from 30s to 10s for faster failure detection
                    response = requests.get(f"{self.AWDB_BASE_URL}/data", params=params, timeout=10)
                    
                    # Check for 5xx errors before raising
                    # Use getattr to safely check status_code (handles mocks in tests)
                    status_code = getattr(response, 'status_code', None)
                    if status_code is not None and isinstance(status_code, int) and status_code >= 500:
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                            logger.debug(f"AWDB API 5xx error ({status_code}) for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        else:
                            # Final attempt failed, raise the error
                            response.raise_for_status()
                    
                    # For non-5xx errors, raise immediately if there's an error
                    response.raise_for_status()
                    # Success - break out of retry loop
                    break
                    
                except requests.exceptions.HTTPError as e:
                    # HTTPError from raise_for_status() - check if it's a 5xx error
                    if hasattr(e, 'response') and e.response is not None:
                        error_status_code = getattr(e.response, 'status_code', None)
                        if error_status_code is not None and isinstance(error_status_code, int) and error_status_code >= 500:
                            if attempt < max_retries:
                                delay = base_delay * (2 ** attempt)
                                logger.debug(f"AWDB API 5xx error ({error_status_code}) for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                                time.sleep(delay)
                                continue
                    # Non-5xx HTTP error or final attempt - re-raise
                    raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    # Retry timeouts and connection errors (network issues are often transient)
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        error_type = "timeout" if isinstance(e, requests.exceptions.Timeout) else "connection error"
                        logger.debug(f"AWDB API {error_type} for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    # Final attempt failed - re-raise
                    raise
                except requests.exceptions.RequestException as e:
                    # Other request exceptions - don't retry
                    raise
            
            # If we get here, we should have a successful response
            data = response.json()
            
            if not data or len(data) == 0:
                logger.debug(f"No data returned from AWDB API for station {station_id}")
                return None
            
            # Parse AWDB response structure
            # Response is: [{"stationTriplet": "...", "data": [{"stationElement": {...}, "values": [...]}]}]
            station_data = data[0]
            elements_data = station_data.get('data', [])
            
            # Build DataFrame from response
            records = []
            wteq_values = {}
            snwd_values = {}
            
            for element_data in elements_data:
                element_code = element_data['stationElement']['elementCode']
                values = element_data.get('values', [])
                
                for value_obj in values:
                    date_str = value_obj['date']
                    value = value_obj.get('value')
                    
                    if element_code == 'WTEQ':
                        wteq_values[date_str] = value
                    elif element_code == 'SNWD':
                        snwd_values[date_str] = value
            
            # Combine all dates
            all_dates = set(wteq_values.keys()) | set(snwd_values.keys())
            
            for date_str in sorted(all_dates):
                records.append({
                    'date': date_str,
                    'WTEQ': wteq_values.get(date_str),
                    'SNWD': snwd_values.get(date_str),
                })
            
            if not records:
                logger.debug(f"No data records parsed from AWDB API response for station {station_id}")
                return None
            
            df = pd.DataFrame(records)
            logger.debug(f"Fetched {len(df)} records from AWDB API for station {station_id} ({begin_date} to {end_date})")
            return df
            
        except requests.exceptions.RequestException as e:
            # Only warn once per station to avoid log spam (shared across all instances)
            with AWDBClient._warning_lock:
                if station_id not in AWDBClient._warned_api_failures:
                    logger.warning(f"AWDB API request failed for station {station_id}: {e}")
                    AWDBClient._warned_api_failures.add(station_id)
            return None
        except Exception as e:
            logger.warning(f"Error parsing AWDB API response for station {station_id}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _estimate_snow_from_elevation(self, lat: float, lon: float, 
                                     date: datetime, elevation_ft: Optional[float] = None,
                                     station_distance_km: Optional[float] = None) -> Dict:
        """
        Estimate snow based on elevation and date (fallback).
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date for estimation
            elevation_ft: Optional elevation in feet. If None, attempts to sample from DEM,
                         otherwise defaults to 8500 ft.
            station_distance_km: Optional distance to nearest SNOTEL station in km.
                                If provided, included in result for data quality tracking.
        """
        # Use provided elevation, or try to sample from DEM, or use default
        if elevation_ft is not None:
            elevation = elevation_ft
        else:
            # Try to sample from DEM if available
            elevation = self._get_elevation_from_dem(lat, lon)
        
        month = date.month
        
        # Simple heuristic
        if month in [12, 1, 2]:  # Winter
            snow_depth = max(0, (elevation - 6000) / 100)
        elif month in [3, 4]:  # Spring
            snow_depth = max(0, (elevation - 7000) / 150)
        elif month in [10, 11]:  # Fall
            snow_depth = max(0, (elevation - 9000) / 200)
        else:  # Summer
            snow_depth = max(0, (elevation - 10000) / 100)
        
        return {
            "depth": snow_depth,
            "swe": snow_depth * 0.25,
            "crust": False,
            "station": None,
            "station_distance_km": station_distance_km
        }
    
    def _get_elevation_from_dem(self, lat: float, lon: float) -> float:
        """
        Attempt to get elevation from DEM if available.
        
        Falls back to default elevation if DEM not accessible.
        """
        try:
            # Try to use DEM if available in data_dir
            dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
            if dem_path.exists():
                try:
                    import rasterio
                    from rasterio.warp import transform_geom
                    from shapely.geometry import Point, mapping
                    
                    # Sample DEM at location
                    point = Point(lon, lat)
                    geom = mapping(point)
                    
                    # Transform to DEM CRS if needed (assuming DEM is in UTM or similar)
                    # For simplicity, assume DEM is in EPSG:4326 or can be sampled directly
                    with rasterio.open(dem_path) as src:
                        # Sample the point
                        for val in src.sample([(lon, lat)]):
                            if val[0] is not None and not (val[0] < -1000 or val[0] > 15000):
                                # Convert meters to feet if needed
                                # Assume DEM is in meters, convert to feet
                                elevation_m = float(val[0])
                                if elevation_m < 500:  # Likely in feet already
                                    return elevation_m
                                else:  # Likely in meters
                                    return elevation_m * 3.28084  # meters to feet
                except Exception as e:
                    logger.debug(f"Could not sample DEM for elevation: {e}")
                    pass
            
        except Exception as e:
            logger.debug(f"Error accessing DEM for elevation: {e}")
            pass
        
        # Default fallback
        return 8500.0


class WeatherClient:
    """
    Client for weather data using multiple sources optimized for each data type.

    Data Sources:
        - PRISM: Historical temperature and precipitation (bulk downloaded, local files)
        - ERA5: Historical cloud cover (bulk downloaded, local NetCDF files)
        - OpenMeteo: Forecast/current data ONLY (API calls, rate-limited)

    OpenMeteo is NOT suitable for bulk historical data retrieval due to aggressive
    rate limiting. Use PRISM and ERA5 for training data pipelines.
    """

    def __init__(self, data_dir: Optional[Path] = None, use_real_data: bool = True):
        """
        Initialize weather client.

        Args:
            data_dir: Data directory for PRISM/ERA5 cache
            use_real_data: If False, uses placeholder values (for testing/fallback)
        """
        self.use_real_data = use_real_data
        self.cache = {}
        self.era5_client = None

        if data_dir is None:
            data_dir = Path("data")
        self.data_dir = data_dir

        if use_real_data:
            try:
                from .prism_client import PRISMClient
                from .openmeteo_client import OpenMeteoClient
                from .era5_client import ERA5CloudClient

                self.prism_client = PRISMClient(data_dir / "prism")
                self.openmeteo_client = OpenMeteoClient()

                # ERA5 for historical cloud cover (optional - gracefully degrade if not available)
                try:
                    self.era5_client = ERA5CloudClient(data_dir / "era5")
                    coverage = self.era5_client.get_coverage_info()
                    if coverage['available']:
                        logger.info(f"ERA5 cloud cover available: years {coverage['year_range'][0]}-{coverage['year_range'][1]}")
                    else:
                        logger.warning("ERA5 cloud cover data not downloaded. Run scripts/bulk_download_era5_cloud.py")
                except Exception as e:
                    logger.warning(f"ERA5 client not available: {e}. Cloud cover will use fallback values.")
                    self.era5_client = None

            except ImportError as e:
                logger.warning(f"Weather clients not available: {e}, using placeholder values")
                self.use_real_data = False
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather for location and date"""
        
        # Check cache
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check if date is in future (forecast) or past (historical)
        today = datetime.now().date()
        target_date = date.date()
        
        if target_date > today:
            result = self._get_forecast(lat, lon, date)
        else:
            result = self._get_historical(lat, lon, date)
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    def _get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast using Open-Meteo"""
        if not self.use_real_data:
            # Placeholder fallback
            return {
                "temp": 45.0,
                "temp_high": 55.0,
                "temp_low": 35.0,
                "precip_7d": 0.3,
                "cloud_cover": 30,
                "wind_mph": 10
            }
        
        try:
            # Get forecast for target date
            forecast = self.openmeteo_client.get_forecast_for_date(
                lat, lon, date.strftime("%Y-%m-%d")
            )
            
            if forecast:
                # Get 7-day precipitation window
                start_date = date - timedelta(days=7)
                forecasts_7d = self.openmeteo_client.get_forecast_for_location(lat, lon)
                precip_7d = sum(
                    f.get("precipitation_inches", 0) or 0
                    for f in forecasts_7d[:7]
                )

                return {
                    "temp": forecast.get("temp_mean_f", 45.0),
                    "temp_high": forecast.get("temp_max_f", 55.0),
                    "temp_low": forecast.get("temp_min_f", 35.0),
                    "precip_7d": precip_7d,
                    "cloud_cover": forecast.get("cloud_cover_percent", 30),
                    "wind_mph": 10  # Open-Meteo doesn't provide wind in daily data
                }
            else:
                raise ValueError("Forecast not available for target date")
                
        except Exception as e:
            logger.warning(f"Failed to get Open-Meteo forecast: {e}, using placeholder")
            return {
                "temp": 45.0,
                "temp_high": 55.0,
                "temp_low": 35.0,
                "precip_7d": 0.3,
                "cloud_cover": 30,
                "wind_mph": 10
            }
    
    def _get_historical(self, lat: float, lon: float, date: datetime) -> Dict:
        """
        Get historical weather using PRISM (temp/precip) and ERA5 (cloud cover).

        Data sources:
            - Temperature: PRISM (4km resolution, local files)
            - Precipitation: PRISM (4km resolution, local files)
            - Cloud cover: ERA5 (25km resolution, local NetCDF files)

        OpenMeteo is NOT used for historical data due to rate limiting.
        If PRISM/ERA5 data is not available, returns fallback values.
        """
        if not self.use_real_data:
            # Placeholder fallback
            return {
                "temp": 42.0,
                "temp_high": 52.0,
                "temp_low": 32.0,
                "precip_7d": 0.5,
                "cloud_cover": 40,
                "wind_mph": 12
            }

        # Default fallback values (used only if PRISM data unavailable)
        # NOTE: If these values appear in output, it indicates PRISM data retrieval failed
        temp_mean_f = 42.0  # Default fallback (should be replaced by PRISM data)
        temp_high_f = 52.0
        temp_low_f = 32.0
        precip_7d = 0.5
        cloud_cover_pct = 40.0

        # Get temperature from PRISM
        try:
            temp_data = self.prism_client.get_temperature(lat, lon, date)

            if temp_data is not None and not all(v is None for v in temp_data.values()):
                # Convert temps to Fahrenheit
                if temp_data.get("temp_mean_c") is not None:
                    temp_mean_f = (temp_data["temp_mean_c"] * 9/5) + 32
                    logger.debug(f"PRISM temperature for ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}: "
                               f"{temp_data['temp_mean_c']:.2f}°C = {temp_mean_f:.2f}°F")
                else:
                    logger.warning(f"PRISM temp_mean_c is None for ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}, "
                                 f"using fallback {temp_mean_f:.1f}°F")
                    
                if temp_data.get("temp_max_c") is not None:
                    temp_high_f = (temp_data["temp_max_c"] * 9/5) + 32
                elif temp_data.get("temp_max_f") is not None:
                    temp_high_f = temp_data["temp_max_f"]
                else:
                    logger.debug(f"PRISM temp_max not available, using fallback {temp_high_f:.1f}°F")
                    
                if temp_data.get("temp_min_c") is not None:
                    temp_low_f = (temp_data["temp_min_c"] * 9/5) + 32
                elif temp_data.get("temp_min_f") is not None:
                    temp_low_f = temp_data["temp_min_f"]
                else:
                    logger.debug(f"PRISM temp_min not available, using fallback {temp_low_f:.1f}°F")
            else:
                logger.warning(f"PRISM temperature not available for ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}, "
                             f"using fallback values (mean={temp_mean_f:.1f}°F)")
        except Exception as e:
            logger.warning(f"Failed to get PRISM temperature for ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}: {e}, "
                         f"using fallback {temp_mean_f:.1f}°F")
            import traceback
            logger.debug(f"PRISM temperature error traceback: {traceback.format_exc()}")

        # Get 7-day precipitation from PRISM
        try:
            start_date = date - timedelta(days=7)
            precip_mm = 0.0
            current_date = start_date
            while current_date <= date:
                ppt = self.prism_client.get_precipitation(lat, lon, current_date)
                if ppt is not None:
                    precip_mm += ppt
                current_date += timedelta(days=1)
            precip_7d = precip_mm / 25.4  # Convert mm to inches
        except Exception as e:
            logger.debug(f"Failed to get PRISM precipitation: {e}")

        # Get cloud cover from ERA5 (NOT OpenMeteo - it's rate-limited)
        if self.era5_client is not None:
            try:
                era5_cloud = self.era5_client.get_cloud_cover(lat, lon, date)
                if era5_cloud is not None:
                    cloud_cover_pct = era5_cloud
                else:
                    logger.debug(f"ERA5 cloud cover not available for {date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.debug(f"Failed to get ERA5 cloud cover: {e}")
        else:
            logger.debug("ERA5 client not available, using fallback cloud cover")

        return {
            "temp": temp_mean_f,
            "temp_high": temp_high_f,
            "temp_low": temp_low_f,
            "precip_7d": precip_7d,
            "cloud_cover": cloud_cover_pct,
            "wind_mph": 12  # Wind data not available from PRISM/ERA5 daily
        }
    
class SatelliteClient:
    """Client for satellite imagery (NDVI) using AppEEARS"""
    
    def __init__(self, use_real_data: bool = True, appeears_username: Optional[str] = None, 
                 appeears_password: Optional[str] = None):
        """
        Initialize satellite client.
        
        Args:
            use_real_data: If False, uses placeholder values (for testing/fallback)
            appeears_username: AppEEARS username (or from env)
            appeears_password: AppEEARS password (or from env)
        """
        self.use_real_data = use_real_data
        self.cache = {}
        self.appeears_client = None
        
        if use_real_data:
            try:
                from .appeears_client import AppEEARSClient
                import os
                
                username = appeears_username or os.getenv("APPEEARS_USERNAME")
                password = appeears_password or os.getenv("APPEEARS_PASSWORD")
                
                if username and password:
                    self.appeears_client = AppEEARSClient(username, password)
                    logger.info(f"AppEEARS client initialized successfully (username: {username[:3]}***)")
                else:
                    logger.warning("AppEEARS credentials not available, using placeholder NDVI. "
                                 "Set APPEEARS_USERNAME and APPEEARS_PASSWORD environment variables.")
                    self.use_real_data = False
            except ImportError:
                logger.warning("AppEEARS client not available, using placeholder NDVI")
                self.use_real_data = False
            except Exception as e:
                logger.warning(f"Failed to initialize AppEEARS client: {e}, using placeholder NDVI")
                self.use_real_data = False
    
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """
        Get NDVI for location and date.
        
        Note: For batch processing, use extract_ndvi_batch() method instead
        as it's more efficient than calling this for individual points.
        """
        # Check cache
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.use_real_data or not self.appeears_client:
            # Placeholder fallback with seasonal variation
            month = date.month
            if month in [6, 7, 8]:  # Summer - high NDVI
                ndvi = 0.70
            elif month in [9, 10]:  # Fall - declining
                ndvi = 0.55
            elif month in [11, 12, 1, 2, 3]:  # Winter - low
                ndvi = 0.30
            else:  # Spring - increasing
                ndvi = 0.50
            
            result = {
                "ndvi": ndvi,
                "age_days": 8,
                "irg": 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0,
                "cloud_free": True
            }
            self.cache[cache_key] = result
            return result
        
        try:
            # Check if AppEEARS client is available
            if self.appeears_client is None:
                raise ValueError("AppEEARS client not initialized. Check credentials.")
            
            # Use AppEEARS to get NDVI
            points = [(lat, lon, date.strftime("%Y-%m-%d"))]
            try:
                # Use shorter timeout for single-point requests (5 minutes)
                ndvi_df = self.appeears_client.get_ndvi_for_points(
                    points,
                    product="modis_ndvi",
                    date_buffer_days=7,
                    max_wait_minutes=5
                )
            except (TimeoutError, RuntimeError) as e:
                logger.warning(f"AppEEARS request timed out or failed: {e}, using placeholder NDVI")
                # Fall back to placeholder
                month = date.month
                if month in [6, 7, 8]:  # Summer - high NDVI
                    ndvi = 0.70
                elif month in [9, 10]:  # Fall - decreasing NDVI
                    ndvi = 0.55
                elif month in [11, 12, 1, 2, 3]:  # Winter/Early Spring - low NDVI
                    ndvi = 0.30
                else:  # Spring - increasing NDVI
                    ndvi = 0.50
                
                result = {
                    "ndvi": ndvi,
                    "age_days": 16,
                    "irg": 0.0,
                    "cloud_free": False
                }
                self.cache[cache_key] = result
                return result
            
            if len(ndvi_df) > 0 and pd.notna(ndvi_df.iloc[0]["ndvi"]):
                ndvi_value = ndvi_df.iloc[0]["ndvi"]
                qa_flags = ndvi_df.iloc[0].get("qa_flags", 0)
                
                # Calculate IRG (simple approximation - would need time series for accurate)
                month = date.month
                irg = 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0
                
                result = {
                    "ndvi": float(ndvi_value),
                    "age_days": 8,  # Approximate (would need image date from AppEEARS)
                    "irg": irg,
                    "cloud_free": qa_flags == 0 if qa_flags is not None else True
                }
            else:
                # Fallback to placeholder if no data
                month = date.month
                if month in [6, 7, 8]:
                    ndvi = 0.70
                elif month in [9, 10]:
                    ndvi = 0.55
                elif month in [11, 12, 1, 2, 3]:
                    ndvi = 0.30
                else:
                    ndvi = 0.50
                
                result = {
                    "ndvi": ndvi,
                    "age_days": 16,
                    "irg": 0.0,
                    "cloud_free": False
                }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get NDVI from AppEEARS: {e}, using placeholder")
            # Fallback to placeholder
            month = date.month
            if month in [6, 7, 8]:
                ndvi = 0.70
            elif month in [9, 10]:
                ndvi = 0.55
            elif month in [11, 12, 1, 2, 3]:
                ndvi = 0.30
            else:
                ndvi = 0.50
            
            result = {
                "ndvi": ndvi,
                "age_days": 16,
                "irg": 0.0,
                "cloud_free": False
            }
            self.cache[cache_key] = result
            return result
    
    def extract_ndvi_batch(
        self,
        points: List[Tuple[float, float, datetime]],
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Extract NDVI for multiple points efficiently using batch processing.
        
        Args:
            points: List of (lat, lon, date) tuples
            batch_size: Number of points per AppEEARS request
            
        Returns:
            DataFrame with latitude, longitude, date, ndvi, qa_flags columns
        """
        if not self.use_real_data or not self.appeears_client:
            # Return placeholder DataFrame
            import pandas as pd
            data = []
            for lat, lon, date in points:
                month = date.month
                if month in [6, 7, 8]:
                    ndvi = 0.70
                elif month in [9, 10]:
                    ndvi = 0.55
                elif month in [11, 12, 1, 2, 3]:
                    ndvi = 0.30
                else:
                    ndvi = 0.50
                
                data.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": date.strftime("%Y-%m-%d"),
                    "ndvi": ndvi,
                    "qa_flags": 0
                })
            return pd.DataFrame(data)
        
        try:
            # Convert to AppEEARS format
            appeears_points = [
                (lat, lon, date.strftime("%Y-%m-%d"))
                for lat, lon, date in points
            ]
            
            # Process in batches
            all_results = []
            for i in range(0, len(appeears_points), batch_size):
                batch = appeears_points[i:i + batch_size]
                try:
                    # Use longer timeout for batch requests (15 minutes)
                    batch_results = self.appeears_client.get_ndvi_for_points(
                        batch,
                        product="modis_ndvi",
                        date_buffer_days=7,
                        max_wait_minutes=15
                    )
                    all_results.append(batch_results)
                except (TimeoutError, RuntimeError) as e:
                    logger.warning(f"AppEEARS batch request timed out or failed: {e}, skipping batch")
                    # Continue with other batches rather than failing completely
                    continue
            
            if not all_results:
                # All batches failed - return empty DataFrame
                logger.warning("All AppEEARS batch requests failed, returning empty DataFrame")
                return pd.DataFrame(columns=["latitude", "longitude", "date", "ndvi", "qa_flags"])
            
            return pd.concat(all_results, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Failed to extract NDVI batch: {e}")
            # Return placeholder DataFrame
            import pandas as pd
            data = []
            for lat, lon, date in points:
                month = date.month
                if month in [6, 7, 8]:
                    ndvi = 0.70
                elif month in [9, 10]:
                    ndvi = 0.55
                elif month in [11, 12, 1, 2, 3]:
                    ndvi = 0.30
                else:
                    ndvi = 0.50
                
                data.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": date.strftime("%Y-%m-%d"),
                    "ndvi": ndvi,
                    "qa_flags": None
                })
            return pd.DataFrame(data)
    
    def get_integrated_ndvi(self, lat: float, lon: float, 
                           start_date: datetime, end_date: datetime) -> float:
        """Get integrated NDVI over date range
        
        Note: AppEEARS is an async API that requires task submission and waiting.
        For historical data or real-time processing, this is too slow.
        Returns placeholder 60.0 if AppEEARS is not available or for historical dates.
        """
        # For historical data (before 2015), AppEEARS is not practical
        if start_date.year < 2015:
            logger.debug(f"Skipping AppEEARS for historical date {start_date.year}, returning placeholder 60.0")
            return 60.0
        
        if not self.use_real_data:
            logger.debug(f"SatelliteClient.use_real_data=False, returning placeholder 60.0 for integrated NDVI")
            return 60.0  # Placeholder
        
        if not self.appeears_client:
            logger.debug(f"AppEEARS client not initialized for integrated NDVI at ({lat:.4f}, {lon:.4f}), "
                        f"date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. "
                        f"Returning placeholder 60.0.")
            return 60.0  # Placeholder
        
        try:
            # Sample every 16 days (Landsat revisit cycle)
            current_date = start_date
            ndvi_values = []
            dates_checked = []
            
            logger.debug(f"Getting integrated NDVI from AppEEARS for ({lat:.4f}, {lon:.4f}), "
                        f"date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            while current_date <= end_date:
                dates_checked.append(current_date.strftime('%Y-%m-%d'))
                ndvi_data = self.get_ndvi(lat, lon, current_date)
                if ndvi_data.get("ndvi") is not None:
                    ndvi_values.append(ndvi_data["ndvi"])
                    logger.debug(f"  {current_date.strftime('%Y-%m-%d')}: NDVI={ndvi_data['ndvi']:.3f}")
                else:
                    logger.debug(f"  {current_date.strftime('%Y-%m-%d')}: No NDVI data available")
                current_date += timedelta(days=16)
            
            if not ndvi_values:
                logger.warning(f"No NDVI values found from AppEEARS for integrated NDVI at ({lat:.4f}, {lon:.4f}), "
                             f"date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. "
                             f"Checked dates: {', '.join(dates_checked)}. Returning placeholder 60.0.")
                return 60.0
            
            # Integrated NDVI = sum of NDVI values
            integrated_ndvi = sum(ndvi_values)
            logger.debug(f"Integrated NDVI from AppEEARS: {integrated_ndvi:.2f} (sum of {len(ndvi_values)} values: {ndvi_values})")
            return integrated_ndvi
            
        except Exception as e:
            logger.warning(f"Failed to get integrated NDVI from AppEEARS at ({lat:.4f}, {lon:.4f}): {e}")
            import traceback
            logger.debug(f"Integrated NDVI error traceback: {traceback.format_exc()}")
            return 60.0