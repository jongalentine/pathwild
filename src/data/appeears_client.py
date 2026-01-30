"""
NASA AppEEARS Client for NDVI Data

Handles NDVI data retrieval via NASA AppEEARS API for both training and inference.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
import sqlite3
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logger = logging.getLogger(__name__)


class AppEEARSCache:
    """
    Persistent cache for NDVI data retrieved from NASA AppEEARS API.
    
    Uses SQLite for efficient storage and retrieval. Cache keys are based on:
    - Location (lat, lon rounded to 4 decimal places ~11m precision)
    - Date (to the day)
    - Product name
    - date_buffer_days
    
    Since historical data never changes, cached entries are permanent.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize AppEEARS cache.
        
        Args:
            cache_dir: Directory for cache database. Defaults to data/cache/appeears_cache.db
        """
        if cache_dir is None:
            cache_dir = Path('data/cache')
        else:
            cache_dir = Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = cache_dir / 'appeears_cache.db'
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        logger.info(f"AppEEARS cache initialized: {self.cache_db}")
    
    def _init_db(self):
        """Initialize cache database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                # Create table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS appeears_cache (
                        cache_key TEXT PRIMARY KEY,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        date TEXT NOT NULL,
                        product TEXT NOT NULL,
                        date_buffer_days INTEGER NOT NULL,
                        ndvi REAL,
                        qa_flags INTEGER,
                        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Create index for fast lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_location_date 
                    ON appeears_cache(lat, lon, date)
                """)
                conn.commit()
            finally:
                conn.close()
    
    def _make_cache_key(
        self,
        lat: float,
        lon: float,
        date_str: str,
        product: str,
        date_buffer_days: int
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            lat: Latitude (rounded to 4 decimal places)
            lon: Longitude (rounded to 4 decimal places)
            date_str: Date string (YYYY-MM-DD)
            product: Product name
            date_buffer_days: Date buffer days parameter
            
        Returns:
            Cache key string
        """
        # Round coordinates to 4 decimal places (~11m precision)
        lat_rounded = round(lat, 4)
        lon_rounded = round(lon, 4)
        
        # Create hash for cache key
        key_data = f"{lat_rounded:.4f},{lon_rounded:.4f},{date_str},{product},{date_buffer_days}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(
        self,
        lat: float,
        lon: float,
        date_str: str,
        product: str,
        date_buffer_days: int
    ) -> Optional[float]:
        """
        Retrieve NDVI value from cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            date_str: Date string (YYYY-MM-DD)
            product: Product name
            date_buffer_days: Date buffer days parameter
            
        Returns:
            NDVI value (float) or None if not in cache
        """
        cache_key = self._make_cache_key(lat, lon, date_str, product, date_buffer_days)
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("""
                    SELECT ndvi
                    FROM appeears_cache
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                ndvi = row[0]
                if ndvi is None:
                    return None
                
                return float(ndvi)
            finally:
                conn.close()
    
    def put(
        self,
        lat: float,
        lon: float,
        date_str: str,
        product: str,
        date_buffer_days: int,
        ndvi: float,
        qa_flags: int = 0
    ):
        """
        Store NDVI value in cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            date_str: Date string (YYYY-MM-DD)
            product: Product name
            date_buffer_days: Date buffer days parameter
            ndvi: NDVI value to cache
            qa_flags: Quality assurance flags (optional)
        """
        cache_key = self._make_cache_key(lat, lon, date_str, product, date_buffer_days)
        
        # Round coordinates for storage
        lat_rounded = round(lat, 4)
        lon_rounded = round(lon, 4)
        
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO appeears_cache
                    (cache_key, lat, lon, date, product, date_buffer_days, ndvi, qa_flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    lat_rounded,
                    lon_rounded,
                    date_str,
                    product,
                    date_buffer_days,
                    ndvi,
                    qa_flags
                ))
                conn.commit()
            finally:
                conn.close()
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with 'total_entries', 'total_size_mb' keys
        """
        with self._lock:
            conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM appeears_cache")
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
                conn.execute("DELETE FROM appeears_cache")
                conn.commit()
                logger.info("AppEEARS cache cleared")
            finally:
                conn.close()


class AppEEARSClient:
    """Client for NASA AppEEARS API to retrieve NDVI data"""
    
    API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
    
    # Configurable batching parameters (can be overridden via environment variables)
    DEFAULT_DATE_BUFFER_DAYS = int(os.getenv('APPEEARS_DATE_BUFFER_DAYS', '10'))  # Conservative optimization: 10 days
    DEFAULT_MAX_POINTS_PER_BATCH = int(os.getenv('APPEEARS_MAX_POINTS_PER_BATCH', '200'))  # Conservative optimization: 200 points
    
    # NDVI Products - MODIS products are most commonly available in AppEEARS
    # HLS products may not be available in all regions/time periods
    PRODUCTS = {
        # MODIS Vegetation Indices (most reliable, available globally)
        "modis_ndvi": "MOD13Q1.061",  # MODIS Terra 16-day NDVI (250m)
        "modis_ndvi_aqua": "MYD13Q1.061",  # MODIS Aqua 16-day NDVI (250m)
        "modis_ndvi_combined": "MxD13Q1.061",  # Combined Terra/Aqua
        
        # HLS Vegetation Indices (newer, higher resolution, but may not be available)
        "landsat_ndvi": "HLSL30.002",  # Landsat HLS - may need to calculate NDVI from reflectance
        "sentinel_ndvi": "HLSS30.002",  # Sentinel-2 HLS - may need to calculate NDVI from reflectance
        
        # Legacy mappings for backward compatibility
        "landsat_reflectance": "HLSL30.002",
        "sentinel_reflectance": "HLSS30.002",
    }
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize AppEEARS client.
        
        Args:
            username: AppEEARS username (or from APPEEARS_USERNAME env var)
            password: AppEEARS password (or from APPEEARS_PASSWORD env var)
            cache_dir: Directory for cache database (defaults to data/cache)
            use_cache: If True, use persistent cache for retrieved NDVI values (default: True)
        """
        self.username = username or os.getenv("APPEEARS_USERNAME")
        self.password = password or os.getenv("APPEEARS_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError(
                "AppEEARS credentials required. Set APPEEARS_USERNAME and APPEEARS_PASSWORD "
                "environment variables or pass username/password to constructor."
            )
        
        self.token = None
        self.token_expires = None
        self.use_cache = use_cache
        
        # Initialize HTTP session for connection pooling
        self.session = requests.Session()
        # Set default timeout
        self.session.timeout = 30
        
        # Initialize cache if enabled
        if self.use_cache:
            self.cache = AppEEARSCache(cache_dir=cache_dir)
            cache_stats = self.cache.get_stats()
            logger.info(
                f"AppEEARS cache enabled: {cache_stats['total_entries']:,} entries "
                f"({cache_stats['total_size_mb']:.2f} MB)"
            )
        else:
            self.cache = None
            logger.info("AppEEARS cache disabled")
        
        self._authenticate()
    
    def _authenticate(self) -> str:
        """Authenticate with AppEEARS API and get bearer token"""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token
        
        try:
            response = self.session.post(
                f"{self.API_BASE}/login",
                auth=(self.username, self.password),
                timeout=30
            )
            
            if response.status_code == 401:
                raise ValueError(
                    "AppEEARS authentication failed. Check your APPEEARS_USERNAME and APPEEARS_PASSWORD. "
                    "You need a NASA Earthdata account with AppEEARS access. "
                    "Sign up at: https://urs.earthdata.nasa.gov/"
                )
            
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get("token") or data.get("access_token")
            
            if not self.token:
                raise ValueError(f"AppEEARS authentication response missing token. Response: {data}")
            
            # Token typically expires in 24 hours, refresh after 20 hours
            self.token_expires = datetime.now() + timedelta(hours=20)
            
            logger.info("AppEEARS authentication successful")
            return self.token
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "AppEEARS authentication failed. Check your APPEEARS_USERNAME and APPEEARS_PASSWORD. "
                    "You need a NASA Earthdata account with AppEEARS access. "
                    "Sign up at: https://urs.earthdata.nasa.gov/"
                ) from e
            raise
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers with authentication"""
        self._authenticate()  # Refresh token if needed
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _get_layer_name(self, product_id: str) -> str:
        """
        Get the correct layer name for a product.
        
        Args:
            product_id: Product ID (e.g., "MOD13Q1.061")
            
        Returns:
            Layer name string
        """
        # For MODIS vegetation index products in AppEEARS, layer name is "_250m_16_days_NDVI"
        # Note: AppEEARS uses underscore-prefixed layer names, not the simple "NDVI" used in GEE
        if "MOD13Q1" in product_id or "MYD13Q1" in product_id or "MxD13Q1" in product_id:
            return "_250m_16_days_NDVI"
        # For HLS VI (Vegetation Index) products, the layer is typically "NDVI"
        elif "_VI" in product_id:
            return "NDVI"
        # For HLS reflectance products, use NIR band (B05) or red band (B04) for NDVI calculation
        elif "HLSL30" in product_id or "HLSS30" in product_id:
            return "B05"  # NIR band for NDVI calculation
        else:
            # Default fallback - try common NDVI layer names
            return "NDVI"
    
    def get_product_layers(self, product_id: str) -> List[Dict]:
        """
        Query AppEEARS API for available layers in a product.
        
        Useful for verifying layer names before submission.
        Uses GET /product/{product_id} endpoint.
        
        Args:
            product_id: Product ID (e.g., "HLSL30_VI.002")
            
        Returns:
            List of layer information dictionaries
        """
        try:
            response = self.session.get(
                f"{self.API_BASE}/product/{product_id}",
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            product_info = response.json()
            
            # Product info is a dict keyed by layer name
            layers = []
            for layer_name, layer_info in product_info.items():
                if isinstance(layer_info, dict):
                    layers.append({
                        "name": layer_name,
                        **layer_info
                    })
            return layers
        except Exception as e:
            logger.warning(f"Could not fetch layer info for {product_id}: {e}")
            return []
    
    def _group_points_by_date_range(
        self,
        points: List[Tuple[float, float, str]],
        date_buffer_days: Optional[int] = None,
        max_points_per_batch: Optional[int] = None
    ) -> List[List[Tuple[float, float, str]]]:
        """
        Group points by overlapping date ranges for efficient batching.
        
        Points with overlapping date ranges (considering buffer) are grouped together
        so they can be submitted in a single AppEEARS task.
        
        Args:
            points: List of (latitude, longitude, date) tuples
            date_buffer_days: Days before/after target date to search for images
                (defaults to DEFAULT_DATE_BUFFER_DAYS if not specified)
            max_points_per_batch: Maximum number of points per batch
                (defaults to DEFAULT_MAX_POINTS_PER_BATCH if not specified)
            
        Returns:
            List of point groups, where each group can be submitted as one task
        """
        if not points:
            return []
        
        # Use class defaults if not specified
        if date_buffer_days is None:
            date_buffer_days = self.DEFAULT_DATE_BUFFER_DAYS
        if max_points_per_batch is None:
            max_points_per_batch = self.DEFAULT_MAX_POINTS_PER_BATCH
        
        # Calculate date ranges for each point
        point_ranges = []
        for lat, lon, date_str in points:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = target_date - timedelta(days=date_buffer_days)
            end_date = target_date + timedelta(days=date_buffer_days)
            point_ranges.append((start_date, end_date, (lat, lon, date_str)))
        
        # Group points with overlapping date ranges
        groups = []
        used = set()
        
        for i, (start1, end1, point1) in enumerate(point_ranges):
            if i in used:
                continue
            
            group = [point1]
            used.add(i)
            
            # Find other points with overlapping date ranges
            for j, (start2, end2, point2) in enumerate(point_ranges[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if date ranges overlap
                if start1 <= end2 and start2 <= end1:
                    # Check if adding this point would exceed batch size
                    if len(group) < max_points_per_batch:
                        group.append(point2)
                        used.add(j)
                    else:
                        # Batch is full, will be handled in next iteration
                        break
            
            groups.append(group)
        
        logger.debug(f"Grouped {len(points)} points into {len(groups)} batch(es)")
        return groups
    
    def submit_point_request(
        self,
        points: List[Tuple[float, float, str]],  # List of (lat, lon, date) tuples
        product: str = "modis_ndvi",  # Default to MODIS which is more widely available
        date_buffer_days: Optional[int] = None  # Defaults to DEFAULT_DATE_BUFFER_DAYS
    ) -> str:
        """
        Submit a point extraction request to AppEEARS.
        
        Args:
            points: List of (latitude, longitude, date) tuples
                   Date format: 'YYYY-MM-DD'
            product: Product ID (default: Landsat NDVI)
            date_buffer_days: Days before/after target date to search for cloud-free images
                (defaults to DEFAULT_DATE_BUFFER_DAYS if not specified)
            
        Returns:
            Task ID for tracking the request
        """
        # Use class default if not specified
        if date_buffer_days is None:
            date_buffer_days = self.DEFAULT_DATE_BUFFER_DAYS
        
        if product not in self.PRODUCTS:
            raise ValueError(f"Unknown product: {product}. Available: {list(self.PRODUCTS.keys())}")
        
        product_id = self.PRODUCTS[product]
        
        # Prepare task request
        task_name = f"pathwild_ndvi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get layer name - query API if possible, otherwise use heuristic
        layer_name = self._get_layer_name(product_id)
        
        # Try to query actual available layers to verify
        try:
            available_layers = self.get_product_layers(product_id)
            if available_layers:
                # Look for NDVI layer in available layers
                ndvi_layers = [l for l in available_layers if "NDVI" in l.get("name", "").upper()]
                if ndvi_layers:
                    layer_name = ndvi_layers[0]["name"]
                    logger.debug(f"Found NDVI layer '{layer_name}' for product {product_id}")
                else:
                    logger.warning(f"No NDVI layer found in product {product_id}, using default: {layer_name}")
            else:
                logger.warning(f"Could not query layers for product {product_id}, using default: {layer_name}")
        except Exception as e:
            logger.warning(f"Error querying layers for {product_id}: {e}, using default: {layer_name}")
        
        # AppEEARS API supports multiple coordinates in a single task request
        # Calculate date range that covers all points (min date - buffer to max date + buffer)
        dates = [datetime.strptime(date_str, "%Y-%m-%d") for _, _, date_str in points]
        min_date = min(dates)
        max_date = max(dates)
        
        # Create date range that covers all points with buffer
        start_date = (min_date - timedelta(days=date_buffer_days)).strftime("%m-%d-%Y")
        end_date = (max_date + timedelta(days=date_buffer_days)).strftime("%m-%d-%Y")
        
        # Build coordinates array for all points
        coordinates = []
        for lat, lon, date_str in points:
            coordinates.append({
                "latitude": lat,
                "longitude": lon
            })
        
        logger.debug(f"Submitting task with {len(coordinates)} coordinate(s) for date range {start_date} to {end_date}")
        
        task_payload = {
            "task_type": "point",
            "task_name": task_name,
            "params": {
                "dates": [
                    {
                        "startDate": start_date,
                        "endDate": end_date
                    }
                ],
                "layers": [
                    {
                        "product": product_id,
                        "layer": layer_name
                    }
                ],
                "coordinates": coordinates
            }
        }
        
        # Retry logic for rate limiting (429 errors)
        max_retries = 5
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Log request details for debugging (without sensitive data)
                if attempt == 0:
                    logger.debug(f"Submitting AppEEARS task to {self.API_BASE}/task")
                    logger.debug(f"Payload structure: task_type={task_payload.get('task_type') if isinstance(task_payload, dict) else 'N/A (array)'}")
                    if isinstance(task_payload, dict):
                        logger.debug(f"Payload: task_name={task_payload.get('task_name')}, product={task_payload.get('params', {}).get('layers', [{}])[0].get('product', 'N/A')}")
                
                response = self.session.post(
                    f"{self.API_BASE}/task",
                    headers=self._get_headers(),
                    json=task_payload,
                    timeout=60
                )
                
                # Log response details for debugging
                logger.debug(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    try:
                        response_text = getattr(response, 'text', '')
                        if isinstance(response_text, str) and len(response_text) > 0:
                            logger.debug(f"Response text: {response_text[:500]}")
                    except Exception:
                        pass  # Skip logging if response.text is not accessible
                
                # Check for rate limiting before raising
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        logger.warning(f"Rate limited (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        # Last attempt failed
                        response.raise_for_status()
                
                # For other status codes, raise immediately
                response.raise_for_status()
                
                # Success - parse response and return
                result = response.json()
                logger.debug(f"Response JSON: {result}")
                
                # Try multiple response formats based on AppEEARS API documentation
                task_id = None
                if isinstance(result, dict):
                    task_id = result.get("task_id") or result.get("TaskID")
                    # Also check nested structures
                    if not task_id and "task" in result:
                        task_id = result["task"].get("task_id") if isinstance(result["task"], dict) else None
                elif isinstance(result, list) and len(result) > 0:
                    # Batch response - get first task ID
                    first_result = result[0]
                    task_id = first_result.get("task_id") or first_result.get("TaskID") if isinstance(first_result, dict) else None
                
                if not task_id:
                    logger.error(f"Could not extract task_id from AppEEARS response: {result}")
                    raise ValueError(f"Could not extract task_id from AppEEARS response. Response structure: {result}")
                
                logger.info(f"Submitted AppEEARS task: {task_id}")
                return task_id
                
            except requests.exceptions.HTTPError as e:
                # HTTPError should always have a response, but check to be safe
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_text = e.response.text if hasattr(e.response, 'text') else str(e.response)
                    except:
                        error_text = "No response text available"
                    status_code = e.response.status_code
                else:
                    # Unexpected: HTTPError without response
                    error_text = str(e)
                    status_code = None
                    logger.error(f"AppEEARS API HTTPError without response: {e}")
                    # Re-raise immediately (don't retry)
                    raise
                
                # Handle different HTTP status codes
                if status_code == 429:
                    # Rate limiting - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        # Last attempt failed - log and re-raise
                        logger.error(f"AppEEARS API error {status_code}: Rate limited after {max_retries} attempts")
                        raise
                elif status_code == 401:
                    # Authentication failed - try to re-authenticate once
                    if attempt == 0:  # Only try re-auth on first attempt
                        logger.warning("AppEEARS authentication expired, re-authenticating...")
                        self.token = None
                        self.token_expires = None
                        self._authenticate()
                        # Retry immediately after re-authentication
                        continue
                    else:
                        # Already tried re-auth, log and re-raise
                        logger.error(f"AppEEARS API error {status_code}: Authentication failed after re-auth attempt")
                        raise
                elif status_code == 400:
                    # Bad Request - usually indicates invalid payload format
                    error_detail = error_text
                    try:
                        error_json = e.response.json()
                        error_message = error_json.get("message", error_detail)
                    except:
                        error_message = error_detail
                    
                    logger.error(f"AppEEARS API returned 400 Bad Request:\n"
                               f"  Message: {error_message}\n"
                               f"  This usually indicates invalid request payload format.\n"
                               f"  Check that task_type, task_name, and params are correctly structured.")
                    
                    # Log the payload we tried to send (for debugging)
                    try:
                        logger.debug(f"Request payload that failed: {json.dumps(task_payload, indent=2)}")
                    except Exception as log_err:
                        logger.debug(f"Request payload that failed: {task_payload} (could not serialize: {log_err})")
                    
                    raise ValueError(f"AppEEARS API rejected request (400 Bad Request): {error_message}")
                elif status_code == 404:
                    # Not Found - might indicate wrong endpoint or resource doesn't exist
                    logger.error(f"AppEEARS API returned 404 Not Found:\n"
                               f"  Endpoint: {self.API_BASE}/task\n"
                               f"  Response: {error_text}\n"
                               f"  This might indicate:\n"
                               f"  1. Invalid API endpoint\n"
                               f"  2. Authentication failure (some APIs return 404 for auth failures)\n"
                               f"  3. Invalid request format")
                    raise ValueError(f"AppEEARS API endpoint not found (404). Response: {error_text}")
                elif status_code == 500:
                    # Internal Server Error - server-side issue, log details for debugging
                    logger.error(f"AppEEARS API returned 500 Internal Server Error:\n"
                               f"  This is a server-side error from AppEEARS.\n"
                               f"  Response: {error_text}\n"
                               f"  Possible causes:\n"
                               f"  1. Invalid payload structure (check task_type, params format)\n"
                               f"  2. Invalid product/layer combination\n"
                               f"  3. Date range issues\n"
                               f"  4. Coordinate validation failed\n"
                               f"  5. Temporary server issue (retry later)")
                    # Log the payload we tried to send (for debugging)
                    try:
                        logger.debug(f"Request payload that failed: {json.dumps(task_payload, indent=2)}")
                    except Exception as log_err:
                        logger.debug(f"Request payload that failed: {task_payload} (could not serialize: {log_err})")
                    raise ValueError(f"AppEEARS API server error (500): {error_text}")
                else:
                    # Other HTTP errors - log and re-raise immediately (don't retry)
                    if status_code:
                        logger.error(f"AppEEARS API error {status_code}: {error_text}")
                    else:
                        logger.error(f"AppEEARS API error (unknown status): {error_text}")
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Connection/timeout errors - don't retry, re-raise immediately
                logger.error(f"AppEEARS API connection error: {e}")
                raise
            except Exception as e:
                # Other unexpected errors - don't retry
                logger.error(f"AppEEARS API unexpected error: {e}")
                raise
    
    def submit_batch_requests(
        self,
        points: List[Tuple[float, float, str]],
        product: str = "modis_ndvi",
        date_buffer_days: Optional[int] = None,
        max_workers: int = 3,  # Reduced from 10 to 3 to avoid rate limiting
        max_points_per_batch: Optional[int] = None  # Defaults to DEFAULT_MAX_POINTS_PER_BATCH
    ) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        Submit multiple point extraction requests using batched submissions.
        
        Groups points by overlapping date ranges and submits one task per group,
        significantly reducing the number of API calls compared to individual submissions.
        
        Args:
            points: List of (latitude, longitude, date) tuples
            product: Product ID
            date_buffer_days: Days before/after target date
                (defaults to DEFAULT_DATE_BUFFER_DAYS if not specified)
            max_workers: Maximum parallel submission threads (default: 3 to avoid rate limiting)
            max_points_per_batch: Maximum number of points per batch
                (defaults to DEFAULT_MAX_POINTS_PER_BATCH if not specified)
            
        Returns:
            Dict mapping task_id -> list of (lat, lon, date) tuples for tracking
        """
        if not points:
            return {}
        
        # Use class defaults if not specified
        if date_buffer_days is None:
            date_buffer_days = self.DEFAULT_DATE_BUFFER_DAYS
        if max_points_per_batch is None:
            max_points_per_batch = self.DEFAULT_MAX_POINTS_PER_BATCH
        
        # Group points by date range for efficient batching
        point_groups = self._group_points_by_date_range(
            points,
            date_buffer_days=date_buffer_days,
            max_points_per_batch=max_points_per_batch
        )
        
        logger.info(f"Grouped {len(points)} points into {len(point_groups)} batch(es) for submission")
        
        task_map = {}
        
        def submit_batch(group):
            """Submit a batch of points and return (task_id, group) tuple."""
            try:
                task_id = self.submit_point_request(group, product, date_buffer_days)
                return (task_id, group)
            except Exception as e:
                logger.warning(f"Failed to submit batch of {len(group)} points: {e}")
                return None
        
        # Submit batches in parallel with reduced workers to avoid rate limiting
        # Add small delay between batches to be respectful to the API
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(submit_batch, group) for group in point_groups]
            
            completed_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    task_id, group = result
                    # Only add to task_map if task_id is valid (not None)
                    if task_id is not None:
                        task_map[task_id] = group
                    else:
                        logger.warning(f"Received None task_id for batch of {len(group)} points, skipping")
                
                completed_count += 1
                # Add small delay every 10 submissions to avoid overwhelming the API
                if completed_count % 10 == 0 and completed_count < len(point_groups):
                    time.sleep(0.5)  # 500ms delay every 10 submissions
        
        total_points_submitted = sum(len(group) for group in task_map.values())
        logger.info(f"Submitted {len(task_map)} batch task(s) covering {total_points_submitted}/{len(points)} points")
        return task_map
    
    def wait_for_tasks_parallel(
        self,
        task_map: Dict[str, List[Tuple[float, float, str]]],
        max_wait_minutes: int = 30,
        poll_workers: int = 3  # Reduced from 10 to 3 to avoid rate limiting
    ) -> Dict[str, Dict]:
        """
        Wait for multiple tasks to complete, polling them in parallel.
        
        Args:
            task_map: Dict mapping task_id -> list of (lat, lon, date) tuples (batched points)
            max_wait_minutes: Maximum time to wait
            poll_workers: Number of parallel polling threads
            
        Returns:
            Dict mapping task_id -> status dict
        """
        completed_tasks = {}
        failed_tasks = {}  # Track failed tasks separately
        pending_tasks = dict(task_map)
        start_time = time.time()
        max_seconds = max_wait_minutes * 60
        
        def poll_task(task_id):
            """Poll a single task and return (task_id, status, result_type) or None if still pending.
            
            Returns:
                Tuple of (task_id, status_dict, result_type) where result_type is:
                - "completed" for done tasks
                - "failed" for failed/error/cancelled tasks
                - None if still pending
            """
            if task_id is None:
                logger.warning("Attempted to poll task with None task_id, skipping")
                return None
            try:
                status = self.check_task_status(task_id)
                task_status = status.get("status", "unknown").lower()
                
                # Completed successfully
                if task_status == "done":
                    return (task_id, status, "completed")
                
                # Failed states - treat all as terminal failures
                elif task_status in ("failed", "error", "cancelled", "invalid"):
                    error_msg = status.get("message", status.get("error", "Unknown error"))
                    logger.warning(f"Task {task_id} {task_status}: {error_msg}")
                    return (task_id, status, "failed")
                
                # Unknown status - log for debugging
                elif task_status not in ("pending", "processing", "queued"):
                    logger.warning(f"Task {task_id} has unknown status: {task_status}. Treating as pending.")
                    return None
                
                # Still pending/processing/queued
                else:
                    return None
            except Exception as e:
                logger.warning(f"Error polling task {task_id}: {e}")
                return None
        
        # Initial immediate check for fast-completing tasks
        logger.info(f"Checking {len(pending_tasks)} tasks immediately for fast completion...")
        with ThreadPoolExecutor(max_workers=poll_workers) as executor:
            futures = {executor.submit(poll_task, task_id): task_id for task_id in pending_tasks.keys()}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    task_id, status, result_type = result
                    if result_type == "completed":
                        completed_tasks[task_id] = status
                    elif result_type == "failed":
                        failed_tasks[task_id] = status
                    del pending_tasks[task_id]
        
        if not pending_tasks:
            logger.info(f"All {len(completed_tasks)} tasks completed immediately!")
            return completed_tasks
        
        # Adaptive polling: start with 60s, reduce to 10s
        initial_interval = 60
        final_interval = 10
        transition_time = 120  # 2 minutes
        
        iteration = 0
        while pending_tasks:
            elapsed = time.time() - start_time
            
            if elapsed > max_seconds:
                logger.warning(f"Timeout: {len(pending_tasks)} task(s) did not complete within {max_wait_minutes} minutes")
                # Log details about timed-out tasks
                for task_id in list(pending_tasks.keys()):
                    points_group = task_map.get(task_id, [])
                    points_count = len(points_group)
                    logger.warning(f"  - Task {task_id[:8]}... ({points_count} point(s)): Timed out after {max_wait_minutes} minutes")
                    if points_count <= 5:  # Only show point details if small number
                        for lat, lon, date_str in points_group:
                            logger.warning(f"    Point: ({lat:.4f}, {lon:.4f}) on {date_str}")
                break
            
            # Calculate adaptive interval
            if elapsed < transition_time:
                progress = elapsed / transition_time
                current_interval = initial_interval - (initial_interval - final_interval) * progress
                current_interval = max(final_interval, int(current_interval))
            else:
                current_interval = final_interval
            
            iteration += 1
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            logger.info(f"Polling iteration {iteration}: {len(pending_tasks)} pending, {len(completed_tasks)} completed "
                       f"(elapsed: {elapsed_min}m {elapsed_sec}s, interval: {current_interval}s)")
            
            # Poll all pending tasks in parallel with throttling
            with ThreadPoolExecutor(max_workers=poll_workers) as executor:
                futures = {executor.submit(poll_task, task_id): task_id for task_id in pending_tasks.keys()}
                
                completed_count = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        task_id, status, result_type = result
                        if result_type == "completed":
                            completed_tasks[task_id] = status
                        elif result_type == "failed":
                            failed_tasks[task_id] = status
                        del pending_tasks[task_id]
                    
                    completed_count += 1
                    # Add small delay every 5 polls to avoid overwhelming the API
                    if completed_count % 5 == 0 and completed_count < len(futures):
                        time.sleep(0.3)  # 300ms delay every 5 polls
            
            if not pending_tasks:
                break
            
            time.sleep(current_interval)
        
        # Log comprehensive summary
        total_tasks = len(task_map)
        completed_count = len(completed_tasks)
        failed_count = len(failed_tasks)
        timeout_count = len(pending_tasks)
        
        logger.info(f"Completed polling: {completed_count}/{total_tasks} tasks finished successfully")
        
        if failed_count > 0:
            logger.warning(f"Failed tasks: {failed_count}/{total_tasks}")
            for task_id, status in failed_tasks.items():
                error_msg = status.get("message", status.get("error", "Unknown error"))
                points_count = len(task_map.get(task_id, []))
                logger.warning(f"  - Task {task_id[:8]}... ({points_count} points): {error_msg}")
        
        if timeout_count > 0:
            logger.warning(f"Timeout: {timeout_count} task(s) did not complete within {max_wait_minutes} minutes")
            for task_id in pending_tasks.keys():
                points_count = len(task_map.get(task_id, []))
                logger.warning(f"  - Task {task_id[:8]}... ({points_count} points): Still pending after timeout")
        
        # Return both completed and failed tasks (caller can decide what to do with failed)
        # Store failed status in completed_tasks dict with a special marker
        for task_id, status in failed_tasks.items():
            completed_tasks[task_id] = status  # Include failed tasks so caller knows about them
        
        return completed_tasks
    
    def group_points_by_location(
        self,
        points: List[Tuple[float, float, str]],
        distance_threshold_km: float = 1.0
    ) -> List[List[Tuple[float, float, str]]]:
        """
        Group nearby points for potential area extraction.
        
        Args:
            points: List of (lat, lon, date) tuples
            distance_threshold_km: Maximum distance in km to group points (default: 1km)
            
        Returns:
            List of point groups, where each group contains nearby points
        """
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km."""
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c
        
        groups = []
        used = set()
        
        for i, point1 in enumerate(points):
            if i in used:
                continue
            
            group = [point1]
            used.add(i)
            
            for j, point2 in enumerate(points[i+1:], start=i+1):
                if j in used:
                    continue
                
                lat1, lon1, _ = point1
                lat2, lon2, _ = point2
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                if distance <= distance_threshold_km:
                    group.append(point2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def check_task_status(self, task_id: str) -> Dict:
        """
        Check the status of a submitted task.
        
        According to AppEEARS API docs: GET /task/{task_id}
        """
        # Retry logic for rate limiting (429 errors)
        max_retries = 5  # Increased from 3 to 5 for better resilience
        base_delay = 2.0  # Increased from 1.0 to 2.0 for longer backoff
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.API_BASE}/task/{task_id}",
                    headers=self._get_headers(),
                    timeout=30
                )
                
                # Check for rate limiting before raising
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.debug(f"Rate limited (429) checking task {task_id}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        # Last attempt failed
                        response.raise_for_status()
                
                # For other status codes, raise immediately if error
                response.raise_for_status()
                status = response.json()
                
                # Log additional details for debugging long waits
                if status.get("status") == "pending":
                    logger.debug(f"Task {task_id} still pending. Response keys: {list(status.keys())}")
                    if "estimate" in status:
                        logger.debug(f"Task estimate: {status.get('estimate')}")
                    if "retry_at" in status:
                        logger.debug(f"Retry at: {status.get('retry_at')}")
                
                return status
                
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    # Rate limiting - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.debug(f"Rate limited (429) checking task {task_id}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        # Last attempt failed - re-raise
                        raise
                else:
                    # Other HTTP errors - re-raise immediately (don't retry)
                    raise
    
    def wait_for_task(
        self,
        task_id: str,
        poll_interval: Optional[int] = None,
        max_wait_minutes: int = 10,
        adaptive_polling: bool = True
    ) -> Dict:
        """
        Wait for a task to complete, polling status periodically.
        
        Uses adaptive polling: starts with longer intervals (60s), reduces to shorter
        intervals (10s) as task approaches completion. Also checks status immediately
        after submission to catch fast-completing tasks.
        
        Args:
            task_id: Task ID to wait for
            poll_interval: Fixed seconds between status checks (if None, uses adaptive)
            max_wait_minutes: Maximum time to wait in minutes
            adaptive_polling: If True, use adaptive polling intervals (default: True)
            
        Returns:
            Final task status dict
        """
        start_time = time.time()
        max_seconds = max_wait_minutes * 60
        
        # Immediate status check to catch fast-completing tasks
        status = self.check_task_status(task_id)
        task_status = status.get("status", "unknown")
        
        if task_status == "done":
            logger.info(f"Task {task_id} completed immediately")
            return status
        elif task_status == "failed":
            error_msg = status.get("message", "Unknown error")
            raise RuntimeError(f"AppEEARS task failed: {error_msg}")
        
        # Adaptive polling: start with 60s, reduce to 10s as time progresses
        if adaptive_polling and poll_interval is None:
            # Start with 60s, reduce to 10s after 2 minutes
            initial_interval = 60
            final_interval = 10
            transition_time = 120  # 2 minutes
        else:
            initial_interval = poll_interval or 30
            final_interval = poll_interval or 30
            transition_time = 0
        
        while True:
            elapsed = time.time() - start_time
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            
            # Calculate adaptive interval
            if adaptive_polling and poll_interval is None:
                if elapsed < transition_time:
                    # Gradually reduce from initial to final interval
                    progress = elapsed / transition_time
                    current_interval = initial_interval - (initial_interval - final_interval) * progress
                    current_interval = max(final_interval, int(current_interval))
                else:
                    current_interval = final_interval
            else:
                current_interval = poll_interval or 30
            
            logger.info(f"Task {task_id} status: {task_status} (elapsed: {elapsed_min}m {elapsed_sec}s, next check in {current_interval}s)")
            
            # Log additional status info for debugging long waits
            if task_status == "pending" and elapsed > 120:  # After 2 minutes
                logger.debug(f"Task still pending after {elapsed_min}m {elapsed_sec}s. Full status: {status}")
            
            if task_status == "done":
                return status
            elif task_status == "failed":
                error_msg = status.get("message", "Unknown error")
                raise RuntimeError(f"AppEEARS task failed: {error_msg}")
            
            if elapsed > max_seconds:
                raise TimeoutError(f"Task {task_id} did not complete within {max_wait_minutes} minutes")
            
            time.sleep(current_interval)
            
            # Check status again after sleep
            status = self.check_task_status(task_id)
            task_status = status.get("status", "unknown")
    
    def _retry_request(self, request_func, max_retries: int = 5, base_delay: float = 2.0):
        """
        Retry a request function with exponential backoff for rate limiting (429) errors.
        
        Args:
            request_func: Function that makes the request and returns response
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.HTTPError: If request fails after all retries
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = request_func()
                # Check status code before calling raise_for_status
                if response.status_code == 429:
                    # Rate limited - will retry below
                    last_exception = requests.exceptions.HTTPError(
                        f"Rate limited (429) on attempt {attempt + 1}"
                    )
                elif response.status_code >= 400:
                    # Other HTTP error - raise immediately
                    response.raise_for_status()
                else:
                    # Success (2xx) - return response
                    return response
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    # Rate limited - will retry below
                    last_exception = e
                else:
                    # Other HTTP error - re-raise immediately
                    raise
            except Exception as e:
                # Other exceptions - re-raise immediately
                raise
            
            # If we got here, we need to retry (429 error)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                # Last attempt failed - raise the last exception
                if last_exception:
                    raise requests.exceptions.HTTPError(
                        f"Rate limited after {max_retries} attempts. "
                        f"AppEEARS API is experiencing high load. Please try again later."
                    ) from last_exception
                else:
                    raise requests.exceptions.HTTPError("Request failed after retries")
        
        # Should not reach here, but just in case
        raise requests.exceptions.HTTPError("Request failed after retries")
    
    def download_task_results(self, task_id: str, output_dir: Path) -> List[Path]:
        """
        Download results from a completed task.
        
        Args:
            task_id: Task ID
            output_dir: Directory to save downloaded files
            
        Returns:
            List of paths to downloaded files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get task bundle info
        status = self.check_task_status(task_id)
        if status.get("status") != "done":
            raise ValueError(f"Task {task_id} is not complete (status: {status.get('status')})")
        
        # According to AppEEARS API docs, when task is "done", use task_id directly to access bundle
        # The bundle endpoint is: GET /bundle/{task_id} (not /bundle/{bundle_id})
        # List files in bundle using task_id with retry logic
        def get_bundle_info():
            return self.session.get(
                f"{self.API_BASE}/bundle/{task_id}",
                headers=self._get_headers(),
                timeout=30
            )
        
        response = self._retry_request(get_bundle_info)
        response.raise_for_status()
        
        files_info = response.json()
        downloaded_files = []
        
        # Download each file with retry logic
        for file_info in files_info.get("files", []):
            file_id = file_info.get("file_id")
            filename = file_info.get("file_name", f"{task_id}_{file_id}.csv")
            file_path = output_dir / filename
            
            # Download file (use task_id as bundle identifier) with retry logic
            def download_file():
                return self.session.get(
                    f"{self.API_BASE}/bundle/{task_id}/{file_id}",
                    headers=self._get_headers(),
                    stream=True,
                    timeout=300
                )
            
            response = self._retry_request(download_file)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files.append(file_path)
            logger.info(f"Downloaded: {file_path}")
        
        return downloaded_files
    
    def get_ndvi_for_points(
        self,
        points: List[Tuple[float, float, str]],
        output_dir: Optional[Path] = None,
        product: str = "modis_ndvi",
        date_buffer_days: Optional[int] = None,  # Defaults to DEFAULT_DATE_BUFFER_DAYS
        max_wait_minutes: int = 30,
        use_batch: bool = True,
        max_submit_workers: int = 3,  # Reduced from 10 to 3 to avoid rate limiting
        max_poll_workers: int = 3,  # Reduced from 10 to 3 to avoid rate limiting
        max_points_per_batch: Optional[int] = None  # Defaults to DEFAULT_MAX_POINTS_PER_BATCH
    ) -> pd.DataFrame:
        """
        Complete workflow: Submit requests in batch, wait in parallel, download, and parse results.
        
        Optimized with:
        - Batched task submission (multiple coordinates per task, grouped by date range)
        - Parallel polling of multiple tasks
        - Adaptive polling intervals
        - Immediate status checks
        - Connection pooling
        - Conservative optimization defaults (10 days buffer, 200 points per batch)
        
        Args:
            points: List of (latitude, longitude, date) tuples
            output_dir: Directory for temporary downloads (optional)
            product: Product to use
            date_buffer_days: Buffer days around target date
                (defaults to DEFAULT_DATE_BUFFER_DAYS if not specified)
            max_wait_minutes: Maximum time to wait for task completion (default: 30 minutes)
            use_batch: If True, use batch submission and parallel polling (default: True)
            max_submit_workers: Max parallel threads for task submission
            max_poll_workers: Max parallel threads for polling
            max_points_per_batch: Maximum number of points per batch
                (defaults to DEFAULT_MAX_POINTS_PER_BATCH if not specified)
            
        Returns:
            DataFrame with columns: latitude, longitude, date, ndvi, qa_flags
            
        Raises:
            TimeoutError: If tasks do not complete within max_wait_minutes
        """
        # Use class defaults if not specified
        if date_buffer_days is None:
            date_buffer_days = self.DEFAULT_DATE_BUFFER_DAYS
        if max_points_per_batch is None:
            max_points_per_batch = self.DEFAULT_MAX_POINTS_PER_BATCH
        
        if output_dir is None:
            output_dir = Path("/tmp/appeears_downloads")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate cached and uncached points
        uncached_points = []
        cached_results = []
        
        logger.info(f"Processing {len(points)} points...")
        
        # Check cache for all points first
        if self.use_cache and self.cache:
            for point in points:
                lat, lon, date_str = point
                cached_ndvi = self.cache.get(lat, lon, date_str, product, date_buffer_days)
                if cached_ndvi is not None:
                    cached_results.append({
                        "latitude": lat,
                        "longitude": lon,
                        "date": date_str,
                        "ndvi": cached_ndvi,
                        "qa_flags": 0
                    })
                else:
                    uncached_points.append(point)
        else:
            uncached_points = points
        
        logger.info(f"Cache hits: {len(cached_results)}, Cache misses: {len(uncached_points)}")
        
        if not uncached_points:
            # All points were cached
            return pd.DataFrame(cached_results) if cached_results else pd.DataFrame(
                columns=["latitude", "longitude", "date", "ndvi", "qa_flags"]
            )
        
        all_results = []
        
        if use_batch and len(uncached_points) > 1:
            # Use optimized batch submission and parallel polling
            try:
                # Step 1: Submit all tasks in batches (grouped by date range)
                logger.info(f"Submitting {len(uncached_points)} points in batches...")
                task_map = self.submit_batch_requests(
                    uncached_points,
                    product=product,
                    date_buffer_days=date_buffer_days,
                    max_workers=max_submit_workers,
                    max_points_per_batch=max_points_per_batch
                )
                
                if not task_map:
                    logger.warning("No tasks were successfully submitted")
                else:
                    # Step 2: Poll all tasks in parallel
                    logger.info(f"Polling {len(task_map)} tasks in parallel...")
                    completed_tasks = self.wait_for_tasks_parallel(
                        task_map,
                        max_wait_minutes=max_wait_minutes,
                        poll_workers=max_poll_workers
                    )
                    
                    # Step 3: Download results in parallel (only for successfully completed tasks)
                    # Filter out failed tasks
                    successful_tasks = {
                        task_id: status for task_id, status in completed_tasks.items()
                        if status.get("status", "").lower() == "done"
                    }
                    failed_task_ids = set(completed_tasks.keys()) - set(successful_tasks.keys())
                    
                    if failed_task_ids:
                        failed_points_count = sum(len(task_map.get(task_id, [])) for task_id in failed_task_ids)
                        logger.warning(f"Skipping {len(failed_task_ids)} failed/stuck task(s) affecting {failed_points_count} point(s)")
                        for task_id in failed_task_ids:
                            points_group = task_map.get(task_id, [])
                            logger.warning(f"  - Task {task_id[:8]}...: {len(points_group)} point(s) will remain as placeholders")
                    
                    if successful_tasks:
                        logger.info(f"Downloading results for {len(successful_tasks)} successfully completed tasks...")
                        
                        def download_and_parse(task_id, points_group):
                            """Download and parse results for a batched task."""
                            try:
                                files = self.download_task_results(task_id, output_dir)
                                # Parse results for all points in this batch
                                batch_results = self._parse_csv_results(files, points_group, product, date_buffer_days)
                                return batch_results
                            except Exception as e:
                                logger.warning(f"Failed to download/parse task {task_id}: {e}")
                                return []
                        
                        with ThreadPoolExecutor(max_workers=max_poll_workers) as executor:
                            futures = {
                                executor.submit(download_and_parse, task_id, task_map[task_id]): task_id
                                for task_id in successful_tasks.keys()
                            }
                            
                            for future in as_completed(futures):
                                batch_results = future.result()
                                all_results.extend(batch_results)
                    else:
                        logger.warning("No successfully completed tasks to download results from")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}, falling back to sequential processing")
                use_batch = False
        
        if not use_batch or len(uncached_points) == 1:
            # Fallback to sequential processing (original method)
            for i, point in enumerate(uncached_points):
                lat, lon, date_str = point
                logger.debug(f"Processing point {i+1}/{len(uncached_points)}: ({lat:.4f}, {lon:.4f}) on {date_str}")
                
                try:
                    task_id = self.submit_point_request([point], product, date_buffer_days)
                    self.wait_for_task(task_id, max_wait_minutes=max_wait_minutes)
                    files = self.download_task_results(task_id, output_dir)
                    point_results = self._parse_csv_results(files, [point], product, date_buffer_days)
                    all_results.extend(point_results)
                except Exception as e:
                    logger.warning(f"Failed to get NDVI for point ({lat:.4f}, {lon:.4f}) on {date_str}: {e}")
                    continue
        
        # Log summary of what was retrieved
        api_results_count = len(all_results)
        cached_count = len(cached_results)
        total_requested = len(points)
        missing_count = total_requested - api_results_count - cached_count
        
        if missing_count > 0:
            logger.warning(f"NDVI retrieval summary: {api_results_count} from API, {cached_count} from cache, {missing_count} missing (will remain as placeholders)")
        else:
            logger.info(f"NDVI retrieval summary: {api_results_count} from API, {cached_count} from cache, all {total_requested} points retrieved")
        
        # Combine cached and API results
        all_results.extend(cached_results)
        
        # Match results to requested points and cache the best matches
        return self._match_results_to_points(
            points, 
            all_results,
            product=product,
            date_buffer_days=date_buffer_days,
            cache_results=True
        )
    
    def _parse_csv_results(
        self,
        files: List[Path],
        points: List[Tuple[float, float, str]],
        product: str,
        date_buffer_days: int
    ) -> List[Dict]:
        """
        Parse CSV results from downloaded files.
        
        With batching, a single CSV file may contain results for multiple points.
        This method parses all results and matches them to the requested points.
        
        Args:
            files: List of CSV file paths from AppEEARS
            points: List of (latitude, longitude, date) tuples that were requested
            product: Product ID used
            date_buffer_days: Date buffer days used
            
        Returns:
            List of result dictionaries with latitude, longitude, date, ndvi, qa_flags
        """
        all_results = []
        
        for file_path in files:
            if file_path.suffix != ".csv":
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Find NDVI column
                ndvi_col = None
                for col in df.columns:
                    if "NDVI" in col.upper() and "QUALITY" not in col.upper():
                        ndvi_col = col
                        break
                
                if ndvi_col is None:
                    logger.warning(f"No NDVI column found in {file_path}. Columns: {list(df.columns)}")
                    continue
                
                # Normalize column names
                lat_col = "Latitude" if "Latitude" in df.columns else "latitude"
                lon_col = "Longitude" if "Longitude" in df.columns else "longitude"
                date_col = "Date" if "Date" in df.columns else "date"
                
                # Find QA column
                qa_col = None
                for col in df.columns:
                    col_upper = col.upper()
                    if ("QUALITY" in col_upper or "QA" in col_upper) and \
                       "DESCRIPTION" not in col_upper and \
                       "BITMASK" not in col_upper and \
                       "MODLAND" not in col_upper:
                        qa_col = col
                        break
                
                for _, row in df.iterrows():
                    result_lat = row.get(lat_col)
                    result_lon = row.get(lon_col)
                    date_val = row.get(date_col)
                    
                    if pd.isna(result_lat) or pd.isna(result_lon) or pd.isna(date_val):
                        continue
                    
                    ndvi_raw = row.get(ndvi_col)
                    if pd.isna(ndvi_raw):
                        continue
                    
                    # Normalize NDVI
                    if ndvi_raw > 1:
                        ndvi = float(ndvi_raw) / 10000.0
                    else:
                        ndvi = float(ndvi_raw)
                    
                    if not (-1 <= ndvi <= 1):
                        # -3000 is a fill value from AppEEARS for missing/invalid data (expected)
                        if abs(ndvi - (-3000)) < 0.1 or abs(ndvi_raw - (-3000)) < 0.1:
                            logger.debug(f"Skipping fill value NDVI: {ndvi} (raw: {ndvi_raw}) - expected for missing data")
                        else:
                            logger.warning(f"Invalid NDVI value: {ndvi} (raw: {ndvi_raw})")
                        continue
                    
                    qa_value = row.get(qa_col, 0) if qa_col else 0
                    
                    result = {
                        "latitude": float(result_lat),
                        "longitude": float(result_lon),
                        "date": str(date_val),
                        "ndvi": ndvi,
                        "qa_flags": qa_value
                    }
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing {file_path}: {e}")
                continue
        
        return all_results
    
    def _match_results_to_points(
        self,
        points: List[Tuple[float, float, str]],
        all_results: List[Dict],
        product: str = "modis_ndvi",
        date_buffer_days: Optional[int] = None,
        cache_results: bool = True
    ) -> pd.DataFrame:
        """
        Match results to requested points and optionally cache them.
        
        Args:
            points: List of (latitude, longitude, date) tuples
            all_results: List of result dictionaries from parsing
            product: Product ID (for caching)
            date_buffer_days: Date buffer days (for caching)
                (defaults to DEFAULT_DATE_BUFFER_DAYS if not specified)
            cache_results: If True, cache matched results (default: True)
            
        Returns:
            DataFrame with matched results
        """
        if not all_results:
            return pd.DataFrame(columns=["latitude", "longitude", "date", "ndvi", "qa_flags"])

        results_df = pd.DataFrame(all_results)
        matched_results = []

        # MODIS pixels are ~250m, which is ~0.002-0.003 degrees
        # Use tolerance of 0.01 degrees (~1km) to handle pixel center vs input coordinate differences
        COORD_TOLERANCE = 0.01

        for lat, lon, date_str in points:
            # Try strict matching first (exact rounding to 4 decimals)
            match = results_df[
                (results_df['latitude'].round(4) == round(lat, 4)) &
                (results_df['longitude'].round(4) == round(lon, 4)) &
                (results_df['date'].astype(str).str.startswith(date_str[:10]))
            ]

            # If strict matching fails, try tolerance-based matching
            # This handles cases where AppEEARS returns MODIS pixel center coordinates
            if len(match) == 0:
                match = results_df[
                    (abs(results_df['latitude'] - lat) <= COORD_TOLERANCE) &
                    (abs(results_df['longitude'] - lon) <= COORD_TOLERANCE) &
                    (results_df['date'].astype(str).str.startswith(date_str[:10]))
                ]
                if len(match) > 0:
                    logger.debug(
                        f"Used tolerance-based matching for ({lat:.4f}, {lon:.4f}) on {date_str}: "
                        f"found {len(match)} result(s)"
                    )

            if len(match) > 0:
                match = match.copy()
                match['date_diff'] = abs(pd.to_datetime(match['date']) - pd.to_datetime(date_str))
                best_match = match.loc[match['date_diff'].idxmin()]

                ndvi = best_match['ndvi']
                qa_flags = best_match.get('qa_flags', 0)

                # Cache the best match for this point
                if cache_results and self.use_cache and self.cache and ndvi is not None:
                    try:
                        self.cache.put(
                            lat, lon, date_str, product, date_buffer_days,
                            ndvi, qa_flags
                        )
                        logger.debug(f"Cached NDVI for ({lat:.4f}, {lon:.4f}) on {date_str}: {ndvi:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to cache NDVI for ({lat:.4f}, {lon:.4f}): {e}")

                matched_results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": date_str,
                    "ndvi": ndvi,
                    "qa_flags": qa_flags
                })
            else:
                # No match found - add None entry
                logger.debug(
                    f"No NDVI match found for ({lat:.4f}, {lon:.4f}) on {date_str} "
                    f"(checked {len(results_df)} results)"
                )
                matched_results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": date_str,
                    "ndvi": None,
                    "qa_flags": None
                })

        return pd.DataFrame(matched_results)
    
    def get_ndvi_for_dates(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        product: str = "modis_ndvi",
        date_buffer_days: int = 5
    ) -> pd.DataFrame:
        """
        Get NDVI time series for a single point over a date range.
        
        Uses smart caching: checks cache for date range before making API calls.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: Product to use
            date_buffer_days: Buffer days around target date (default: 5)
            
        Returns:
            DataFrame with date and NDVI columns
        """
        # Generate daily points
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append((lat, lon, current.strftime("%Y-%m-%d")))
            current += timedelta(days=1)
        
        return self.get_ndvi_for_points(dates, product=product, date_buffer_days=date_buffer_days)
    
    def cache_date_range(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        product: str = "modis_ndvi",
        date_buffer_days: int = 5
    ) -> int:
        """
        Pre-cache NDVI values for a date range at a location.
        
        Useful for pre-loading common date ranges (e.g., all summer 2020 data).
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: Product to use
            date_buffer_days: Buffer days around target date
            
        Returns:
            Number of values cached
        """
        logger.info(f"Pre-caching NDVI for ({lat:.4f}, {lon:.4f}) from {start_date} to {end_date}")
        df = self.get_ndvi_for_dates(lat, lon, start_date, end_date, product, date_buffer_days)
        cached_count = len(df[df['ndvi'].notna()])
        logger.info(f"Cached {cached_count} NDVI values for date range")
        return cached_count

