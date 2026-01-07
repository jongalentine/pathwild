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

logger = logging.getLogger(__name__)


class AppEEARSClient:
    """Client for NASA AppEEARS API to retrieve NDVI data"""
    
    API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
    
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
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize AppEEARS client.
        
        Args:
            username: AppEEARS username (or from APPEEARS_USERNAME env var)
            password: AppEEARS password (or from APPEEARS_PASSWORD env var)
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
        self._authenticate()
    
    def _authenticate(self) -> str:
        """Authenticate with AppEEARS API and get bearer token"""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token
        
        try:
            response = requests.post(
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
        # For MODIS vegetation index products, layer is "250m_16_days_NDVI"
        if "MOD13Q1" in product_id or "MYD13Q1" in product_id or "MxD13Q1" in product_id:
            return "250m_16_days_NDVI"
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
            response = requests.get(
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
    
    def submit_point_request(
        self,
        points: List[Tuple[float, float, str]],  # List of (lat, lon, date) tuples
        product: str = "modis_ndvi",  # Default to MODIS which is more widely available
        date_buffer_days: int = 7
    ) -> str:
        """
        Submit a point extraction request to AppEEARS.
        
        Args:
            points: List of (latitude, longitude, date) tuples
                   Date format: 'YYYY-MM-DD'
            product: Product ID (default: Landsat NDVI)
            date_buffer_days: Days before/after target date to search for cloud-free images
            
        Returns:
            Task ID for tracking the request
        """
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
        
        # Group points by date range
        tasks = []
        for lat, lon, date_str in points:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            # AppEEARS API requires dates in MM-DD-YYYY format
            start_date = (target_date - timedelta(days=date_buffer_days)).strftime("%m-%d-%Y")
            end_date = (target_date + timedelta(days=date_buffer_days)).strftime("%m-%d-%Y")
            
            task = {
                "task_type": "point",
                "task_name": f"{task_name}_{len(tasks)}",
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
                    "coordinates": [
                        {
                            "latitude": lat,
                            "longitude": lon
                        }
                    ]
                }
            }
            tasks.append(task)
        
        # Submit task (AppEEARS accepts batch requests)
        # According to AppEEARS API documentation: POST /task endpoint
        # Single tasks should be sent as objects (not wrapped in arrays)
        # Multiple tasks should be sent as arrays
        if len(tasks) == 1:
            # Single task: send task object directly (not wrapped)
            task_payload = tasks[0]
        else:
            # Multiple tasks: send as array
            task_payload = tasks
        
        try:
            # Log request details for debugging (without sensitive data)
            logger.debug(f"Submitting AppEEARS task to {self.API_BASE}/task")
            logger.debug(f"Payload structure: task_type={task_payload.get('task_type') if isinstance(task_payload, dict) else 'N/A (array)'}")
            if isinstance(task_payload, dict):
                logger.debug(f"Payload: task_name={task_payload.get('task_name')}, product={task_payload.get('params', {}).get('layers', [{}])[0].get('product', 'N/A')}")
            
            response = requests.post(
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
            
            response.raise_for_status()
            
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
            # Log detailed error information
            error_text = e.response.text if e.response else "No response"
            logger.error(f"AppEEARS API error {e.response.status_code}: {error_text}")
            
            if e.response.status_code == 401:
                # Authentication failed - try to re-authenticate
                logger.warning("AppEEARS authentication expired, re-authenticating...")
                self.token = None
                self.token_expires = None
                self._authenticate()
                # Retry once after re-authentication (use same format as original request)
                if len(tasks) == 1:
                    retry_payload = tasks[0]  # Single task as object
                else:
                    retry_payload = tasks  # Multiple tasks as array
                response = requests.post(
                    f"{self.API_BASE}/task",
                    headers=self._get_headers(),
                    json=retry_payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                task_id = result.get("task_id") or (result.get("task", {}).get("task_id") if isinstance(result.get("task"), dict) else None)
                if not task_id:
                    raise ValueError(f"Could not extract task_id from AppEEARS response: {result}")
                logger.info(f"Submitted AppEEARS task (after re-auth): {task_id}")
                return task_id
            elif e.response.status_code == 400:
                # Bad Request - usually indicates invalid payload format
                error_detail = e.response.text
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
            elif e.response.status_code == 404:
                # Not Found - might indicate wrong endpoint or resource doesn't exist
                error_detail = e.response.text
                logger.error(f"AppEEARS API returned 404 Not Found:\n"
                           f"  Endpoint: {self.API_BASE}/task\n"
                           f"  Response: {error_detail}\n"
                           f"  This might indicate:\n"
                           f"  1. Invalid API endpoint\n"
                           f"  2. Authentication failure (some APIs return 404 for auth failures)\n"
                           f"  3. Invalid request format")
                raise ValueError(f"AppEEARS API endpoint not found (404). Response: {error_detail}")
            elif e.response.status_code == 500:
                # Internal Server Error - server-side issue, log details for debugging
                error_detail = e.response.text
                logger.error(f"AppEEARS API returned 500 Internal Server Error:\n"
                           f"  This is a server-side error from AppEEARS.\n"
                           f"  Response: {error_detail}\n"
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
                raise ValueError(f"AppEEARS API server error (500): {error_detail}")
            else:
                raise
    
    def check_task_status(self, task_id: str) -> Dict:
        """
        Check the status of a submitted task.
        
        According to AppEEARS API docs: GET /task/{task_id}
        """
        response = requests.get(
            f"{self.API_BASE}/task/{task_id}",
            headers=self._get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(
        self,
        task_id: str,
        poll_interval: int = 30,
        max_wait_minutes: int = 10
    ) -> Dict:
        """
        Wait for a task to complete, polling status periodically.
        
        Args:
            task_id: Task ID to wait for
            poll_interval: Seconds between status checks
            max_wait_minutes: Maximum time to wait in minutes
            
        Returns:
            Final task status dict
        """
        start_time = time.time()
        max_seconds = max_wait_minutes * 60
        
        while True:
            status = self.check_task_status(task_id)
            task_status = status.get("status", "unknown")
            
            logger.info(f"Task {task_id} status: {task_status}")
            
            if task_status == "done":
                return status
            elif task_status == "failed":
                error_msg = status.get("message", "Unknown error")
                raise RuntimeError(f"AppEEARS task failed: {error_msg}")
            
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                raise TimeoutError(f"Task {task_id} did not complete within {max_wait_minutes} minutes")
            
            time.sleep(poll_interval)
    
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
        
        bundle_id = status.get("bundle_id")
        if not bundle_id:
            raise ValueError(f"No bundle_id found for task {task_id}")
        
        # List files in bundle
        response = requests.get(
            f"{self.API_BASE}/bundle/{bundle_id}",
            headers=self._get_headers(),
            timeout=30
        )
        response.raise_for_status()
        
        files_info = response.json()
        downloaded_files = []
        
        # Download each file
        for file_info in files_info.get("files", []):
            file_id = file_info.get("file_id")
            filename = file_info.get("file_name", f"{task_id}_{file_id}.csv")
            file_path = output_dir / filename
            
            # Download file
            response = requests.get(
                f"{self.API_BASE}/bundle/{bundle_id}/{file_id}",
                headers=self._get_headers(),
                stream=True,
                timeout=300
            )
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
        date_buffer_days: int = 7,
        max_wait_minutes: int = 10
    ) -> pd.DataFrame:
        """
        Complete workflow: Submit request, wait for completion, download, and parse results.
        
        Args:
            points: List of (latitude, longitude, date) tuples
            output_dir: Directory for temporary downloads (optional)
            product: Product to use
            date_buffer_days: Buffer days around target date
            max_wait_minutes: Maximum time to wait for task completion (default: 10 minutes)
            
        Returns:
            DataFrame with columns: latitude, longitude, date, ndvi, qa_flags
            
        Raises:
            TimeoutError: If task does not complete within max_wait_minutes
        """
        # Submit request
        task_id = self.submit_point_request(points, product, date_buffer_days)
        
        # Wait for completion
        logger.info(f"Waiting for task {task_id} to complete (max {max_wait_minutes} minutes)...")
        self.wait_for_task(task_id, max_wait_minutes=max_wait_minutes)
        
        # Download results
        if output_dir is None:
            output_dir = Path("/tmp/appeears_downloads")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = self.download_task_results(task_id, output_dir)
        
        # Parse CSV results
        results = []
        for file_path in files:
            if file_path.suffix != ".csv":
                continue
            
            df = pd.read_csv(file_path)
            
            # AppEEARS CSV format: Date, Latitude, Longitude, Product_Layer, Value, QA
            for _, row in df.iterrows():
                # Filter by QA flags (if available)
                qa_value = row.get("QA", 0) if "QA" in row else 0
                
                # Extract NDVI value (typically in 0-1 range, sometimes scaled to 0-10000)
                ndvi_raw = row.get("NDVI", row.get("Value", None))
                if pd.isna(ndvi_raw):
                    continue
                
                # Normalize NDVI to 0-1 range if needed
                if ndvi_raw > 1:
                    ndvi = ndvi_raw / 10000.0
                else:
                    ndvi = ndvi_raw
                
                # Validate NDVI range
                if not (-1 <= ndvi <= 1):
                    logger.warning(f"Invalid NDVI value: {ndvi} (raw: {ndvi_raw})")
                    continue
                
                results.append({
                    "latitude": row.get("Latitude"),
                    "longitude": row.get("Longitude"),
                    "date": row.get("Date"),
                    "ndvi": ndvi,
                    "qa_flags": qa_value
                })
        
        result_df = pd.DataFrame(results)
        
        # Merge back with original points to ensure all are included
        points_df = pd.DataFrame(points, columns=["latitude", "longitude", "date"])
        result_df = points_df.merge(
            result_df,
            on=["latitude", "longitude", "date"],
            how="left"
        )
        
        return result_df
    
    def get_ndvi_for_dates(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        product: str = "landsat_ndvi"
    ) -> pd.DataFrame:
        """
        Get NDVI time series for a single point over a date range.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: Product to use
            
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
        
        return self.get_ndvi_for_points(dates, product=product, date_buffer_days=7)

