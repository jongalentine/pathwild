"""
PRISM Climate Data Client for Historical Weather/Temperature Data

Handles downloading and extracting PRISM gridded climate data.
"""

import os
import zipfile
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import logging
import tempfile
import time
import fcntl  # For Unix file locking
try:
    import msvcrt  # For Windows file locking
    WINDOWS = True
except ImportError:
    WINDOWS = False

logger = logging.getLogger(__name__)


class PRISMRateLimitError(Exception):
    """Exception raised when PRISM rate limit is hit."""
    pass


class PRISMClient:
    """Client for downloading and extracting PRISM climate data
    
    Uses the PRISM web service (as of October 2025) which provides COG format files
    via HTTPS. The old FTP service was discontinued on September 30, 2025.
    
    Web service documentation:
    https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf
    """
    
    # PRISM web service base URL (new service as of Oct 2025)
    # Format: https://services.nacse.org/prism/data/get/{region}/{res}/{element}/{date}
    # Example: https://services.nacse.org/prism/data/get/us/4km/tmean/20240101
    BASE_URL = "https://services.nacse.org/prism/data/get"
    
    # Region: us (CONUS), hi (Hawaii - not yet), ak (Alaska - not yet), pr (Puerto Rico - not yet)
    REGION = "us"
    
    # Resolution: 4km (for daily data), 800m, 400m (400m not yet implemented)
    RESOLUTION = "4km"
    
    # Fallback to FTP if web service unavailable (for older cached data)
    FTP_FALLBACK_URL = "https://ftp.prism.oregonstate.edu/daily"
    
    VARIABLES = {
        "tmean": "Mean temperature",
        "tmin": "Minimum temperature",
        "tmax": "Maximum temperature",
        "ppt": "Precipitation"
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize PRISM client.
        
        Args:
            data_dir: Directory to cache downloaded PRISM files
        """
        if data_dir is None:
            data_dir = Path("data/prism")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each variable
        for var in self.VARIABLES.keys():
            (self.data_dir / var).mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, variable: str, date: datetime) -> Path:
        """Get the expected file path for a PRISM variable/date
        
        New web service uses COG format (delivered in ZIP), old FTP used BIL.
        """
        date_str = date.strftime("%Y%m%d")
        # Web service delivers ZIP files containing COG .tif files
        # Filename format: prism_{variable}_us_{resolution}_{date_str}.zip
        # Extracted COG: prism_{variable}_us_{resolution}_{date_str}.tif
        cog_filename = f"prism_{variable}_{self.REGION}_{self.RESOLUTION}_{date_str}.tif"
        zip_filename = f"prism_{variable}_{self.REGION}_{self.RESOLUTION}_{date_str}.zip"
        
        # Old FTP format for backward compatibility
        old_bil_filename = f"PRISM_{variable}_stable_4kmD2_{date_str}_bil.bil"
        
        cog_path = self.data_dir / variable / cog_filename
        zip_path = self.data_dir / variable / zip_filename
        bil_path = self.data_dir / variable / old_bil_filename
        
        # Return existing file if it exists (any format)
        if cog_path.exists():
            return cog_path
        if bil_path.exists():
            return bil_path
        # If ZIP exists but not extracted, extract it now
        if zip_path.exists() and zip_path.stat().st_size > 0:
            try:
                # Extract COG from ZIP
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    tif_member = None
                    for member in zip_ref.namelist():
                        if member.endswith(".tif") and not member.endswith(".aux.xml"):
                            tif_member = member
                            break
                    if tif_member:
                        zip_ref.extract(tif_member, cog_path.parent)
                        extracted_path = cog_path.parent / Path(tif_member).name
                        if extracted_path != cog_path:
                            extracted_path.rename(cog_path)
                        logger.debug(f"Extracted COG from existing ZIP: {cog_path}")
                        return cog_path
            except Exception as e:
                logger.warning(f"Failed to extract COG from ZIP {zip_path}: {e}. Will re-download.")
        
        # Default to COG for new downloads
        return cog_path
    
    def _get_rate_limit_marker_path(self, variable: str, date: datetime) -> Path:
        """Get path to rate limit marker file (tracks when we hit rate limit for a file)"""
        date_str = date.strftime("%Y%m%d")
        filename = f"prism_{variable}_{self.REGION}_{self.RESOLUTION}_{date_str}.ratelimit"
        return self.data_dir / variable / filename
    
    def _check_rate_limit_marker(self, variable: str, date: datetime) -> bool:
        """
        Check if we've hit rate limit for this file recently (within 24 hours).
        
        Returns:
            True if rate limit marker exists and is recent (< 24h old), False otherwise
        """
        marker_path = self._get_rate_limit_marker_path(variable, date)
        if not marker_path.exists():
            return False
        
        try:
            # Check marker age
            marker_age = time.time() - marker_path.stat().st_mtime
            # PRISM rate limit is 24 hours, so check if marker is less than 23 hours old
            # (give 1 hour buffer to be safe)
            if marker_age < 23 * 3600:  # 23 hours in seconds
                return True
            else:
                # Marker is stale (> 24h), remove it
                marker_path.unlink()
                return False
        except Exception as e:
            logger.debug(f"Error checking rate limit marker: {e}")
            return False
    
    def _create_rate_limit_marker(self, variable: str, date: datetime):
        """Create a rate limit marker file to prevent repeated attempts"""
        marker_path = self._get_rate_limit_marker_path(variable, date)
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(f"Rate limit hit at {datetime.now().isoformat()}\n")
            logger.debug(f"Created rate limit marker: {marker_path}")
        except Exception as e:
            logger.debug(f"Failed to create rate limit marker: {e}")
    
    def _get_web_service_url(self, variable: str, date: datetime, format: Optional[str] = None) -> str:
        """
        Get the download URL for PRISM web service.
        
        Format: https://services.nacse.org/prism/data/get/{region}/{res}/{element}/{date}<?format=...>
        
        Args:
            variable: PRISM variable (tmean, tmin, tmax, ppt)
            date: Date to download
            format: Optional format (nc, asc, bil). Default is COG (.tif in ZIP)
        """
        date_str = date.strftime("%Y%m%d")
        url = f"{self.BASE_URL}/{self.REGION}/{self.RESOLUTION}/{variable}/{date_str}"
        
        if format:
            url += f"?format={format}"
        
        return url
    
    def _get_ftp_url(self, variable: str, date: datetime) -> str:
        """Get the download URL for old FTP service (ZIP format) - deprecated"""
        year = date.year
        date_str = date.strftime("%Y%m%d")
        zip_filename = f"PRISM_{variable}_stable_4kmD2_{date_str}_bil.zip"
        return f"{self.FTP_FALLBACK_URL}/{variable}/{year}/{zip_filename}"
    
    def _acquire_file_lock(self, lock_path: Path, timeout: float = 300.0) -> bool:
        """
        Acquire an exclusive file lock to prevent concurrent downloads.
        
        Uses a simple file-based lock: creates lock file if it doesn't exist.
        If lock file exists, waits for it to be removed (another worker is downloading).
        Uses platform-specific advisory locking (fcntl on Unix, msvcrt on Windows) when available
        to provide additional protection, but the primary mechanism is file existence.
        
        Args:
            lock_path: Path to the lock file
            timeout: Maximum time to wait for lock (seconds)
            
        Returns:
            True if lock acquired, False if timeout
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to create lock file exclusively (fails if file exists)
                try:
                    # Use 'x' mode to create file exclusively (fails if exists)
                    lock_file = open(lock_path, 'x')
                    
                    # Write PID to lock file for debugging
                    lock_file.write(str(os.getpid()))
                    lock_file.flush()
                    
                    # Try to acquire advisory lock (non-blocking) for additional protection
                    # Note: On Unix, fcntl locks are released when file is closed, so we
                    # primarily rely on file existence. The advisory lock provides extra
                    # protection for processes that check locks.
                    try:
                        if WINDOWS:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (IOError, OSError):
                        # Advisory lock failed, but file was created - that's OK
                        # The file existence is the primary lock mechanism
                        pass
                    
                    # Close file - lock is held by file existence
                    lock_file.close()
                    return True
                    
                except FileExistsError:
                    # Lock file exists - another worker is downloading
                    # Check if lock file is stale (older than 10 minutes)
                    try:
                        if lock_path.exists():
                            lock_age = time.time() - lock_path.stat().st_mtime
                            if lock_age > 600:  # 10 minutes
                                # Stale lock - remove it and try again
                                logger.warning(f"Removing stale lock file: {lock_path}")
                                lock_path.unlink()
                                continue
                    except Exception:
                        pass
                
                # Lock is held - wait a bit and retry
                time.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"Error acquiring lock: {e}")
                time.sleep(0.5)
        
        # Timeout - couldn't acquire lock
        logger.warning(f"Could not acquire lock for {lock_path} within {timeout}s")
        return False
    
    def _release_file_lock(self, lock_path: Path):
        """Release a file lock."""
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception as e:
            logger.debug(f"Error releasing lock {lock_path}: {e}")
    
    def _wait_for_file(self, file_path: Path, timeout: float = 300.0) -> bool:
        """
        Wait for a file to be created by another process (e.g., after lock is released).
        
        Args:
            file_path: Path to wait for
            timeout: Maximum time to wait (seconds)
            
        Returns:
            True if file exists, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if file_path.exists():
                return True
            time.sleep(0.5)
        return False
    
    def _download_prism_file(
        self,
        variable: str,
        date: datetime,
        max_retries: int = 3,
        use_web_service: bool = True
    ) -> Optional[Path]:
        """
        Download a PRISM file if it doesn't exist locally.
        
        Uses file locking to prevent concurrent downloads by multiple workers.
        Uses the new PRISM web service (COG format) by default, with fallback
        to old FTP service for backward compatibility with cached data.
        
        PRISM files are persisted permanently - once downloaded, they are reused
        across all pipeline runs. This is especially important for historical
        training data which never changes.
        
        Args:
            variable: PRISM variable (tmean, tmin, tmax, ppt)
            date: Date to download
            max_retries: Maximum retry attempts
            use_web_service: If True, use new web service (COG), else try FTP (deprecated)
            
        Returns:
            Path to the downloaded file (COG or BIL), or None if not available
        """
        if variable not in self.VARIABLES:
            raise ValueError(f"Unknown variable: {variable}. Available: {list(self.VARIABLES.keys())}")
        
        file_path = self._get_file_path(variable, date)
        lock_path = file_path.parent / f"{file_path.name}.lock"
        
        # Return cached file if it exists (either COG or BIL)
        # This check happens first to avoid unnecessary work
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.debug(f"Using cached PRISM file: {file_path.name}")
            return file_path
        
        # Check if we've hit rate limit recently for this file
        if self._check_rate_limit_marker(variable, date):
            logger.debug(f"Rate limit marker exists for {file_path.name}, skipping download attempt")
            return None
        
        # Try to acquire lock for downloading
        lock_acquired = self._acquire_file_lock(lock_path, timeout=300.0)
        
        if not lock_acquired:
            # Another worker is downloading - wait for file to be created
            logger.debug(f"Another worker is downloading {file_path.name}, waiting...")
            if self._wait_for_file(file_path, timeout=300.0):
                return file_path
            else:
                logger.warning(f"Timeout waiting for {file_path.name} to be created by another worker")
                return None
        
        try:
            # We have the lock - proceed with download
            # Double-check file doesn't exist (another worker might have created it)
            if file_path.exists() and file_path.stat().st_size > 0:
                return file_path
            
            # Try web service first (new format - COG)
            if use_web_service:
                url = self._get_web_service_url(variable, date)
                logger.info(f"Downloading PRISM {variable} for {date.strftime('%Y-%m-%d')} from web service (COG format)...")
                try:
                    result = self._download_cog_file(url, file_path, variable, date, max_retries)
                    if result is not None:
                        return result
                except PRISMRateLimitError:
                    # Rate limit hit - create marker to prevent repeated attempts
                    self._create_rate_limit_marker(variable, date)
                    logger.warning(f"Rate limit hit for {date.strftime('%Y-%m-%d')}. "
                                 f"Created marker to prevent repeated attempts for 24h.")
                    # Don't try FTP fallback (it will also be rate-limited for same file)
                    return None
            
            # Fallback to FTP (for older data or if web service unavailable)
            # Skip if we hit rate limit (FTP will also be rate-limited for same file)
            logger.info(f"Trying FTP fallback for PRISM {variable} {date.strftime('%Y-%m-%d')}...")
            ftp_url = self._get_ftp_url(variable, date)
            bil_path = file_path.with_suffix(".bil")
            result = self._download_zip_file(ftp_url, bil_path, variable, date, max_retries)
            if result is not None:
                return result
            
            return None
            
        finally:
            # Always release lock
            self._release_file_lock(lock_path)
    
    def _download_cog_file(
        self,
        url: str,
        output_path: Path,
        variable: str,
        date: datetime,
        max_retries: int
    ) -> Optional[Path]:
        """
        Download a COG (Cloud Optimized GeoTIFF) file from web service.
        
        According to PRISM documentation, the web service always returns ZIP files
        containing COG .tif files and ancillary files. Each ZIP contains:
        - <filename>.tif (the COG raster data file)
        - <filename>.tif.aux.xml, .info.txt, .prj, .stn.csv, .stx, .xml (ancillary files)
        """
        date_str = date.strftime("%Y%m%d")
        zip_filename = f"prism_{variable}_{self.REGION}_{self.RESOLUTION}_{date_str}.zip"
        zip_path = output_path.parent / zip_filename
        
        for attempt in range(max_retries):
            try:
                # Download ZIP file from web service
                response = requests.get(url, stream=True, timeout=300, allow_redirects=True)
                
                if response.status_code == 404:
                    logger.debug(f"PRISM web service: data not found at {url}")
                    return None
                
                response.raise_for_status()
                
                # Check Content-Type header if available
                # Handle case where headers might be a Mock in tests
                content_type = response.headers.get("Content-Type", "")
                if isinstance(content_type, str):
                    content_type = content_type.lower()
                    if content_type and "html" in content_type:
                        # Server returned HTML (likely an error page)
                        logger.warning(f"PRISM web service returned HTML instead of ZIP (status {response.status_code})")
                        # Try to read a bit to see if it's an error message
                        try:
                            content_preview = response.content[:200].decode("utf-8", errors="ignore")
                            if "error" in content_preview.lower() or "not found" in content_preview.lower():
                                logger.debug(f"Error page content: {content_preview[:100]}")
                        except:
                            pass
                        return None
                
                # Save ZIP file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify it's actually a ZIP file by checking file signature
                with open(zip_path, "rb") as f:
                    first_bytes = f.read(4)
                    if not first_bytes.startswith(b"PK"):
                        # Not a ZIP file - might be an error page or rate limit message
                        # Check if it's a rate limit error (starts with "You ")
                        try:
                            with open(zip_path, "r", encoding="utf-8", errors="ignore") as text_file:
                                content_preview = text_file.read(200)
                                if content_preview.strip().startswith("You "):
                                    # Rate limit error - don't retry, raise exception to skip FTP fallback
                                    # Note: Rate limit marker will be created by caller
                                    logger.warning(f"PRISM rate limit hit for {date.strftime('%Y-%m-%d')} "
                                                 f"(file downloaded twice in 24h). Will skip future attempts for 24h.")
                                    zip_path.unlink()
                                    raise PRISMRateLimitError(f"PRISM rate limit hit for {date.strftime('%Y-%m-%d')}")
                        except PRISMRateLimitError:
                            # Re-raise rate limit errors immediately
                            raise
                        except Exception:
                            # Catch other errors when reading file (e.g., file I/O errors)
                            pass
                        
                        # Not a rate limit - might be other error page
                        logger.error(f"Downloaded file is not a ZIP (first bytes: {first_bytes.hex()}). "
                                   f"PRISM web service may have returned an error page.")
                        zip_path.unlink()
                        return None
                
                # Extract COG .tif file from ZIP
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        # Find the .tif file in the ZIP
                        tif_member = None
                        for member in zip_ref.namelist():
                            if member.endswith(".tif") and not member.endswith(".aux.xml"):
                                tif_member = member
                                break
                        
                        if tif_member is None:
                            logger.error(f"No .tif file found in PRISM ZIP: {zip_path}")
                            zip_path.unlink()
                            return None
                        
                        # Extract the .tif file
                        zip_ref.extract(tif_member, output_path.parent)
                        extracted_path = output_path.parent / Path(tif_member).name
                        
                        # Rename to standard name if needed
                        if extracted_path != output_path:
                            extracted_path.rename(output_path)
                except zipfile.BadZipFile as e:
                    logger.error(f"Invalid ZIP file downloaded from PRISM web service: {e}. "
                               f"File may be corrupted or server returned an error page.")
                    if zip_path.exists():
                        zip_path.unlink()
                    return None
                
                # Clean up ZIP file (keep extracted COG)
                zip_path.unlink()
                
                logger.info(f"Downloaded and extracted COG file: {output_path}")
                return output_path
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.debug(f"PRISM web service: data not found at {url}")
                    return None
                # Check if response content indicates rate limit
                try:
                    if hasattr(e.response, 'text') and e.response.text:
                        if e.response.text.strip().startswith("You "):
                            # Rate limit error - don't retry, raise exception to skip FTP fallback
                            # Note: Rate limit marker will be created by caller
                            logger.warning(f"PRISM rate limit hit (HTTP {e.response.status_code}). "
                                         f"Will skip future attempts for 24h for {date.strftime('%Y-%m-%d')}.")
                            raise PRISMRateLimitError(f"PRISM rate limit hit for {date.strftime('%Y-%m-%d')}")
                except PRISMRateLimitError:
                    raise  # Re-raise rate limit exception
                except Exception:
                    pass
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Web service download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Web service download failed after {max_retries} attempts: {e}")
                    return None
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Web service download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Web service download failed after {max_retries} attempts: {e}")
                    return None
            except PRISMRateLimitError:
                # Rate limit errors should propagate immediately (no retries)
                raise
            except Exception as e:
                logger.error(f"Error processing PRISM file: {e}")
                if zip_path.exists():
                    zip_path.unlink()
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def _download_zip_file(
        self,
        url: str,
        bil_path: Path,
        variable: str,
        date: datetime,
        max_retries: int
    ) -> Optional[Path]:
        """Download a ZIP file from FTP (old format) and extract BIL"""
        zip_path = bil_path.with_suffix(".zip")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=300)
                
                if response.status_code == 404:
                    logger.debug(f"PRISM FTP: data not found at {url}")
                    return None
                
                response.raise_for_status()
                
                # Save ZIP file
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract BIL file from ZIP
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        if member.endswith(".bil"):
                            zip_ref.extract(member, bil_path.parent)
                            extracted_path = bil_path.parent / Path(member).name
                            if extracted_path != bil_path:
                                extracted_path.rename(bil_path)
                            break
                
                # Clean up ZIP file
                zip_path.unlink()
                
                logger.info(f"Downloaded and extracted BIL: {bil_path}")
                return bil_path
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    return None
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"FTP download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"FTP download failed after {max_retries} attempts: {e}")
                    return None
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"FTP download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"FTP download failed after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def extract_value(
        self,
        variable: str,
        lat: float,
        lon: float,
        date: datetime
    ) -> Optional[float]:
        """
        Extract PRISM value at a specific location and date.
        
        Args:
            variable: PRISM variable (tmean, tmin, tmax, ppt)
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            Extracted value (in original units) or None if not available
        """
        # Download file if needed
        file_path = self._download_prism_file(variable, date)
        
        # If download returned None (404/not available), return None
        if file_path is None:
            return None
        
        # Extract value at point (works for both COG and BIL formats)
        point = Point(lon, lat)
        
        try:
            with rasterio.open(file_path) as src:
                # Check if point is within bounds
                if not src.bounds.left <= lon <= src.bounds.right or \
                   not src.bounds.bottom <= lat <= src.bounds.top:
                    logger.warning(f"Point ({lat}, {lon}) outside PRISM bounds for {date}")
                    return None
                
                # Use rasterio's sample method for more reliable point sampling
                # This works better for both COG and BIL formats
                try:
                    sample_iter = src.sample([(lon, lat)])
                    value = float(next(sample_iter)[0])
                except (StopIteration, IndexError):
                    # Fallback to mask method if sample doesn't work
                    data, _ = mask(src, [point], crop=True)
                    value = float(data[0, 0, 0])
                
                # Check for nodata
                if value < -9000:  # PRISM nodata is typically -9999
                    return None
                
                return value
                
        except Exception as e:
            logger.error(f"Error extracting PRISM value: {e}")
            return None
    
    def get_temperature(
        self,
        lat: float,
        lon: float,
        date: datetime
    ) -> Dict[str, Optional[float]]:
        """
        Get temperature data (mean, min, max) for a location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            Dictionary with temp_mean_c, temp_min_c, temp_max_c (in °C)
        """
        tmean = self.extract_value("tmean", lat, lon, date)
        tmin = self.extract_value("tmin", lat, lon, date)
        tmax = self.extract_value("tmax", lat, lon, date)
        
        # PRISM stores temperature as °C × 100, convert to °C
        result = {}
        if tmean is not None:
            result["temp_mean_c"] = tmean / 100.0
        else:
            result["temp_mean_c"] = None
        
        if tmin is not None:
            result["temp_min_c"] = tmin / 100.0
        else:
            result["temp_min_c"] = None
        
        if tmax is not None:
            result["temp_max_c"] = tmax / 100.0
        else:
            result["temp_max_c"] = None
        
        return result
    
    def get_precipitation(
        self,
        lat: float,
        lon: float,
        date: datetime
    ) -> Optional[float]:
        """
        Get precipitation for a location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            Precipitation in mm (PRISM units), or None
        """
        ppt = self.extract_value("ppt", lat, lon, date)
        
        if ppt is None:
            return None
        
        # PRISM stores precipitation as mm × 100, convert to mm
        return ppt / 100.0
    
    def get_weather_for_date_range(
        self,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get weather data for a location over a date range.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dictionaries with date, temperature, and precipitation data
        """
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            temp_data = self.get_temperature(lat, lon, current_date)
            ppt_mm = self.get_precipitation(lat, lon, current_date)
            
            # Convert to °F
            result = {
                "date": current_date.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,
                "temp_mean_f": None,
                "temp_min_f": None,
                "temp_max_f": None,
                "precipitation_mm": ppt_mm,
                "precipitation_inches": None
            }
            
            if temp_data["temp_mean_c"] is not None:
                result["temp_mean_f"] = (temp_data["temp_mean_c"] * 9/5) + 32
            if temp_data["temp_min_c"] is not None:
                result["temp_min_f"] = (temp_data["temp_min_c"] * 9/5) + 32
            if temp_data["temp_max_c"] is not None:
                result["temp_max_f"] = (temp_data["temp_max_c"] * 9/5) + 32
            
            if ppt_mm is not None:
                result["precipitation_inches"] = ppt_mm / 25.4
            
            results.append(result)
            current_date += timedelta(days=1)
        
        return results

