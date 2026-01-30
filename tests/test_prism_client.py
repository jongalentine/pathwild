"""
Tests for PRISM client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import zipfile
import tempfile
import numpy as np
import requests
import os

# Mock rasterio before importing PRISMClient to avoid import errors in sandbox
import sys
from types import ModuleType

# Create a context manager-compatible mock for rasterio.open
mock_rasterio_open = MagicMock()
mock_rasterio_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
mock_rasterio_open.return_value.__exit__ = MagicMock(return_value=False)

# Create a mock rasterio module
mock_rasterio = ModuleType('rasterio')
mock_rasterio.open = mock_rasterio_open
sys.modules['rasterio'] = mock_rasterio

mock_rasterio_mask = ModuleType('rasterio.mask')
mock_rasterio_mask.mask = Mock()
sys.modules['rasterio.mask'] = mock_rasterio_mask

from src.data.prism_client import PRISMClient, PRISMRateLimitError


class TestPRISMClient:
    """Test PRISM client for historical weather data."""
    
    @pytest.mark.unit
    def test_init(self, tmp_path):
        """Test PRISM client initialization."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        assert client.data_dir == prism_dir
        assert (prism_dir / "tmean").exists()
        assert (prism_dir / "tmin").exists()
        assert (prism_dir / "tmax").exists()
        assert (prism_dir / "ppt").exists()
    
    @pytest.mark.unit
    def test_get_file_path(self, tmp_path):
        """Test PRISM file path generation (new COG format)."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        file_path = client._get_file_path("tmean", date)
        # New web service format: prism_{variable}_us_4km_{date}.tif
        expected = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        assert file_path == expected
    
    @pytest.mark.unit
    def test_get_file_path_with_existing_bil(self, tmp_path):
        """Test PRISM file path generation falls back to BIL if exists."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create old BIL file
        old_bil_path = prism_dir / "tmean" / "PRISM_tmean_stable_4kmD2_20240615_bil.bil"
        old_bil_path.parent.mkdir(parents=True, exist_ok=True)
        old_bil_path.touch()
        
        file_path = client._get_file_path("tmean", date)
        # Should return existing BIL file
        assert file_path == old_bil_path
    
    @pytest.mark.unit
    def test_get_web_service_url(self, tmp_path):
        """Test PRISM web service URL generation."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        url = client._get_web_service_url("tmean", date)
        expected = "https://services.nacse.org/prism/data/get/us/4km/tmean/20240615"
        assert url == expected
    
    @pytest.mark.unit
    def test_get_web_service_url_with_format(self, tmp_path):
        """Test PRISM web service URL generation with format parameter."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        url = client._get_web_service_url("tmean", date, format="bil")
        expected = "https://services.nacse.org/prism/data/get/us/4km/tmean/20240615?format=bil"
        assert url == expected
    
    @pytest.mark.unit
    def test_get_ftp_url(self, tmp_path):
        """Test PRISM FTP URL generation (fallback)."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        url = client._get_ftp_url("tmean", date)
        assert "2024" in url
        assert "tmean" in url
        assert "20240615" in url
        assert "ftp.prism.oregonstate.edu" in url
    
    @pytest.mark.unit
    @patch('src.data.prism_client.rasterio.open')
    def test_extract_value_with_existing_file(self, mock_rasterio, tmp_path):
        """Test extracting a PRISM value from existing COG file."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create existing COG file with content (not empty)
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        cog_path.write_bytes(b"dummy tif content")  # Write actual content, not empty
        
        # Mock rasterio dataset
        mock_dataset = MagicMock()
        mock_dataset.bounds.left = -110.0
        mock_dataset.bounds.right = -105.0
        mock_dataset.bounds.bottom = 41.0
        mock_dataset.bounds.top = 45.0
        mock_dataset.sample.return_value = iter([[2500.0]])  # Returns iterator
        mock_rasterio.return_value.__enter__.return_value = mock_dataset
        
        value = client.extract_value("tmean", 44.0, -107.0, date)
        
        # Should return value in original units (×100)
        assert value == 2500.0
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_prism_file')
    @patch('src.data.prism_client.rasterio.open')
    def test_extract_value_downloads_file(self, mock_rasterio, mock_download, tmp_path):
        """Test extracting a PRISM value when file needs to be downloaded."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock download to return file path
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        # Create the file so the existence check passes
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        cog_path.write_bytes(b'dummy tif content')
        mock_download.return_value = cog_path
        
        # Mock rasterio dataset
        mock_dataset = MagicMock()
        mock_dataset.bounds.left = -110.0
        mock_dataset.bounds.right = -105.0
        mock_dataset.bounds.bottom = 41.0
        mock_dataset.bounds.top = 45.0
        mock_dataset.sample.return_value = iter([[2500.0]])
        mock_rasterio.return_value.__enter__.return_value = mock_dataset
        
        value = client.extract_value("tmean", 44.0, -107.0, date)
        
        assert value == 2500.0
        mock_download.assert_called_once_with("tmean", date)
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    @patch('src.data.prism_client.zipfile.ZipFile')
    def test_download_cog_file(self, mock_zipfile, mock_get, tmp_path):
        """Test downloading COG file from web service."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock ZIP file containing COG .tif
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = [
            "prism_tmean_us_4km_20240615.tif",
            "prism_tmean_us_4km_20240615.tif.aux.xml",
            "prism_tmean_us_4km_20240615.prj"
        ]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Mock download response
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"PK\x03\x04fake zip data"]  # Valid ZIP signature
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/zip"}  # Properly mock headers
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        assert result == output_path
        mock_get.assert_called_once()
        mock_zipfile.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    def test_download_cog_file_404(self, mock_get, tmp_path):
        """Test COG download handles 404 errors."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        assert result is None
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    @patch('src.data.prism_client.zipfile.ZipFile')
    def test_download_cog_file_handles_html_error_page(self, mock_zipfile, mock_get, tmp_path):
        """Test COG download detects and handles HTML error pages."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock response with HTML Content-Type (error page)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.content = b"<html><body>Error: You have exceeded download limit</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        # Should return None when HTML error page is detected
        assert result is None
        mock_get.assert_called_once()
        # Should not attempt to extract ZIP
        mock_zipfile.assert_not_called()
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    @patch('src.data.prism_client.zipfile.ZipFile')
    def test_download_cog_file_handles_mock_headers_gracefully(self, mock_zipfile, mock_get, tmp_path):
        """Test COG download handles Mock headers gracefully (defensive code for tests)."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock ZIP file
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["prism_tmean_us_4km_20240615.tif"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Mock response with Mock headers (simulating incomplete test mocks)
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"PK\x03\x04fake zip data"]
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        # Headers is a Mock that returns Mock when .get() is called
        mock_headers = Mock()
        mock_headers.get.return_value = Mock()  # Returns Mock instead of string
        mock_response.headers = mock_headers
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        # Should not crash when headers.get() returns a Mock
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        # Should still work (defensive code skips HTML check when content_type is not a string)
        assert result == output_path
        mock_get.assert_called_once()
        mock_zipfile.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    @patch('src.data.prism_client.zipfile.ZipFile')
    def test_download_cog_file_handles_missing_content_type(self, mock_zipfile, mock_get, tmp_path):
        """Test COG download handles missing Content-Type header."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock ZIP file
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["prism_tmean_us_4km_20240615.tif"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Mock response without Content-Type header
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"PK\x03\x04fake zip data"]
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}  # Empty headers
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        # Should work fine without Content-Type header
        assert result == output_path
        mock_get.assert_called_once()
        mock_zipfile.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    @patch('src.data.prism_client.zipfile.ZipFile')
    def test_download_cog_file_handles_non_string_content_type(self, mock_zipfile, mock_get, tmp_path):
        """Test COG download handles non-string Content-Type values gracefully."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock ZIP file
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["prism_tmean_us_4km_20240615.tif"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Mock response with non-string Content-Type (edge case)
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"PK\x03\x04fake zip data"]
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": 12345}  # Non-string value
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        # Should not crash when Content-Type is not a string
        result = client._download_cog_file(url, output_path, "tmean", date, max_retries=1)
        
        # Should still work (defensive code skips HTML check when content_type is not a string)
        assert result == output_path
        mock_get.assert_called_once()
        mock_zipfile.assert_called_once()
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_get_file_path')
    def test_get_temperature(self, mock_get_file_path, tmp_path):
        """Test getting temperature data."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        
        # Mock _get_file_path to return a non-existent .bil path so magnitude check is used
        # This ensures the 2500.0 value gets divided by 100 to become 25.0
        bil_path = prism_dir / "tmean" / "PRISM_tmean_stable_4kmD2_20240615_bil.bil"
        mock_get_file_path.return_value = bil_path
        
        # Mock extract_value for each variable
        client.extract_value = Mock(side_effect=[
            2500.0,  # tmean (25.0°C)
            2000.0,  # tmin (20.0°C)
            3000.0   # tmax (30.0°C)
        ])
        
        result = client.get_temperature(44.0, -107.0, datetime(2024, 6, 15))
        
        assert result["temp_mean_c"] == 25.0
        assert result["temp_min_c"] == 20.0
        assert result["temp_max_c"] == 30.0
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_get_file_path')
    def test_get_temperature_with_none_values(self, mock_get_file_path, tmp_path):
        """Test getting temperature data when some values are None."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        
        # Mock _get_file_path to return a non-existent .bil path so magnitude check is used
        bil_path = prism_dir / "tmean" / "PRISM_tmean_stable_4kmD2_20240615_bil.bil"
        mock_get_file_path.return_value = bil_path
        
        # Mock extract_value returning None for some variables
        client.extract_value = Mock(side_effect=[
            2500.0,  # tmean (25.0°C)
            None,    # tmin (not available)
            3000.0   # tmax (30.0°C)
        ])
        
        result = client.get_temperature(44.0, -107.0, datetime(2024, 6, 15))
        
        assert result["temp_mean_c"] == 25.0
        assert result["temp_min_c"] is None
        assert result["temp_max_c"] == 30.0
    
    @pytest.mark.unit
    def test_get_precipitation(self, tmp_path):
        """Test getting precipitation data."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        
        # Mock extract_value returning precipitation in mm × 100
        client.extract_value = Mock(return_value=5000.0)  # 50.0 mm
        
        result = client.get_precipitation(44.0, -107.0, datetime(2024, 6, 15))
        
        assert result == 50.0  # Should be converted from mm × 100 to mm
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_download_zip_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_web_service_first(self, mock_release, mock_acquire, mock_ftp, mock_web, tmp_path):
        """Test that web service is tried before FTP fallback."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock lock acquisition
        mock_acquire.return_value = True
        
        # Web service succeeds
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        mock_web.return_value = cog_path
        mock_ftp.return_value = None
        
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        assert result == cog_path
        mock_web.assert_called_once()
        mock_ftp.assert_not_called()
        mock_release.assert_called_once()
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_download_zip_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_ftp_fallback(self, mock_release, mock_acquire, mock_ftp, mock_web, tmp_path):
        """Test that FTP fallback is used when web service fails."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock lock acquisition
        mock_acquire.return_value = True
        
        # Web service fails (not rate limit), FTP succeeds
        mock_web.return_value = None
        bil_path = prism_dir / "tmean" / "PRISM_tmean_stable_4kmD2_20240615_bil.bil"
        mock_ftp.return_value = bil_path
        
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        assert result == bil_path
        mock_web.assert_called_once()
        mock_ftp.assert_called_once()
        mock_release.assert_called_once()
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_wait_for_file')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_uses_file_locking(self, mock_release, mock_wait, mock_acquire, mock_download, tmp_path):
        """Test that file locking is used to prevent concurrent downloads."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock successful lock acquisition and download
        mock_acquire.return_value = True
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        mock_download.return_value = cog_path
        
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        assert result == cog_path
        mock_acquire.assert_called_once()
        mock_download.assert_called_once()
        mock_release.assert_called_once()
        mock_wait.assert_not_called()  # Should not wait if lock acquired
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_wait_for_file')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_waits_when_lock_held(self, mock_release, mock_wait, mock_acquire, mock_download, tmp_path):
        """Test that download waits for file when another worker holds the lock."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mock lock not acquired (another worker is downloading)
        mock_acquire.return_value = False
        # Mock wait_for_file to simulate file being created by another worker
        # We'll create the file when wait_for_file is called
        def create_file_on_wait(file_path, timeout):
            # Simulate another worker creating the file
            file_path.touch()
            return True
        mock_wait.side_effect = create_file_on_wait
        
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        assert result == cog_path
        assert cog_path.exists()
        mock_acquire.assert_called_once()
        mock_wait.assert_called_once()
        mock_download.assert_not_called()  # Should not download if file exists after waiting
        mock_release.assert_not_called()  # Should not release lock if not acquired
    
    @pytest.mark.unit
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_skips_ftp_on_rate_limit(self, mock_release, mock_acquire, mock_download, tmp_path):
        """Test that FTP fallback is skipped when rate limit is hit."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock lock acquisition
        mock_acquire.return_value = True
        
        # Mock rate limit exception from web service
        from src.data.prism_client import PRISMRateLimitError
        mock_download.side_effect = PRISMRateLimitError("PRISM rate limit hit")
        
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        # Should return None immediately without trying FTP
        assert result is None
        mock_acquire.assert_called_once()
        mock_download.assert_called_once()
        mock_release.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    def test_download_cog_file_detects_rate_limit_in_file(self, mock_get, tmp_path):
        """Test that rate limit errors in downloaded file are detected and raise exception."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock response that returns rate limit message (not a ZIP)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.raise_for_status = Mock()
        # Rate limit message starts with "You "
        mock_response.iter_content.return_value = [b"You have exceeded the download limit for this file"]
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        # Should raise PRISMRateLimitError immediately without retrying
        with pytest.raises(PRISMRateLimitError):
            client._download_cog_file(url, output_path, "tmean", date, max_retries=3)
        
        # Should only try once (no retries for rate limit)
        assert mock_get.call_count == 1
    
    @pytest.mark.unit
    @patch('src.data.prism_client.requests.get')
    def test_download_cog_file_detects_rate_limit_in_http_error(self, mock_get, tmp_path):
        """Test that rate limit errors in HTTP error response are detected and raise exception."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock HTTPError with rate limit message in response text
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "You have exceeded the download limit for this file"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        url = client._get_web_service_url("tmean", date)
        output_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        
        # Should raise PRISMRateLimitError immediately without retrying
        with pytest.raises(PRISMRateLimitError):
            client._download_cog_file(url, output_path, "tmean", date, max_retries=3)
        
        # Should only try once (no retries for rate limit)
        assert mock_get.call_count == 1
    
    @pytest.mark.unit
    def test_acquire_file_lock_creates_lock_file(self, tmp_path):
        """Test that file lock is created when acquired."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        lock_path = tmp_path / "test.lock"
        
        # Mock fcntl to succeed (Unix)
        with patch('src.data.prism_client.fcntl.flock') as mock_flock:
            mock_flock.return_value = None
            with patch('src.data.prism_client.WINDOWS', False):
                result = client._acquire_file_lock(lock_path, timeout=1.0)
                
                # Should succeed (lock acquired)
                assert result is True
    
    @pytest.mark.unit
    def test_release_file_lock_removes_lock_file(self, tmp_path):
        """Test that lock file is removed when released."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        lock_path = tmp_path / "test.lock"
        
        # Create lock file
        lock_path.touch()
        assert lock_path.exists()
        
        # Release lock
        client._release_file_lock(lock_path)
        
        # Lock file should be removed
        assert not lock_path.exists()
    
    @pytest.mark.unit
    def test_wait_for_file_waits_until_file_exists(self, tmp_path):
        """Test that wait_for_file waits until file is created."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        file_path = tmp_path / "test.tif"
        
        # File doesn't exist initially
        assert not file_path.exists()
        
        # Create file after a short delay (simulate another worker)
        import threading
        def create_file():
            import time
            time.sleep(0.1)
            file_path.touch()
        
        thread = threading.Thread(target=create_file)
        thread.start()
        
        # Wait for file (should succeed)
        result = client._wait_for_file(file_path, timeout=1.0)
        
        thread.join()
        
        assert result is True
        assert file_path.exists()
    
    @pytest.mark.unit
    def test_wait_for_file_times_out_if_file_not_created(self, tmp_path):
        """Test that wait_for_file times out if file is never created."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        file_path = tmp_path / "test.tif"
        
        # File doesn't exist and won't be created
        assert not file_path.exists()
        
        # Wait for file (should timeout)
        result = client._wait_for_file(file_path, timeout=0.1)
        
        assert result is False
        assert not file_path.exists()
    
    # ============================================
    # Tests for Rate Limit Marker Functionality
    # ============================================
    
    @pytest.mark.unit
    def test_get_rate_limit_marker_path(self, tmp_path):
        """Test rate limit marker path generation."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        expected = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.ratelimit"
        
        assert marker_path == expected
    
    @pytest.mark.unit
    def test_check_rate_limit_marker_no_marker(self, tmp_path):
        """Test rate limit marker check when marker doesn't exist."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # No marker exists
        result = client._check_rate_limit_marker("tmean", date)
        
        assert result is False
    
    @pytest.mark.unit
    def test_check_rate_limit_marker_recent_marker(self, tmp_path):
        """Test rate limit marker check when marker is recent (< 23h old)."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create recent marker file
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        
        # Check marker (should be recent)
        result = client._check_rate_limit_marker("tmean", date)
        
        assert result is True
        assert marker_path.exists()
    
    @pytest.mark.unit
    def test_check_rate_limit_marker_stale_marker(self, tmp_path):
        """Test rate limit marker check when marker is stale (> 24h old)."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create stale marker file (older than 24h)
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        
        # Make file appear old by setting mtime to 25 hours ago
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(marker_path, (old_time, old_time))
        
        # Check marker (should be stale and removed)
        result = client._check_rate_limit_marker("tmean", date)
        
        assert result is False
        assert not marker_path.exists()  # Should be removed
    
    @pytest.mark.unit
    def test_create_rate_limit_marker(self, tmp_path):
        """Test rate limit marker creation."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        
        # Marker doesn't exist
        assert not marker_path.exists()
        
        # Create marker
        client._create_rate_limit_marker("tmean", date)
        
        # Marker should exist and contain timestamp
        assert marker_path.exists()
        content = marker_path.read_text()
        assert "Rate limit hit at" in content
    
    # ============================================
    # Tests for ZIP File Extraction
    # ============================================
    
    @pytest.mark.unit
    def test_get_file_path_extracts_from_existing_zip(self, tmp_path):
        """Test that _get_file_path extracts COG from existing ZIP file."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create ZIP file with COG inside
        zip_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a valid ZIP file with a .tif file inside
        with zipfile.ZipFile(zip_path, 'w') as zip_ref:
            # Create a dummy .tif file content
            zip_ref.writestr("prism_tmean_us_4km_20240615.tif", b"dummy tif content")
        
        # COG file doesn't exist yet
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        assert not cog_path.exists()
        
        # Get file path (should extract from ZIP)
        result_path = client._get_file_path("tmean", date)
        
        # COG should now exist (extracted from ZIP)
        assert cog_path.exists()
        assert result_path == cog_path
    
    @pytest.mark.unit
    def test_get_file_path_handles_empty_zip(self, tmp_path):
        """Test that _get_file_path handles empty ZIP files gracefully."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create empty ZIP file
        zip_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.touch()  # Empty file
        
        # Get file path (should handle gracefully)
        result_path = client._get_file_path("tmean", date)
        
        # Should return COG path (default), but extraction should fail gracefully
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        assert result_path == cog_path
        # COG should not exist (extraction failed)
        assert not cog_path.exists()
    
    @pytest.mark.unit
    def test_get_file_path_handles_corrupted_zip(self, tmp_path):
        """Test that _get_file_path handles corrupted ZIP files gracefully."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create corrupted ZIP file (not a valid ZIP)
        zip_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(b"not a zip file")
        
        # Get file path (should handle gracefully)
        result_path = client._get_file_path("tmean", date)
        
        # Should return COG path (default), but extraction should fail gracefully
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        assert result_path == cog_path
        # COG should not exist (extraction failed)
        assert not cog_path.exists()
    
    @pytest.mark.unit
    def test_get_file_path_prefers_existing_cog_over_zip(self, tmp_path):
        """Test that _get_file_path prefers existing COG over ZIP."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create both COG and ZIP files
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        zip_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.zip"
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        cog_path.write_bytes(b"existing cog content")
        
        with zipfile.ZipFile(zip_path, 'w') as zip_ref:
            zip_ref.writestr("prism_tmean_us_4km_20240615.tif", b"zip cog content")
        
        # Get file path (should prefer existing COG)
        result_path = client._get_file_path("tmean", date)
        
        # Should return existing COG, not extract from ZIP
        assert result_path == cog_path
        # Verify COG content wasn't overwritten
        assert cog_path.read_bytes() == b"existing cog content"
    
    # ============================================
    # Integration Tests for Rate Limit Handling
    # ============================================
    
    @pytest.mark.integration
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_skips_on_rate_limit_marker(self, mock_release, mock_acquire, mock_download, tmp_path):
        """Test that download is skipped when rate limit marker exists."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create rate limit marker
        client._create_rate_limit_marker("tmean", date)
        
        # Verify marker exists
        assert client._check_rate_limit_marker("tmean", date) is True
        
        # Try to download (should skip due to marker)
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        # Should return None without attempting download
        assert result is None
        mock_acquire.assert_not_called()  # Should not even try to acquire lock
        mock_download.assert_not_called()
    
    @pytest.mark.integration
    @patch.object(PRISMClient, '_download_cog_file')
    @patch.object(PRISMClient, '_acquire_file_lock')
    @patch.object(PRISMClient, '_release_file_lock')
    def test_download_prism_file_creates_rate_limit_marker_on_error(self, mock_release, mock_acquire, mock_download, tmp_path):
        """Test that rate limit marker is created when rate limit error occurs."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Mock lock acquisition
        mock_acquire.return_value = True
        
        # Mock rate limit exception
        mock_download.side_effect = PRISMRateLimitError("PRISM rate limit hit")
        
        # Marker doesn't exist yet
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        assert not marker_path.exists()
        
        # Try to download (should hit rate limit)
        result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
        
        # Should return None
        assert result is None
        
        # Marker should be created
        assert marker_path.exists()
        assert client._check_rate_limit_marker("tmean", date) is True
    
    @pytest.mark.integration
    def test_download_prism_file_uses_cached_file(self, tmp_path):
        """Test that cached file is used without attempting download."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create existing COG file
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        cog_path.write_bytes(b"cached file content")
        
        # Mock download to verify it's not called
        with patch.object(PRISMClient, '_download_cog_file') as mock_download, \
             patch.object(PRISMClient, '_acquire_file_lock') as mock_acquire:
            
            result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
            
            # Should return cached file
            assert result == cog_path
            
            # Should not attempt download or acquire lock
            mock_download.assert_not_called()
            mock_acquire.assert_not_called()
    
    @pytest.mark.integration
    def test_download_prism_file_uses_cached_file_with_size_check(self, tmp_path):
        """Test that cached file size is validated."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Create existing COG file with content
        cog_path = prism_dir / "tmean" / "prism_tmean_us_4km_20240615.tif"
        cog_path.parent.mkdir(parents=True, exist_ok=True)
        cog_path.write_bytes(b"valid file content")
        
        # Verify file has size > 0
        assert cog_path.stat().st_size > 0
        
        # Mock download to verify it's not called
        with patch.object(PRISMClient, '_download_cog_file') as mock_download:
            result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
            
            # Should return cached file
            assert result == cog_path
            mock_download.assert_not_called()
    
    @pytest.mark.integration
    def test_full_flow_with_rate_limit_marker(self, tmp_path):
        """Test full flow: rate limit marker prevents download, then expires."""
        prism_dir = tmp_path / "prism"
        client = PRISMClient(data_dir=prism_dir)
        date = datetime(2024, 6, 15)
        
        # Step 1: Create rate limit marker
        client._create_rate_limit_marker("tmean", date)
        assert client._check_rate_limit_marker("tmean", date) is True
        
        # Step 2: Try to download (should skip)
        with patch.object(PRISMClient, '_download_cog_file') as mock_download:
            result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
            assert result is None
            mock_download.assert_not_called()
        
        # Step 3: Make marker stale (older than 24h)
        marker_path = client._get_rate_limit_marker_path("tmean", date)
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(marker_path, (old_time, old_time))
        
        # Step 4: Marker should be expired
        assert client._check_rate_limit_marker("tmean", date) is False
        assert not marker_path.exists()  # Should be removed
        
        # Step 5: Now download should be attempted (marker expired)
        with patch.object(PRISMClient, '_download_cog_file') as mock_download, \
             patch.object(PRISMClient, '_acquire_file_lock', return_value=True), \
             patch.object(PRISMClient, '_release_file_lock'):
            mock_download.return_value = None  # Simulate download failure
            result = client._download_prism_file("tmean", date, max_retries=1, use_web_service=True)
            # Should attempt download now (marker expired)
            mock_download.assert_called_once()

