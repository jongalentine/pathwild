"""
Tests for AppEEARS client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import pandas as pd
import os

from src.data.appeears_client import AppEEARSClient


class TestAppEEARSClient:
    """Test AppEEARS client for NDVI data retrieval."""
    
    @pytest.mark.unit
    def test_init_without_credentials(self):
        """Test initialization fails without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="credentials required"):
                AppEEARSClient()
    
    @pytest.mark.unit
    @patch('src.data.appeears_client.requests.post')
    def test_init_with_credentials(self, mock_post):
        """Test successful initialization with credentials."""
        mock_response = Mock()
        mock_response.json.return_value = {"token": "test_token_12345"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test_user", "APPEEARS_PASSWORD": "test_pass"}):
            client = AppEEARSClient()
            assert client.token == "test_token_12345"
            assert client.username == "test_user"
            assert client.password == "test_pass"
    
    @pytest.mark.unit
    @patch('src.data.appeears_client.requests.post')
    @patch('src.data.appeears_client.requests.get')
    @patch('time.sleep')
    def test_submit_point_request(self, mock_sleep, mock_get, mock_post):
        """Test submitting a point extraction request."""
        # Mock authentication
        auth_response = Mock()
        auth_response.json.return_value = {"token": "test_token"}
        auth_response.raise_for_status = Mock()
        mock_post.return_value = auth_response
        
        # Mock product layers query (happens before submission now)
        product_layers_response = Mock()
        product_layers_response.json.return_value = {
            "_250m_16_days_NDVI": {"Description": "16 day NDVI average"}
        }
        product_layers_response.raise_for_status = Mock()
        
        # Mock task submission
        task_response = Mock()
        task_response.json.return_value = {"task_id": "task_123", "status": "pending"}
        task_response.status_code = 202
        task_response.raise_for_status = Mock()
        
        # Setup GET to return product layers, POST to return task response
        mock_get.return_value = product_layers_response
        mock_post.side_effect = [auth_response, task_response]
        
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test_user", "APPEEARS_PASSWORD": "test_pass"}):
            client = AppEEARSClient()
            
            points = [(44.0, -107.0, "2024-06-15")]
            task_id = client.submit_point_request(points, product="modis_ndvi")
            
            assert task_id == "task_123"
            # Should have called GET for product layers and POST for task submission
            assert mock_get.called
            assert mock_post.call_count >= 2  # At least auth + submission
    
    @pytest.mark.unit
    @patch('src.data.appeears_client.requests.post')
    @patch('src.data.appeears_client.requests.get')
    @patch('time.sleep')
    @patch.object(AppEEARSClient, 'wait_for_task')
    def test_get_ndvi_for_points(self, mock_wait_for_task, mock_sleep, mock_get, mock_post):
        """Test complete NDVI extraction workflow."""
        # Mock authentication
        auth_response = Mock()
        auth_response.json.return_value = {"token": "test_token"}
        auth_response.raise_for_status = Mock()
        
        # Mock task submission
        task_response = Mock()
        task_response.json.return_value = {"task_id": "task_123"}
        task_response.raise_for_status = Mock()
        
        # Mock wait_for_task to return immediately with done status
        mock_wait_for_task.return_value = {"status": "done", "bundle_id": "bundle_456"}
        
        # Mock status check (task done) - for download_task_results
        status_response = Mock()
        status_response.json.return_value = {"status": "done", "bundle_id": "bundle_456"}
        status_response.raise_for_status = Mock()
        
        # Mock bundle files list - note: download_task_results uses bundle_id, not task_id
        files_response = Mock()
        files_response.json.return_value = {
            "files": [{"file_id": "file_789", "file_name": "ndvi_results.csv"}]
        }
        files_response.raise_for_status = Mock()
        
        # Mock file download
        download_response = Mock()
        download_response.iter_content.return_value = [
            b"Date,Latitude,Longitude,NDVI,QA\n",
            b"2024-06-15,44.0,-107.0,7000,0\n"
        ]
        download_response.raise_for_status = Mock()
        
        # Chain: status check (for download_task_results), bundle files, file download
        mock_get.side_effect = [status_response, files_response, download_response]
        
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test_user", "APPEEARS_PASSWORD": "test_pass"}):
            import tempfile
            
            # Create a temporary directory for downloads
            with tempfile.TemporaryDirectory() as tmpdir:
                client = AppEEARSClient()
                mock_post.return_value = task_response
                
                # Create a real CSV file that will be read by pd.read_csv
                csv_file = Path(tmpdir) / "ndvi_results.csv"
                csv_file.write_text("Date,Latitude,Longitude,NDVI,QA\n2024-06-15,44.0,-107.0,7000,0\n")
                
                # Mock the download to return our test file
                with patch.object(client, 'download_task_results', return_value=[csv_file]):
                    points = [(44.0, -107.0, "2024-06-15")]
                    result_df = client.get_ndvi_for_points(points, output_dir=Path(tmpdir))
                    
                    assert len(result_df) > 0
                    assert "ndvi" in result_df.columns
                    assert "latitude" in result_df.columns  
                    assert "longitude" in result_df.columns

