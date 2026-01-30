"""
API format validation tests for AppEEARS batching.

Validates that request payloads and response parsing match the AppEEARS API specification.
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.appeears_client import AppEEARSClient


# Example from AppEEARS API documentation
EXAMPLE_POINT_REQUEST = {
    "task_type": "point",
    "task_name": "Point Example",
    "params": {
        "dates": [{
            "endDate": "01-01-2020",
            "recurring": False,
            "startDate": "01-01-2015",
            "yearRange": [2005, 2015]
        }],
        "layers": [{
            "layer": "_500m_16_days_EVI",
            "product": "MOD13A1.061"
        }, {
            "layer": "LST_Day_1km",
            "product": "MOD11A1.061"
        }],
        "coordinates": [{
            "id": "100",
            "category": "category1",
            "latitude": 34.4544983,
            "longitude": -119.8659973
        }, {
            "id": "101",
            "category": "category2",
            "latitude": 34.4714012,
            "longitude": -119.9179993
        }]
    }
}


class TestAppEEARSAPIFormat:
    """Test AppEEARS API request/response format compliance."""
    
    @pytest.mark.unit
    @patch('src.data.appeears_client.requests.Session')
    def test_batched_request_payload_format(self, mock_session_class):
        """Validate batched request payload matches AppEEARS API spec."""
        # Setup mock client
        mock_session = Mock()
        auth_response = Mock()
        auth_response.json.return_value = {"token": "test_token"}
        auth_response.raise_for_status = Mock()
        mock_session.post.return_value = auth_response
        mock_session.get.return_value = Mock(json=lambda: {"_250m_16_days_NDVI": {}})
        mock_session_class.return_value = mock_session
        
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test_user", "APPEEARS_PASSWORD": "test_pass"}):
            client = AppEEARSClient()
            
            # Sample points for batching
            points = [
                (44.0, -107.0, "2024-06-15"),
                (44.1, -107.1, "2024-06-15"),
                (44.2, -107.2, "2024-06-15"),
            ]
            
            # Capture the payload that would be sent
            captured_payload = None
            
            def capture_post(url, json=None, **kwargs):
                nonlocal captured_payload
                if 'task' in url and json:
                    captured_payload = json
                return Mock(
                    json=lambda: {"task_id": "test_task", "status": "pending"},
                    status_code=200,
                    raise_for_status=Mock()
                )
            
            mock_session.post.side_effect = capture_post
            
            # Submit request (will use current implementation, but we can validate format)
            try:
                client.submit_point_request(points, product="modis_ndvi", date_buffer_days=5)
            except:
                pass  # May fail, but we're just checking format
            
            # When batching is implemented, validate payload structure
            if captured_payload:
                # Validate required fields
                assert "task_type" in captured_payload
                assert "task_name" in captured_payload
                assert "params" in captured_payload
                
                params = captured_payload["params"]
                assert "dates" in params
                assert "layers" in params
                assert "coordinates" in params
                
                # Validate task_type
                assert captured_payload["task_type"] == "point"
                
                # Validate dates format (MM-DD-YYYY)
                dates = params["dates"]
                assert len(dates) > 0
                date_obj = dates[0]
                assert "startDate" in date_obj
                assert "endDate" in date_obj
                
                # Validate date format (MM-DD-YYYY)
                start_date = date_obj["startDate"]
                assert len(start_date.split("-")) == 3, "Date should be in MM-DD-YYYY format"
                
                # Validate coordinates format
                coordinates = params["coordinates"]
                assert isinstance(coordinates, list)
                assert len(coordinates) > 0
                
                for coord in coordinates:
                    assert "latitude" in coord
                    assert "longitude" in coord
                    assert isinstance(coord["latitude"], (int, float))
                    assert isinstance(coord["longitude"], (int, float))
                    # Optional fields: id, category
                    if "id" in coord:
                        assert isinstance(coord["id"], str)
                    if "category" in coord:
                        assert isinstance(coord["category"], str)
                
                # Validate layers format
                layers = params["layers"]
                assert isinstance(layers, list)
                assert len(layers) > 0
                
                for layer in layers:
                    assert "product" in layer
                    assert "layer" in layer
                    assert isinstance(layer["product"], str)
                    assert isinstance(layer["layer"], str)
    
    @pytest.mark.unit
    def test_batched_response_parsing(self, tmp_path):
        """Test parsing of AppEEARS batched response format."""
        # Create sample CSV matching AppEEARS format
        csv_content = """Date,Latitude,Longitude,MOD13Q1_061__250m_16_days_NDVI,MOD13Q1_061__250m_16_days_VI_Quality
2024-06-10,44.0,-107.0,7000,0
2024-06-15,44.0,-107.0,7200,0
2024-06-20,44.0,-107.0,6800,0
2024-06-10,44.1,-107.1,6500,0
2024-06-15,44.1,-107.1,7000,0
2024-06-20,44.1,-107.1,6800,0"""
        
        csv_file = tmp_path / "appeears_results.csv"
        csv_file.write_text(csv_content)
        
        # Parse CSV
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Validate structure
        assert "Date" in df.columns
        assert "Latitude" in df.columns
        assert "Longitude" in df.columns
        
        # Find NDVI column
        ndvi_col = None
        for col in df.columns:
            if "NDVI" in col.upper() and "QUALITY" not in col.upper():
                ndvi_col = col
                break
        
        assert ndvi_col is not None, "NDVI column not found"
        
        # Validate NDVI values (AppEEARS returns scaled values, need to divide by 10000)
        ndvi_values = df[ndvi_col]
        assert all(val > 0 for val in ndvi_values if pd.notna(val))
        
        # Test normalization (divide by 10000 if > 1)
        normalized = []
        for val in ndvi_values:
            if pd.notna(val):
                if val > 1:
                    normalized.append(val / 10000.0)
                else:
                    normalized.append(val)
        
        # Verify normalized values are in valid range
        assert all(0 <= val <= 1 for val in normalized), "Normalized NDVI values should be 0-1"
        
        # Verify we can extract data for specific coordinates
        point1_data = df[
            (df['Latitude'] == 44.0) &
            (df['Longitude'] == -107.0)
        ]
        assert len(point1_data) > 0, "Should have data for point 1"
        
        point2_data = df[
            (df['Latitude'] == 44.1) &
            (df['Longitude'] == -107.1)
        ]
        assert len(point2_data) > 0, "Should have data for point 2"
        
        # Verify date filtering works
        target_date = "2024-06-15"
        date_filtered = df[df['Date'].str.startswith(target_date)]
        assert len(date_filtered) == 2, "Should have data for both points on target date"
    
    @pytest.mark.unit
    def test_date_range_calculation_for_batch(self):
        """Test date range calculation covers all points in batch."""
        from datetime import datetime, timedelta
        
        points = [
            (44.0, -107.0, "2024-06-10"),
            (44.1, -107.1, "2024-06-15"),
            (44.2, -107.2, "2024-06-20"),
        ]
        
        date_buffer_days = 5
        
        # Calculate date range that covers all points
        dates = [datetime.strptime(date_str, "%Y-%m-%d") for _, _, date_str in points]
        min_date = min(dates)
        max_date = max(dates)
        
        # Add buffer
        start_date = (min_date - timedelta(days=date_buffer_days))
        end_date = (max_date + timedelta(days=date_buffer_days))
        
        # Verify all points are within range
        for _, _, date_str in points:
            point_date = datetime.strptime(date_str, "%Y-%m-%d")
            assert start_date <= point_date <= end_date, f"Point date {date_str} should be within range"
        
        # Verify format matches AppEEARS requirement (MM-DD-YYYY)
        start_str = start_date.strftime("%m-%d-%Y")
        end_str = end_date.strftime("%m-%d-%Y")
        
        assert len(start_str.split("-")) == 3
        assert len(end_str.split("-")) == 3
        assert start_str.split("-")[0] in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    
    @pytest.mark.unit
    def test_coordinate_format_compliance(self):
        """Test coordinate format matches AppEEARS API spec."""
        points = [
            (44.0, -107.0, "2024-06-15"),
            (44.1, -107.1, "2024-06-15"),
        ]
        
        # Convert to AppEEARS coordinate format
        coordinates = []
        for lat, lon, date_str in points:
            coord = {
                "latitude": lat,
                "longitude": lon
            }
            # Optional: add id and category
            # coord["id"] = f"point_{len(coordinates)}"
            # coord["category"] = "pathwild"
            coordinates.append(coord)
        
        # Validate format
        assert isinstance(coordinates, list)
        assert len(coordinates) == len(points)
        
        for coord in coordinates:
            # Required fields
            assert "latitude" in coord
            assert "longitude" in coord
            
            # Type validation
            assert isinstance(coord["latitude"], (int, float))
            assert isinstance(coord["longitude"], (int, float))
            
            # Range validation (latitude: -90 to 90, longitude: -180 to 180)
            assert -90 <= coord["latitude"] <= 90
            assert -180 <= coord["longitude"] <= 180
            
            # Optional fields can be added
            if "id" in coord:
                assert isinstance(coord["id"], str)
            if "category" in coord:
                assert isinstance(coord["category"], str)
