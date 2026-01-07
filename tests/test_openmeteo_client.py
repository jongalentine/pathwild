"""
Tests for Open-Meteo client.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

from src.data.openmeteo_client import OpenMeteoClient


class TestOpenMeteoClient:
    """Test Open-Meteo client for weather forecasts."""
    
    @pytest.mark.unit
    def test_init(self):
        """Test Open-Meteo client initialization."""
        client = OpenMeteoClient()
        assert client.forecast_days == 7
        assert client.timeout == 30
        assert client.cache == {}
    
    @pytest.mark.unit
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        client = OpenMeteoClient(forecast_days=14, timeout=60)
        assert client.forecast_days == 14
        assert client.timeout == 60
    
    @pytest.mark.unit
    @patch('src.data.openmeteo_client.requests.get')
    def test_get_forecast(self, mock_get):
        """Test getting weather forecast."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2024-06-16", "2024-06-17"],
                "temperature_2m_max": [75.0, 78.0],
                "temperature_2m_min": [50.0, 52.0],
                "temperature_2m_mean": [62.5, 65.0],
                "precipitation_sum": [0.1, 0.2]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = OpenMeteoClient()
        result = client.get_forecast(44.0, -107.0)
        
        assert "daily" in result
        assert len(result["daily"]["time"]) == 2
        mock_get.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.data.openmeteo_client.requests.get')
    def test_parse_forecast_response(self, mock_get):
        """Test parsing forecast response."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2024-06-16", "2024-06-17"],
                "temperature_2m_max": [75.0, 78.0],
                "temperature_2m_min": [50.0, 52.0],
                "temperature_2m_mean": [62.5, 65.0],
                "precipitation_sum": [0.1, 0.2]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = OpenMeteoClient()
        data = client.get_forecast(44.0, -107.0)
        forecasts = client.parse_forecast_response(data)
        
        assert len(forecasts) == 2
        assert forecasts[0]["date"] == "2024-06-16"
        assert forecasts[0]["temp_max_f"] == 75.0
        assert forecasts[0]["temp_min_f"] == 50.0
        assert forecasts[0]["temp_mean_f"] == 62.5
        assert forecasts[0]["precipitation_inches"] == 0.1
    
    @pytest.mark.unit
    @patch('src.data.openmeteo_client.requests.get')
    def test_get_forecast_for_date(self, mock_get):
        """Test getting forecast for specific date."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2024-06-16", "2024-06-17"],
                "temperature_2m_max": [75.0, 78.0],
                "temperature_2m_min": [50.0, 52.0],
                "temperature_2m_mean": [62.5, 65.0],
                "precipitation_sum": [0.1, 0.2]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = OpenMeteoClient()
        forecast = client.get_forecast_for_date(44.0, -107.0, "2024-06-16")
        
        assert forecast is not None
        assert forecast["date"] == "2024-06-16"
        assert forecast["temp_max_f"] == 75.0
    
    @pytest.mark.unit
    @patch('src.data.openmeteo_client.requests.get')
    def test_get_historical(self, mock_get):
        """Test getting historical weather data."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2024-06-15"],
                "temperature_2m_max": [72.0],
                "temperature_2m_min": [48.0],
                "temperature_2m_mean": [60.0],
                "precipitation_sum": [0.15]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = OpenMeteoClient()
        result = client.get_historical(44.0, -107.0, "2024-06-15", "2024-06-15")
        
        assert "daily" in result
        assert len(result["daily"]["time"]) == 1
    
    @pytest.mark.unit
    @patch('src.data.openmeteo_client.requests.get')
    @patch('time.sleep')
    def test_retry_on_failure(self, mock_sleep, mock_get):
        """Test retry logic on API failure."""
        import requests
        
        # First call fails with a RequestException (will be caught and retried)
        # Second call succeeds
        mock_response = Mock()
        mock_response.json.return_value = {"daily": {"time": []}}
        mock_response.raise_for_status = Mock()
        
        mock_get.side_effect = [
            requests.RequestException("Network error"),
            mock_response
        ]
        
        client = OpenMeteoClient()
        result = client._make_request("https://api.open-meteo.com/v1/forecast", {})
        
        assert mock_get.call_count == 2
        assert mock_sleep.called
        assert result == {"daily": {"time": []}}

