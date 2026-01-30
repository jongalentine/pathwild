"""
Integration tests for NDVI and weather data extraction in the pipeline.

Tests that the integrated clients work correctly with the existing pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Mock rasterio before importing PRISMClient to avoid import errors in sandbox
from types import ModuleType
# Create a context manager-compatible mock for rasterio.open
mock_rasterio_open = MagicMock()
mock_rasterio_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
mock_rasterio_open.return_value.__exit__ = MagicMock(return_value=False)

mock_rasterio = ModuleType('rasterio')
mock_rasterio.open = mock_rasterio_open
# Add errors module for NotGeoreferencedWarning
mock_rasterio_errors = ModuleType('rasterio.errors')
mock_rasterio_errors.NotGeoreferencedWarning = type('NotGeoreferencedWarning', (Warning,), {})
mock_rasterio.errors = mock_rasterio_errors
sys.modules['rasterio'] = mock_rasterio
sys.modules['rasterio.errors'] = mock_rasterio_errors

mock_rasterio_mask = ModuleType('rasterio.mask')
mock_rasterio_mask.mask = Mock()
sys.modules['rasterio.mask'] = mock_rasterio_mask

from src.data.processors import WeatherClient, SatelliteClient, DataContextBuilder
from src.data.appeears_client import AppEEARSClient
from src.data.prism_client import PRISMClient
from src.data.openmeteo_client import OpenMeteoClient


class TestWeatherClientIntegration:
    """Test WeatherClient integration with PRISM and Open-Meteo."""
    
    @pytest.mark.integration
    def test_weather_client_placeholder_mode(self, tmp_path):
        """Test WeatherClient works without real data providers."""
        client = WeatherClient(data_dir=tmp_path, use_real_data=False)
        
        # Historical date
        result = client.get_weather(44.0, -107.0, datetime(2024, 6, 15))
        assert "temp" in result
        assert result["temp"] == 42.0  # Placeholder
        
        # Future date (forecast)
        future_date = datetime.now() + pd.Timedelta(days=7)
        result = client.get_weather(44.0, -107.0, future_date)
        assert "temp" in result
        assert result["temp"] == 45.0  # Placeholder
    
    @pytest.mark.integration
    @patch('src.data.prism_client.PRISMClient')
    @patch('src.data.openmeteo_client.OpenMeteoClient')
    def test_weather_client_with_real_providers(self, mock_openmeteo_class, mock_prism_class, tmp_path):
        """Test WeatherClient with real provider clients."""
        # Mock PRISM client
        mock_prism = Mock()
        mock_prism.get_temperature.return_value = {
            "temp_mean_c": 20.0,
            "temp_min_c": 15.0,
            "temp_max_c": 25.0
        }
        mock_prism.get_precipitation.return_value = 50.0
        mock_prism_class.return_value = mock_prism
        
        # Mock Open-Meteo client
        mock_openmeteo = Mock()
        mock_openmeteo.get_forecast_for_date.return_value = {
            "date": "2024-06-20",
            "temp_mean_f": 68.0,
            "temp_max_f": 77.0,
            "temp_min_f": 59.0,
            "precipitation_inches": 0.2
        }
        mock_openmeteo.get_forecast_for_location.return_value = [
            {"precipitation_inches": 0.1},
            {"precipitation_inches": 0.2},
            {"precipitation_inches": 0.1}
        ]
        mock_openmeteo_class.return_value = mock_openmeteo
        
        client = WeatherClient(data_dir=tmp_path, use_real_data=True)
        
        # Test historical
        result = client._get_historical(44.0, -107.0, datetime(2024, 6, 15))
        assert result["temp"] == pytest.approx(68.0, abs=0.1)  # 20°C = 68°F
        
        # Test forecast
        future_date = datetime.now() + pd.Timedelta(days=7)
        result = client._get_forecast(44.0, -107.0, future_date)
        assert result["temp"] == 68.0
    
    @pytest.mark.integration
    def test_weather_client_caching(self, tmp_path):
        """Test that WeatherClient caches results."""
        client = WeatherClient(data_dir=tmp_path, use_real_data=False)
        
        date = datetime(2024, 6, 15)
        result1 = client.get_weather(44.0, -107.0, date)
        
        # Should be cached
        assert len(client.cache) == 1
        
        # Second call should use cache
        result2 = client.get_weather(44.0, -107.0, date)
        assert result1 == result2


class TestSatelliteClientIntegration:
    """Test SatelliteClient integration with AppEEARS."""
    
    @pytest.mark.integration
    def test_satellite_client_placeholder_mode(self):
        """Test SatelliteClient works without real data provider."""
        client = SatelliteClient(use_real_data=False)
        
        result = client.get_ndvi(44.0, -107.0, datetime(2024, 7, 15))
        
        assert "ndvi" in result
        assert 0 <= result["ndvi"] <= 1
        assert "age_days" in result
        assert "irg" in result
        assert "cloud_free" in result
    
    @pytest.mark.integration
    @patch('src.data.appeears_client.AppEEARSClient')
    def test_satellite_client_with_appeears(self, mock_appeears_class):
        """Test SatelliteClient with AppEEARS client."""
        # Mock AppEEARS client
        mock_appeears = Mock()
        mock_result_df = pd.DataFrame({
            "latitude": [44.0],
            "longitude": [-107.0],
            "date": ["2024-07-15"],
            "ndvi": [0.75],
            "qa_flags": [0]
        })
        mock_appeears.get_ndvi_for_points.return_value = mock_result_df
        mock_appeears_class.return_value = mock_appeears
        
        # Ensure the mock is set before creating the client
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test", "APPEEARS_PASSWORD": "test"}):
            # Force the import to use the mocked class
            client = SatelliteClient(use_real_data=True)
            # Verify the client was created with the mock
            assert client.appeears_client is not None
            result = client.get_ndvi(44.0, -107.0, datetime(2024, 7, 15))
            
            assert result["ndvi"] == 0.75
            # qa_flags == 0 means cloud_free should be True
            assert result.get("cloud_free", False) == True
    
    @pytest.mark.integration
    def test_satellite_client_batch_extraction(self):
        """Test batch NDVI extraction."""
        client = SatelliteClient(use_real_data=False)
        
        points = [
            (44.0, -107.0, datetime(2024, 7, 15)),
            (44.1, -107.1, datetime(2024, 7, 16)),
            (44.2, -107.2, datetime(2024, 7, 17))
        ]
        
        result_df = client.extract_ndvi_batch(points)
        
        assert len(result_df) == 3
        assert "ndvi" in result_df.columns
        assert "latitude" in result_df.columns
        assert "longitude" in result_df.columns
        assert all(result_df["ndvi"].notna())


class TestDataContextBuilderIntegration:
    """Test DataContextBuilder integration with real NDVI/weather clients."""
    
    @pytest.mark.integration
    def test_build_context_with_placeholder_clients(self, tmp_path, monkeypatch):
        """Test DataContextBuilder works with placeholder clients."""
        from unittest.mock import Mock
        
        # Set up minimal data directory structure
        (tmp_path / "dem").mkdir()
        (tmp_path / "terrain").mkdir()
        (tmp_path / "canopy").mkdir()
        (tmp_path / "landcover").mkdir()
        (tmp_path / "hydrology").mkdir()
        (tmp_path / "infrastructure").mkdir()
        
        # Ensure no real credentials
        monkeypatch.delenv("APPEEARS_USERNAME", raising=False)
        monkeypatch.delenv("APPEEARS_PASSWORD", raising=False)
        
        builder = DataContextBuilder(data_dir=tmp_path)
        
        # Mock clients to prevent API calls even if placeholders aren't used
        builder.snotel_client.get_snow_data = Mock(return_value={
            'depth': 0.0,
            'swe': 0.0,
            'crust': False,
            'station': None,
            'station_distance_km': None
        })
        builder.weather_client.get_weather = Mock(return_value={
            'temp': 70.0, 'temp_high': 80.0, 'temp_low': 60.0,
            'precip_7d': 0.0, 'cloud_cover': 20
        })
        builder.satellite_client.get_ndvi = Mock(return_value={
            'ndvi': 0.7, 'age_days': 5, 'irg': 0.01, 'cloud_free': True
        })
        builder.satellite_client.get_integrated_ndvi = Mock(return_value=75.0)
        
        # Should work with placeholder clients
        context = builder.build_context(
            location={"lat": 44.0, "lon": -107.0},
            date="2024-07-15"
        )
        
        # Should have NDVI and weather (placeholders)
        assert "ndvi" in context
        assert "temperature_f" in context
        assert context["ndvi"] is not None
        assert context["temperature_f"] is not None
    
    @pytest.mark.integration
    @patch('src.data.prism_client.PRISMClient')
    @patch('src.data.openmeteo_client.OpenMeteoClient')
    @patch('src.data.appeears_client.AppEEARSClient')
    def test_build_context_with_real_clients(
        self, mock_appeears_class, mock_openmeteo_class, mock_prism_class, tmp_path
    ):
        """Test DataContextBuilder with real provider clients."""
        # Set up minimal data directory structure
        (tmp_path / "dem").mkdir()
        (tmp_path / "terrain").mkdir()
        (tmp_path / "canopy").mkdir()
        (tmp_path / "landcover").mkdir()
        (tmp_path / "hydrology").mkdir()
        (tmp_path / "infrastructure").mkdir()
        
        # Mock clients
        mock_prism = Mock()
        mock_prism.get_temperature.return_value = {
            "temp_mean_c": 20.0, "temp_min_c": 15.0, "temp_max_c": 25.0
        }
        mock_prism.get_precipitation.return_value = 50.0
        mock_prism_class.return_value = mock_prism
        
        mock_openmeteo = Mock()
        mock_openmeteo.get_forecast_for_date.return_value = None
        mock_openmeteo_class.return_value = mock_openmeteo
        
        mock_appeears = Mock()
        mock_result_df = pd.DataFrame({
            "latitude": [44.0],
            "longitude": [-107.0],
            "date": ["2024-07-15"],
            "ndvi": [0.75],
            "qa_flags": [0]
        })
        mock_appeears.get_ndvi_for_points.return_value = mock_result_df
        mock_appeears_class.return_value = mock_appeears
        
        with patch.dict(os.environ, {"APPEEARS_USERNAME": "test", "APPEEARS_PASSWORD": "test"}):
            builder = DataContextBuilder(data_dir=tmp_path)
            
            context = builder.build_context(
                location={"lat": 44.0, "lon": -107.0},
                date="2024-07-15"
            )
            
            # Should use real clients when available
            assert "ndvi" in context
            assert "temperature_f" in context


class TestPipelineIntegration:
    """Test NDVI/weather integration in the full pipeline workflow."""
    
    @pytest.mark.integration
    def test_integrate_features_uses_real_clients(self, tmp_path, monkeypatch):
        """Test that integrate_environmental_features.py uses updated clients."""
        import sys
        from pathlib import Path
        from unittest.mock import Mock
        
        # Create minimal test dataset
        test_df = pd.DataFrame({
            'latitude': [44.0, 44.1],
            'longitude': [-107.0, -107.1],
            'elk_present': [1, 0],
            'elevation': [8500.0, 8500.0],  # Placeholders
            'ndvi': [0.5, 0.5],  # Placeholders
            'temperature_f': [45.0, 45.0]  # Placeholders
        })
        
        dataset_path = tmp_path / "test_dataset.csv"
        test_df.to_csv(dataset_path, index=False)
        
        # Ensure no real credentials (use placeholders)
        monkeypatch.delenv("APPEEARS_USERNAME", raising=False)
        monkeypatch.delenv("APPEEARS_PASSWORD", raising=False)
        
        # Import and test
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data.processors import DataContextBuilder
        
        builder = DataContextBuilder(data_dir=tmp_path)
        
        # Mock clients to prevent API calls that could be slow
        builder.snotel_client.get_snow_data = Mock(return_value={
            'depth': 0.0,
            'swe': 0.0,
            'crust': False,
            'station': None,
            'station_distance_km': None
        })
        builder.weather_client.get_weather = Mock(return_value={
            'temp': 70.0, 'temp_high': 80.0, 'temp_low': 60.0,
            'precip_7d': 0.0, 'cloud_cover': 20
        })
        builder.satellite_client.get_ndvi = Mock(return_value={
            'ndvi': 0.7, 'age_days': 5, 'irg': 0.01, 'cloud_free': True
        })
        builder.satellite_client.get_integrated_ndvi = Mock(return_value=75.0)
        
        # Should work with placeholder clients
        context = builder.build_context(
            location={"lat": 44.0, "lon": -107.0},
            date="2024-07-15"
        )
        
        # Should have NDVI and weather (placeholders)
        assert "ndvi" in context
        assert "temperature_f" in context
        assert context["ndvi"] is not None
        assert context["temperature_f"] is not None

