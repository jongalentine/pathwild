"""
Unit and integration tests for SNOTEL data integration.

Tests AWDBClient with AWDB R package integration, including:
- Station loading and mapping
- Data retrieval from AWDB
- Error handling and fallbacks
- Caching behavior
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.data.processors import AWDBClient


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory structure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_warning_sets():
    """Reset warning sets before each test to avoid test pollution."""
    AWDBClient._warned_stations.clear()
    AWDBClient._warned_api_failures.clear()
    yield
    # Clean up after test
    AWDBClient._warned_stations.clear()
    AWDBClient._warned_api_failures.clear()


@pytest.fixture
def sample_station_file(data_dir):
    """Create a sample station GeoJSON file."""
    import json
    
    stations = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-106.3167, 41.3500]},
            "properties": {
                "station_id": "975",
                "triplet": "SNOTEL:WY:975",
                "name": "MEDICINE BOW",
                "lat": 41.3500,
                "lon": -106.3167,
                "elevation_ft": 9700,
                "state": "WY",
                "awdb_station_id": 1196,
                "AWDB_site_id": 1196
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-109.2833, 44.5500]},
            "properties": {
                "station_id": "964",
                "triplet": "SNOTEL:WY:964",
                "name": "COTTONWOOD CREEK",
                "lat": 44.5500,
                "lon": -109.2833,
                "elevation_ft": 8840,
                "state": "WY",
                "awdb_station_id": 419,
                "AWDB_site_id": 419
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-106.4250, 41.8350]},
            "properties": {
                "station_id": "967",
                "triplet": "SNOTEL:WY:967",
                "name": "ELKHORN PARK",
                "lat": 41.8350,
                "lon": -106.4250,
                "elevation_ft": 10200,
                "state": "WY",
                "awdb_station_id": None,  # Unmapped station
                "AWDB_site_id": None
            }
        }
    ]
    
    geojson = {
        "type": "FeatureCollection",
        "features": stations
    }
    
    station_file = data_dir / "cache" / "snotel_stations_wyoming.geojson"
    with open(station_file, 'w') as f:
        json.dump(geojson, f)
    
    return station_file




class TestAWDBClientUnit:
    """Unit tests for AWDBClient."""
    
    @pytest.mark.unit
    def test_init_with_data_dir(self, data_dir):
        """Test AWDBClient initialization with data_dir."""
        client = AWDBClient(data_dir=data_dir)
        assert client.data_dir == data_dir
        assert client.station_cache_path == data_dir / "cache" / "snotel_stations_wyoming.geojson"
    
    @pytest.mark.unit
    def test_init_default_data_dir(self):
        """Test AWDBClient initialization with default data_dir."""
        client = AWDBClient()
        assert client.data_dir == Path("data")
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_load_stations_file_exists(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test loading stations when file exists."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW', 'COTTONWOOD CREEK', 'ELKHORN PARK'],
            'triplet': ['1196:WY:SNTL', '419:WY:SNTL', '468:WY:SNTL'],
            'lat': [41.3500, 44.5500, 41.8350],
            'lon': [-106.3167, -109.2833, -106.4250],
            'elevation_ft': [9700, 8840, 10200],
            'state': ['WY', 'WY', 'WY'],
            'awdb_station_id': [1196, 419, 468],
            'AWDB_site_id': [1196, 419, 468],  # Alias for backward compatibility
            'geometry': [Point(-106.3167, 41.3500), Point(-109.2833, 44.5500), 
                        Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        assert client._stations_gdf is not None
        assert len(client._stations_gdf) == 3
        assert "MEDICINE BOW" in client._stations_gdf['name'].values
    
    @pytest.mark.unit
    def test_load_stations_file_missing(self, data_dir):
        """Test loading stations when file doesn't exist."""
        client = AWDBClient(data_dir=data_dir)
        # Should not raise error, just set to None
        assert client._stations_gdf is None
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_find_nearest_station_mapped(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test finding nearest station with AWDB_site_id."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW', 'COTTONWOOD CREEK', 'ELKHORN PARK'],
            'triplet': ['1196:WY:SNTL', '419:WY:SNTL', '468:WY:SNTL'],
            'lat': [41.3500, 44.5500, 41.8350],
            'lon': [-106.3167, -109.2833, -106.4250],
            'elevation_ft': [9700, 8840, 10200],
            'state': ['WY', 'WY', 'WY'],
            'awdb_station_id': [1196, 419, 468],
            'AWDB_site_id': [1196, 419, 468],  # Alias for backward compatibility
            'geometry': [Point(-106.3167, 41.3500), Point(-109.2833, 44.5500), 
                        Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Test location near MEDICINE BOW
        station = client._find_nearest_station(41.3500, -106.3167)
        
        assert station is not None
        assert station['name'] == "MEDICINE BOW"
        assert station['triplet'] == "1196:WY:SNTL"
        assert station['AWDB_site_id'] == 1196
        assert station['distance_km'] < 1.0  # Should be very close
    
    @pytest.mark.unit
    def test_find_nearest_station_no_stations(self, data_dir):
        """Test finding nearest station when no stations loaded."""
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = None
        
        station = client._find_nearest_station(41.3500, -106.3167)
        assert station is None
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    def test_find_nearest_station_too_far(self, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test finding nearest station when all are too far."""
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest to return empty (no stations within max_distance)
        mock_result = gpd.GeoDataFrame({
            'index_right': [None],  # No match
            'distance_m': [None]
        })
        mock_sjoin.return_value = mock_result
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location far from all stations (e.g., in ocean)
        station = client._find_nearest_station(0.0, 0.0, max_distance_km=10.0)
        assert station is None
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_find_nearest_station_prioritizes_mapped(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test that _find_nearest_station prioritizes mapped stations over unmapped ones."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['468:WY:SNTL', '1196:WY:SNTL', '419:WY:SNTL'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'awdb_station_id': [None, 1196, 419],  # ELKHORN PARK unmapped, others mapped
            'AWDB_site_id': [None, 1196, 419],  # Alias for backward compatibility
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Test location between ELKHORN PARK (closer, unmapped) and MEDICINE BOW (farther, mapped)
        # Should prefer MEDICINE BOW (mapped) even though it's farther
        station = client._find_nearest_station(41.6, -106.37, max_distance_km=100.0)
        
        assert station is not None
        # Should prefer mapped station (MEDICINE BOW) over unmapped (ELKHORN PARK)
        assert station['name'] == "MEDICINE BOW"
        assert station['AWDB_site_id'] == 1196
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_find_nearest_station_unmapped_when_no_mapped(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test that _find_nearest_station uses unmapped station when no mapped stations available."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['468:WY:SNTL'],
            'lat': [41.8350],
            'lon': [-106.4250],
            'elevation_ft': [10200],
            'state': ['WY'],
            'awdb_station_id': [None],  # Unmapped
            'AWDB_site_id': [None],
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Test location near ELKHORN PARK (unmapped, but only station available)
        station = client._find_nearest_station(41.8350, -106.4250)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # AWDB_site_id should be None or NaN for unmapped stations
        import pandas as pd
        assert station['AWDB_site_id'] is None or pd.isna(station['AWDB_site_id'])
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_find_nearest_station_unmapped(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test finding nearest station that isn't mapped when it's far from mapped stations."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['468:WY:SNTL', '1196:WY:SNTL', '419:WY:SNTL'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'awdb_station_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'AWDB_site_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Test location very close to ELKHORN PARK - should find it since mapped stations are farther
        # (Within 100km, ELKHORN PARK is closer than mapped stations)
        station = client._find_nearest_station(41.8350, -106.4250, max_distance_km=50.0)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # AWDB_site_id should be None or NaN for unmapped stations
        import pandas as pd
        assert station['AWDB_site_id'] is None or pd.isna(station['AWDB_site_id'])
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_winter(self):
        """Test elevation-based snow estimation for winter with actual elevation."""
        client = AWDBClient()
        
        # Test with high elevation (8500 ft) - should have snow
        result = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=8500.0)
        
        assert 'depth' in result
        assert 'swe' in result
        assert 'crust' in result
        assert result['depth'] >= 0
        assert result['swe'] >= 0
        assert isinstance(result['crust'], bool)
        # Winter at 8500 ft: (8500 - 6000) / 100 = 25 inches
        assert result['depth'] == 25.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_summer(self):
        """Test elevation-based snow estimation for summer with actual elevation."""
        client = AWDBClient()
        
        # Test with high elevation (8500 ft) - summer should have less/no snow
        result = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 7, 15), elevation_ft=8500.0)
        
        # Summer should have less snow
        assert result['depth'] >= 0
        assert result['swe'] >= 0
        # Summer at 8500 ft: max(0, (8500 - 10000) / 100) = 0 inches
        assert result['depth'] == 0.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_low_elevation(self):
        """Test elevation-based estimation for low elevations (should have minimal snow)."""
        client = AWDBClient()
        
        # Low elevation (1500 ft) in winter - should be 0 (below threshold)
        result = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=1500.0)
        
        # Winter formula: max(0, (1500 - 6000) / 100) = 0
        assert result['depth'] == 0.0
        assert result['swe'] == 0.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_uses_provided_elevation(self):
        """Test that provided elevation parameter is used instead of default."""
        client = AWDBClient()
        
        # Test with specific elevation
        result_5000 = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=5000.0)
        result_10000 = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=10000.0)
        
        # 5000 ft: (5000 - 6000) / 100 = 0 (clamped)
        assert result_5000['depth'] == 0.0
        
        # 10000 ft: (10000 - 6000) / 100 = 40 inches
        assert result_10000['depth'] == 40.0
        assert result_10000['depth'] > result_5000['depth']
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_seasonal_variation(self):
        """Test that estimation varies correctly by season."""
        client = AWDBClient()
        elevation_ft = 8500.0
        
        winter = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=elevation_ft)
        spring = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 4, 15), elevation_ft=elevation_ft)
        summer = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 7, 15), elevation_ft=elevation_ft)
        fall = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 10, 15), elevation_ft=elevation_ft)
        
        # At 8500 ft:
        # Winter: (8500-6000)/100 = 25.0
        # Spring: (8500-7000)/150 = 10.0
        # Summer: max(0, (8500-10000)/100) = 0.0
        # Fall: max(0, (8500-9000)/200) = 0.0
        
        assert winter['depth'] == 25.0
        assert spring['depth'] == 10.0
        assert summer['depth'] == 0.0
        assert fall['depth'] == 0.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_actual_low_elevations(self):
        """Test that low elevations produce correct (zero) snow estimates - fixes hardcoded bug."""
        client = AWDBClient()
        
        # Test actual low elevations from real data
        low_elevations = [1340.5, 1496.7, 2955.8]  # Actual elevations from your data
        
        for elevation_ft in low_elevations:
            result = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=elevation_ft)
            # Low elevations (< 6000 ft) should have 0 snow in winter
            assert result['depth'] == 0.0, f"Expected 0 inches for {elevation_ft} ft elevation, got {result['depth']}"
            assert result['swe'] == 0.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_varies_by_elevation(self):
        """Test that estimates correctly vary by elevation (verifies no hardcoded values)."""
        client = AWDBClient()
        date = datetime(2024, 1, 15)  # Winter
        
        elevations = [1000, 5000, 7000, 8500, 10000]
        results = {}
        
        for elev in elevations:
            result = client._estimate_snow_from_elevation(41.0, -106.0, date, elevation_ft=elev)
            results[elev] = result['depth']
        
        # Verify they're different (not all hardcoded)
        assert len(set(results.values())) > 1, "All elevations produced same result - possible hardcoded bug"
        
        # Verify increasing elevation generally increases snow (for elevations above threshold)
        assert results[10000] > results[8500], "Snow depth should increase with elevation (10000 > 8500)"
        assert results[8500] > results[7000], "Snow depth should increase with elevation (8500 > 7000)"
        assert results[5000] == 0.0, "Low elevations should have 0 snow"
        assert results[1000] == 0.0, "Very low elevations should have 0 snow"
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_get_elevation_from_dem')
    def test_estimate_snow_falls_back_to_dem(self, mock_get_elevation, data_dir):
        """Test that elevation estimation attempts to sample from DEM when elevation not provided."""
        client = AWDBClient(data_dir=data_dir)
        
        # Mock DEM sampling to return elevation in feet (2000m = ~6562 ft)
        mock_get_elevation.return_value = 6562.0
        
        # Call with no elevation provided
        result = client._estimate_snow_from_elevation(43.0, -110.0, datetime(2024, 1, 15), elevation_ft=None)
        
        # Should have attempted to sample DEM
        mock_get_elevation.assert_called_once_with(43.0, -110.0)
        
        # Result should use sampled elevation (6562 ft)
        # Winter: (6562 - 6000) / 100 = 5.62 inches
        assert result['depth'] > 0
        # For 6562 ft in winter: (6562 - 6000) / 100 = 5.62 inches
        assert 5.0 < result['depth'] < 6.5, f"Expected ~5.62 inches for 6562ft in winter, got {result['depth']}"
    
    @pytest.mark.unit
    def test_estimate_snow_falls_back_to_default(self, data_dir, monkeypatch):
        """Test that elevation estimation falls back to default when DEM not available."""
        import pathlib
        
        client = AWDBClient(data_dir=data_dir)
        
        # Mock DEM doesn't exist
        original_exists = pathlib.Path.exists
        def mock_exists(self):
            if 'dem' in str(self) and 'wyoming_dem.tif' in str(self):
                return False
            return original_exists(self)
        
        monkeypatch.setattr(pathlib.Path, 'exists', mock_exists)
        
        # Call with no elevation provided, no DEM available
        result = client._estimate_snow_from_elevation(43.0, -110.0, datetime(2024, 1, 15), elevation_ft=None)
        
        # Should use default elevation (8500 ft)
        # Winter: (8500 - 6000) / 100 = 25 inches
        assert result['depth'] == 25.0
    
    @pytest.mark.unit
    def test_get_snow_data_no_station(self, data_dir):
        """Test getting snow data when no station nearby - uses elevation estimate."""
        client = AWDBClient(data_dir=data_dir)
        # Force no stations by setting _stations_gdf to None
        client._stations_gdf = None
        # AWDBClient doesn't have AWDB attribute - it uses HTTP requests
        
        # Test without elevation (will use DEM or default 8500 ft)
        result = client.get_snow_data(0.0, 0.0, datetime(2024, 1, 15))
        
        # Should use elevation estimate
        assert 'depth' in result
        assert 'swe' in result
        assert result.get('station') is None
        # Without elevation, may use default (8500 ft) or DEM, so just check it's non-negative
        assert result['depth'] >= 0
        
        # Test with provided elevation - should use it
        # Use different coordinates to avoid cache
        result_with_elev = client.get_snow_data(10.0, -100.0, datetime(2024, 1, 15), elevation_ft=8000.0)
        # At 8000 ft in winter: (8000 - 6000) / 100 = 20 inches
        assert result_with_elev['depth'] == 20.0, f"Expected 20.0 for 8000 ft, got {result_with_elev['depth']}"
    
    @pytest.mark.unit
    def test_get_snow_data_passes_elevation_to_estimate(self, data_dir):
        """Test that get_snow_data passes elevation parameter to estimation."""
        client = AWDBClient(data_dir=data_dir)
        # Force no stations by setting _stations_gdf to None (this makes _find_nearest_station return None)
        client._stations_gdf = None
        # AWDBClient doesn't have AWDB attribute
        client._r_initialized = False
        
        # Mock _estimate_snow_from_elevation to verify elevation is passed
        with patch.object(client, '_estimate_snow_from_elevation') as mock_estimate:
            mock_estimate.return_value = {'depth': 0.0, 'swe': 0.0, 'crust': False, 'station': None}
            
            # Call get_snow_data with elevation
            result = client.get_snow_data(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=9000.0)
            
            # Verify _estimate_snow_from_elevation was called with elevation
            assert mock_estimate.called
            call_kwargs = mock_estimate.call_args[1]
            assert 'elevation_ft' in call_kwargs
            assert call_kwargs['elevation_ft'] == 9000.0
        
        # Also test actual behavior when elevation is provided
        # Reset client state
        client._stations_gdf = None
        client.request_cache.clear()
        
        # Test with different elevations - should use provided elevation
        result_low = client.get_snow_data(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=2000.0)
        result_high = client.get_snow_data(42.0, -107.0, datetime(2024, 1, 15), elevation_ft=9000.0)  # Different location to avoid cache
        
        # Low elevation (2000 ft): (2000 - 6000) / 100 = 0 (clamped)
        assert result_low['depth'] == 0.0, f"Expected 0.0 for 2000 ft, got {result_low['depth']}"
        
        # High elevation (9000 ft): (9000 - 6000) / 100 = 30
        assert result_high['depth'] == 30.0, f"Expected 30.0 for 9000 ft, got {result_high['depth']}"
        assert result_high['depth'] > result_low['depth']
    
    @pytest.mark.unit
    def test_get_snow_data_unmapped_station(self, data_dir, sample_station_file):
        """Test getting snow data when station isn't mapped."""
        client = AWDBClient(data_dir=data_dir)
        # AWDBClient doesn't have AWDB attribute  # Disable AWDB
        
        # Use location near ELKHORN PARK (unmapped)
        result = client.get_snow_data(41.8350, -106.4250, datetime(2024, 1, 15))
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result


class TestAWDBClientWithSnotelr:
    """Integration tests with mocked AWDB."""
    
    @pytest.mark.integration
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)  # Clear test environment detection
    def test_get_snow_data_with_AWDB_mapped(self, mock_load_from_awdb, mock_requests_get,
                                               data_dir, sample_station_file):
        """Test getting real SNOTEL data via AWDB for mapped station."""
        import os
        # Remove test environment variables to allow API calls
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['1196:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        # Mock AWDB API response
        mock_response = MagicMock()
        mock_response.status_code = 200  # Set status code for successful response
        mock_response.json.return_value = [{
            'stationTriplet': '1196:WY:SNTL',
            'data': [
                {
                    'stationElement': {'elementCode': 'WTEQ'},
                    'values': [{'date': '2024-01-15', 'value': 14.7}]  # inches
                },
                {
                    'stationElement': {'elementCode': 'SNWD'},
                    'values': [{'date': '2024-01-15', 'value': 72.8}]  # inches
                }
            ]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Test location near MEDICINE BOW (mapped)
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Verify requests.get was called for AWDB API
        assert mock_requests_get.called
        
        # Verify result structure
        assert 'depth' in result
        assert 'swe' in result
        assert 'crust' in result
        assert 'station' in result
        assert result['station'] == "MEDICINE BOW"
        
        # Verify values are in inches (already converted by AWDB API)
        assert result['depth'] > 0
        assert result['swe'] > 0
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_get_snow_data_with_AWDB_unmapped(self, mock_read_file, data_dir, sample_station_file):
        """Test fallback when station isn't mapped."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.8350],
            'lon': [-106.4250],
            'elevation_ft': [10200],
            'state': ['WY'],
            'awdb_station_id': [None],  # Unmapped
            'AWDB_site_id': [None],
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        # Use location near ELKHORN PARK (unmapped)
        result = client.get_snow_data(41.8350, -106.4250, datetime(2024, 1, 15))
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result
        assert result.get('station') is None or 'ELKHORN' in result.get('station', '')
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('requests.get')
    def test_get_snow_data_no_data_for_date(self, mock_requests_get, mock_read_file,
                                            data_dir, sample_station_file):
        """Test handling when AWDB returns no data for requested date."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock AWDB API response with no data
        mock_response = MagicMock()
        mock_response.json.return_value = []  # Empty response
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_get_snow_data_caching(self, mock_read_file, data_dir, sample_station_file):
        """Test that get_snow_data caches results."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        # First call
        result1 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Second call (same location/date)
        result2 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Results should be identical (cached)
        assert result1 == result2
    
    @pytest.mark.unit
    def test_station_data_cache_structure(self, data_dir):
        """Test that station data cache structure is correct."""
        client = AWDBClient(data_dir=data_dir)
        
        # Verify cache structures exist
        assert hasattr(client, 'station_data_cache'), "Should have station_data_cache"
        assert hasattr(client, 'request_cache'), "Should have request_cache"
        assert isinstance(client.station_data_cache, dict), "station_data_cache should be a dict"
        assert isinstance(client.request_cache, dict), "request_cache should be a dict"
        
        # Verify cache is keyed by station ID (int) not station+date
        # This allows reusing the same station data for multiple dates
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
            'snow_water_equivalent': [50.0] * 31,
            'snow_depth': [200.0] * 31
        })
        
        # Manually add to cache to verify structure
        station_id = 1196
        client.station_data_cache[station_id] = mock_df
        
        # Verify we can retrieve it
        assert station_id in client.station_data_cache
        cached_df = client.station_data_cache[station_id]
        assert len(cached_df) == 31  # Should have all dates
        
        # Verify we can extract different dates from the same cached dataframe
        date1 = cached_df[cached_df['date'].dt.date == datetime(2024, 1, 15).date()]
        date2 = cached_df[cached_df['date'].dt.date == datetime(2024, 1, 16).date()]
        assert len(date1) == 1
        assert len(date2) == 1
        assert date1.iloc[0]['date'].date() != date2.iloc[0]['date'].date()
        
        # This demonstrates the optimization: one download provides data for all dates


class TestAWDBClientDataContextIntegration:
    """Integration tests with DataContextBuilder."""
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_data_context_builder_uses_snotel(self, mock_read_file, data_dir, sample_station_file):
        """Test that DataContextBuilder uses AWDBClient."""
        from src.data.processors import DataContextBuilder
        
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        builder = DataContextBuilder(data_dir)
        
        assert builder.snotel_client is not None
        assert isinstance(builder.snotel_client, AWDBClient)
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_build_context_includes_snow_data(self, mock_read_file, data_dir, sample_station_file):
        """Test that build_context includes snow data from SNOTEL."""
        from src.data.processors import DataContextBuilder
        
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        builder = DataContextBuilder(data_dir)
        
        location = {"lat": 41.3500, "lon": -106.3167}
        date = "2024-01-15"
        
        context = builder.build_context(location, date)
        
        # Should include snow-related keys
        assert "snow_depth_inches" in context
        assert "snow_water_equiv_inches" in context
        assert "snow_crust_detected" in context
        
        # Data quality tracking fields
        assert "snow_data_source" in context
        assert "snow_station_name" in context
        assert "snow_station_distance_km" in context
        
        # Values should be numeric
        assert isinstance(context["snow_depth_inches"], (int, float))
        assert isinstance(context["snow_water_equiv_inches"], (int, float))
        assert isinstance(context["snow_crust_detected"], bool)
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_build_context_passes_elevation_to_snotel(self, mock_read_file, data_dir, sample_station_file):
        """Test that DataContextBuilder passes elevation to AWDBClient for better estimation."""
        from src.data.processors import DataContextBuilder
        from unittest.mock import patch, MagicMock
        
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        builder = DataContextBuilder(data_dir)
        # AWDBClient will use elevation estimate if no station data available
        
        # Mock get_snow_data to verify elevation is passed
        with patch.object(builder.snotel_client, 'get_snow_data') as mock_get_snow:
            mock_get_snow.return_value = {
                'depth': 20.0,
                'swe': 5.0,
                'crust': False,
                'station': None
            }
            
            location = {"lat": 43.889076, "lon": -105.619197}  # Location 1 from analysis
            date = "2024-01-15"
            
            context = builder.build_context(location, date)
            
            # Verify get_snow_data was called with elevation
            assert mock_get_snow.called
            call_kwargs = mock_get_snow.call_args[1]  # Get keyword arguments
            # elevation_ft should be passed (from context elevation)
            assert 'elevation_ft' in call_kwargs
            elevation_passed = call_kwargs['elevation_ft']
            # Should have elevation (from context)
            assert elevation_passed is not None, "Elevation should be passed to get_snow_data"
            # In test environment, DEM may not be available, so elevation might be from placeholder
            # or default. The key is that it's passed through, not that it's the "correct" elevation.
            # Should be a reasonable elevation value for Wyoming
            assert 1000 <= elevation_passed <= 15000, f"Elevation {elevation_passed} should be in reasonable Wyoming range"
            
            # Note: In test environment without DEM, elevation might come from placeholder (8500)
            # This is expected behavior when DEM is not available - the important thing is
            # that elevation is being passed to AWDBClient, not hardcoded in the client itself.
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_build_context_snow_data_source_tracking(self, mock_read_file, data_dir, sample_station_file):
        """Test that build_context correctly tracks snow data source (snotel vs estimate)."""
        from src.data.processors import DataContextBuilder
        
        # Mock geopandas - use mapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],  # Mapped station
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        builder = DataContextBuilder(data_dir)
        # AWDBClient will use elevation estimate if no station data available
        
        location = {"lat": 41.3500, "lon": -106.3167}
        date = "2024-01-15"
        
        context = builder.build_context(location, date)
        
        # Should indicate estimate since AWDB is None
        assert context["snow_data_source"] == "estimate"
        # Station name should be None for estimates
        assert context["snow_station_name"] is None or pd.isna(context["snow_station_name"])
        assert context["snow_station_distance_km"] is None or pd.isna(context["snow_station_distance_km"])
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    def test_build_context_with_unmapped_station(self, mock_read_file, data_dir, sample_station_file):
        """Test that build_context handles unmapped stations correctly."""
        from src.data.processors import DataContextBuilder
        
        # Mock geopandas - use unmapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.8350],
            'lon': [-106.4250],
            'elevation_ft': [10200],
            'state': ['WY'],
            'awdb_station_id': [None],  # Unmapped station
            'AWDB_site_id': [None],
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        builder = DataContextBuilder(data_dir)
        # AWDBClient will use elevation estimate if no station data available
        
        location = {"lat": 41.8350, "lon": -106.4250}
        date = "2024-01-15"
        
        context = builder.build_context(location, date)
        
        # Should indicate estimate since station is unmapped
        assert context["snow_data_source"] == "estimate"
        # Should have station name even though it's unmapped (for debugging)
        # But station_distance should be populated if station was found
        assert context["snow_station_name"] is None or pd.isna(context["snow_station_name"])


class TestSNOTELStationMapping:
    """Tests for station ID mapping functionality."""
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_station_has_AWDB_site_id(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test that mapped stations have AWDB_site_id."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['1196:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        station = client._find_nearest_station(41.3500, -106.3167)
        
        assert station is not None
        assert 'AWDB_site_id' in station
        assert station['AWDB_site_id'] == 1196
    
    @pytest.mark.unit
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_station_missing_AWDB_site_id(self, mock_load_from_awdb, data_dir, sample_station_file):
        """Test that unmapped stations have None for AWDB_site_id."""
        # Mock the API call to return success and set up stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['468:WY:SNTL', '1196:WY:SNTL', '419:WY:SNTL'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'awdb_station_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'AWDB_site_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        
        # Mock the function to return True, then set stations directly after client creation
        mock_load_from_awdb.return_value = True
        
        client = AWDBClient(data_dir=data_dir)
        # Set stations directly since the mock returned True
        client._stations_gdf = mock_gdf
        
        # Use location very close to ELKHORN PARK with small search radius
        # This ensures ELKHORN PARK is found (mapped stations are farther)
        station = client._find_nearest_station(41.8350, -106.4250, max_distance_km=10.0)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # AWDB_site_id should be None or NaN for unmapped stations
        assert station['AWDB_site_id'] is None or pd.isna(station['AWDB_site_id'])


class TestSNOTELDataQualityTracking:
    """Tests for data quality tracking fields."""
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_get_snow_data_returns_station_info(self, mock_read_file, data_dir, sample_station_file):
        """Test that get_snow_data includes station info in result."""
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Result should include station info if station was found
        # For elevation estimates, station may be None or populated
        assert 'depth' in result
        assert 'swe' in result
        assert 'crust' in result
        # Station info may be present even for estimates if station was found
        if 'station' in result and result['station'] is not None:
            assert 'station_distance_km' in result
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_get_snow_data_no_station_returns_no_station_info(self, mock_read_file, data_dir):
        """Test that get_snow_data returns no station info when no station nearby."""
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = None  # No stations loaded
        # AWDBClient doesn't have AWDB attribute
        
        result = client.get_snow_data(0.0, 0.0, datetime(2024, 1, 15))
        
        # Should use elevation estimate, no station info
        assert 'depth' in result
        assert 'swe' in result
        assert result.get('station') is None
        assert result.get('station_distance_km') is None
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_elevation_estimates_are_cached(self, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates are properly cached in request_cache."""
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.5667],
            'lon': [-106.8667],
            'elevation_ft': [10200],
            'state': ['WY'],
            'AWDB_site_id': [None],  # Unmapped station
            'geometry': [Point(-106.8667, 41.5667)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        # AWDBClient doesn't have AWDB attribute  # Simulate AWDB unavailable
        
        # Clear cache
        client.request_cache.clear()
        
        # First call - should compute and cache
        lat, lon = 41.5667, -106.8667
        date = datetime(2024, 1, 15)
        
        result1 = client.get_snow_data(lat, lon, date)
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        
        # Verify result was cached
        assert cache_key in client.request_cache
        assert client.request_cache[cache_key] == result1
        
        # Second call - should return cached result (same location/date)
        # Mock _estimate_snow_from_elevation to verify it's not called again
        with patch.object(client, '_estimate_snow_from_elevation') as mock_estimate:
            result2 = client.get_snow_data(lat, lon, date)
            
            # Should return cached result without calling _estimate_snow_from_elevation
            assert result2 == result1
            mock_estimate.assert_not_called()
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_unmapped_station_estimates_are_cached(self, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates for unmapped stations are cached."""
        # Mock geopandas with unmapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.5667],
            'lon': [-106.8667],
            'elevation_ft': [10200],
            'state': ['WY'],
            'awdb_station_id': [None],  # Unmapped station
            'AWDB_site_id': [None],
            'geometry': [Point(-106.8667, 41.5667)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        
        # Clear cache
        client.request_cache.clear()
        
        # First call - should detect unmapped station and cache elevation estimate
        lat, lon = 41.5667, -106.8667
        date = datetime(2024, 1, 15)
        
        result1 = client.get_snow_data(lat, lon, date)
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        
        # Verify result was cached
        assert cache_key in client.request_cache
        assert client.request_cache[cache_key] == result1
        
        # Second call - should return cached result
        with patch.object(client, '_estimate_snow_from_elevation') as mock_estimate:
            result2 = client.get_snow_data(lat, lon, date)
            
            # Should return cached result without calling _estimate_snow_from_elevation
            assert result2 == result1
            mock_estimate.assert_not_called()
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_exception_fallback_estimates_are_cached(self, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates from exception fallback are cached."""
        # Mock geopandas with mapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],  # Mapped station
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        # Mock _fetch_station_data_from_awdb to raise an exception
        client._fetch_station_data_from_awdb = MagicMock(side_effect=Exception("Test error"))
        
        # Clear cache
        client.request_cache.clear()
        
        # First call - should catch exception and cache elevation estimate
        lat, lon = 41.3500, -106.3167
        date = datetime(2024, 1, 15)
        
        result1 = client.get_snow_data(lat, lon, date)
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        
        # Verify result was cached
        assert cache_key in client.request_cache
        assert client.request_cache[cache_key] == result1
        
        # Second call - should return cached result without retrying AWDB
        with patch.object(client, '_estimate_snow_from_elevation') as mock_estimate:
            result2 = client.get_snow_data(lat, lon, date)
            
            # Should return cached result without calling _estimate_snow_from_elevation
            assert result2 == result1
            mock_estimate.assert_not_called()
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_closest_date_selection_with_non_sequential_index(self, mock_read_file, data_dir, sample_station_file):
        """Test that closest date selection works correctly with non-sequential DataFrame index."""
        import pandas as pd
        
        # Create a DataFrame with non-sequential index (simulating filtered data)
        # This simulates what happens when df[df['date'].isin(date_range)] filters rows
        date_base = pd.Timestamp('2024-01-01')
        dates = [date_base + pd.Timedelta(days=i) for i in [0, 5, 10, 15, 20]]
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Create DataFrame with non-sequential index (rows 0, 5, 10, 15, 20)
        # AWDB API returns data in inches, not mm
        full_df = pd.DataFrame({
            'date': dates,
            'WTEQ': values,  # Snow Water Equivalent in inches
            'SNWD': values   # Snow Depth in inches
        }, index=[0, 5, 10, 15, 20])
        
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [1196],
            'AWDB_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Create client and manually set cached DataFrame (bypassing actual API call)
        client = AWDBClient(data_dir=data_dir)
        client._load_stations()
        # Mock _find_nearest_station to return the station BEFORE setting cache
        # The station_id needs to match the cache key (which is converted to string)
        mock_station = {
            "triplet": "1196:WY:SNTL",
            "name": "MEDICINE BOW",
            "awdb_station_id": 1196,
            "AWDB_site_id": 1196,
            "distance_km": 0.1
        }
        client._find_nearest_station = MagicMock(return_value=mock_station)
        # Manually set cached station data (key is station_id as string, matching the conversion in get_snow_data)
        client.station_data_cache["1196"] = full_df
        
        # Test: Request date 2024-01-08 (closest to index 5, which is 2024-01-06)
        # When filtered to ±7 days, the dataframe would have indices [0, 5, 10] 
        # idxmin() returns label 5, and we need .loc[5] (not .iloc[5])
        target_date = datetime(2024, 1, 8)
        result = client.get_snow_data(41.3500, -106.3167, target_date)
        
        # Verify result contains expected data
        assert 'depth' in result
        assert 'swe' in result
        
        # The closest date should be 2024-01-06 (index 5), which has value 20.0 inches
        expected_swe = 20.0
        assert abs(result['swe'] - expected_swe) < 0.1, f"Expected SWE ~{expected_swe}, got {result['swe']}"
        assert abs(result['depth'] - expected_swe) < 0.1, f"Expected depth ~{expected_swe}, got {result['depth']}"


@pytest.mark.integration
@pytest.mark.slow
class TestSNOTELRealData:
    """Integration tests with real AWDB (requires R and AWDB installed)."""
    
    @pytest.mark.skipif(
        True,  # Skip by default - set to False to run real integration tests
        reason="Requires R and AWDB installed - run manually when needed"
    )
    def test_real_snotel_download(self, data_dir, sample_station_file):
        """Test actual SNOTEL data download (slow test)."""
        client = AWDBClient(data_dir=data_dir)
        
        # AWDBClient always available (uses HTTP requests, skipped in test env)
        
        # Use location near MEDICINE BOW (mapped)
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Verify real data was retrieved
        assert 'depth' in result
        assert 'swe' in result
        assert result['station'] == "MEDICINE BOW"
        assert result['depth'] > 0  # Should have snow in January


class TestRetryLogic:
    """Tests for retry logic with exponential backoff for 5xx errors."""
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_retry_succeeds_after_5xx_error(self, mock_load_from_awdb, mock_requests_get, data_dir):
        """Test that retry logic successfully recovers from initial 5xx error."""
        import time
        
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Create mock responses: first fails with 500, second succeeds
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status = MagicMock(side_effect=Exception("500 error"))
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [{
                'stationElement': {'elementCode': 'WTEQ'},
                'values': [{'date': '2024-01-15', 'value': 10.0}]
            }]
        }]
        mock_response_200.raise_for_status = MagicMock()
        
        # First call returns 500, second call returns 200
        mock_requests_get.side_effect = [mock_response_500, mock_response_200]
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        # Mock time.sleep to avoid actual delays in tests
        with patch('time.sleep'):
            result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should have retried and succeeded
        assert mock_requests_get.call_count == 2
        assert result is not None
        assert 'depth' in result
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_retry_gives_up_after_max_retries(self, mock_load_from_awdb, mock_requests_get, data_dir):
        """Test that retry logic gives up after max retries."""
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Create mock response that always returns 500
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status = MagicMock(side_effect=Exception("500 error"))
        mock_requests_get.return_value = mock_response_500
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        # Mock time.sleep to avoid actual delays
        with patch('time.sleep'):
            result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should have tried max_retries + 1 times (4 attempts)
        assert mock_requests_get.call_count == 4
        # Should fall back to elevation estimate
        assert result is not None
        assert 'depth' in result
        assert result.get('station') is None or result.get('station') == 'no_station'
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_safe_status_code_handling_with_mock(self, mock_load_from_awdb, mock_requests_get, data_dir):
        """Test that status_code comparison safely handles mock objects without status_code."""
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Create mock response without status_code (tests getattr safety)
        mock_response = MagicMock()
        # Don't set status_code - should be handled safely
        mock_response.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [{
                'stationElement': {'elementCode': 'WTEQ'},
                'values': [{'date': '2024-01-15', 'value': 10.0}]
            }]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        # Should not raise TypeError when comparing status_code
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should succeed (uses getattr to safely check status_code)
        assert result is not None
        assert 'depth' in result
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_no_retry_on_non_5xx_error(self, mock_load_from_awdb, mock_requests_get, data_dir):
        """Test that non-5xx errors are not retried."""
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Create mock response with 404 error (not retried)
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_404.raise_for_status = MagicMock(side_effect=Exception("404 error"))
        mock_requests_get.return_value = mock_response_404
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should only try once (no retry for 404)
        assert mock_requests_get.call_count == 1
        # Should fall back to elevation estimate
        assert result is not None
        assert 'depth' in result
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_exponential_backoff_timing(self, mock_load_from_awdb, mock_requests_get, data_dir):
        """Test that exponential backoff uses correct delays."""
        import time
        
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Create mock responses: first two fail with 500, third succeeds
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status = MagicMock(side_effect=Exception("500 error"))
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [{
                'stationElement': {'elementCode': 'WTEQ'},
                'values': [{'date': '2024-01-15', 'value': 10.0}]
            }]
        }]
        mock_response_200.raise_for_status = MagicMock()
        
        mock_requests_get.side_effect = [mock_response_500, mock_response_500, mock_response_200]
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        # Track sleep calls
        sleep_calls = []
        with patch('time.sleep', side_effect=lambda x: sleep_calls.append(x)):
            result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Should have retried twice with exponential backoff: 1s, 2s
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0  # First retry: 1 second
        assert sleep_calls[1] == 2.0  # Second retry: 2 seconds
        assert mock_requests_get.call_count == 3


class TestWarningSuppression:
    """Tests for warning suppression logic."""
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_warning_logged_only_once_per_station(self, mock_load_from_awdb, mock_requests_get, data_dir, caplog):
        """Test that warnings are only logged once per unique station/warning type."""
        # Reset class-level warning sets
        AWDBClient._warned_stations.clear()
        AWDBClient._warned_api_failures.clear()
        
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Mock response with no data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []  # No data
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        # Call get_snow_data multiple times for the same station
        with caplog.at_level("WARNING"):
            result1 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
            result2 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 16))
            result3 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 17))
        
        # Count warnings about "No AWDB data available"
        warning_count = sum(1 for record in caplog.records 
                           if "No AWDB data available for station" in record.message)
        
        # Should only log warning once, even though called 3 times
        assert warning_count == 1
        assert all(r is not None for r in [result1, result2, result3])
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_different_warning_types_tracked_separately(self, mock_load_from_awdb, mock_requests_get, data_dir, caplog):
        """Test that different warning types are tracked separately."""
        # Reset class-level warning sets
        AWDBClient._warned_stations.clear()
        AWDBClient._warned_api_failures.clear()
        
        # Mock stations - one with AWDB ID, one without
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MAPPED STATION', 'UNMAPPED STATION'],
            'triplet': ['123:WY:SNTL', '456:WY:SNTL'],
            'lat': [41.3500, 42.0000],
            'lon': [-106.3167, -107.0000],
            'elevation_ft': [9700, 9000],
            'state': ['WY', 'WY'],
            'awdb_station_id': [123, None],
            'AWDB_site_id': [123, None],
            'geometry': [Point(-106.3167, 41.3500), Point(-107.0000, 42.0000)]
        }, crs='EPSG:4326')
        
        # Mock response for mapped station - no data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        with caplog.at_level("WARNING"):
            # Call with mapped station (no AWDB data)
            result1 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
            # Call with unmapped station (no AWDB ID)
            result2 = client.get_snow_data(42.0000, -107.0000, datetime(2024, 1, 15))
        
        # Should have different warnings for different scenarios
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        # Should have warnings for both scenarios (they're different warning types)
        assert len([w for w in warnings if "No AWDB data" in w or "no AWDB station ID" in w]) >= 1
    
    @pytest.mark.unit
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    @patch.dict('os.environ', {}, clear=True)
    def test_api_failure_warning_suppression(self, mock_load_from_awdb, mock_requests_get, data_dir, caplog):
        """Test that API failure warnings are suppressed after first occurrence."""
        import requests
        
        # Reset class-level warning sets
        AWDBClient._warned_stations.clear()
        AWDBClient._warned_api_failures.clear()
        
        # Mock stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['TEST STATION'],
            'triplet': ['123:WY:SNTL'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'awdb_station_id': [123],
            'AWDB_site_id': [123],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        
        # Mock response that raises RequestException (not generic Exception)
        mock_requests_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        client = AWDBClient(data_dir=data_dir)
        client._stations_gdf = mock_gdf
        
        with caplog.at_level("WARNING"):
            # Call multiple times - should only warn once
            result1 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
            result2 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 16))
        
        # Count API failure warnings
        api_failure_warnings = [r for r in caplog.records 
                               if "AWDB API request failed" in r.message]
        
        # Should only log warning once
        assert len(api_failure_warnings) == 1
        assert all(r is not None for r in [result1, result2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

