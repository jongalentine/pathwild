"""
Unit and integration tests for SNOTEL data integration.

Tests SNOTELClient with snotelr R package integration, including:
- Station loading and mapping
- Data retrieval from snotelr
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

from src.data.processors import SNOTELClient


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory structure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    return tmp_path


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
                "snotelr_site_id": 1196
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
                "snotelr_site_id": 419
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
                "snotelr_site_id": None  # Unmapped station
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




class TestSNOTELClientUnit:
    """Unit tests for SNOTELClient."""
    
    @pytest.mark.unit
    def test_init_with_data_dir(self, data_dir):
        """Test SNOTELClient initialization with data_dir."""
        client = SNOTELClient(data_dir=data_dir)
        assert client.data_dir == data_dir
        assert client.station_cache_path == data_dir / "cache" / "snotel_stations_wyoming.geojson"
    
    @pytest.mark.unit
    def test_init_default_data_dir(self):
        """Test SNOTELClient initialization with default data_dir."""
        client = SNOTELClient()
        assert client.data_dir == Path("data")
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_load_stations_file_exists(self, mock_read_file, data_dir, sample_station_file):
        """Test loading stations when file exists."""
        # Mock geopandas to avoid sandbox issues
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW', 'COTTONWOOD CREEK', 'ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:975', 'SNOTEL:WY:964', 'SNOTEL:WY:967'],
            'lat': [41.3500, 44.5500, 41.8350],
            'lon': [-106.3167, -109.2833, -106.4250],
            'elevation_ft': [9700, 8840, 10200],
            'state': ['WY', 'WY', 'WY'],
            'snotelr_site_id': [1196, 419, None],
            'geometry': [Point(-106.3167, 41.3500), Point(-109.2833, 44.5500), 
                        Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = SNOTELClient(data_dir=data_dir)
        # Force reload
        client._load_stations()
        
        assert client._stations_gdf is not None
        assert len(client._stations_gdf) == 3
        assert "MEDICINE BOW" in client._stations_gdf['name'].values
    
    @pytest.mark.unit
    def test_load_stations_file_missing(self, data_dir):
        """Test loading stations when file doesn't exist."""
        client = SNOTELClient(data_dir=data_dir)
        # Should not raise error, just set to None
        assert client._stations_gdf is None
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    def test_find_nearest_station_mapped(self, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test finding nearest station with snotelr_site_id."""
        # Mock geopandas to avoid sandbox issues
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW', 'COTTONWOOD CREEK', 'ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:975', 'SNOTEL:WY:964', 'SNOTEL:WY:967'],
            'lat': [41.3500, 44.5500, 41.8350],
            'lon': [-106.3167, -109.2833, -106.4250],
            'elevation_ft': [9700, 8840, 10200],
            'state': ['WY', 'WY', 'WY'],
            'snotelr_site_id': [1196, 419, None],
            'geometry': [Point(-106.3167, 41.3500), Point(-109.2833, 44.5500), 
                        Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest to return first station
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]  # 0 meters = very close
        })
        mock_sjoin.return_value = mock_result
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location near MEDICINE BOW
        station = client._find_nearest_station(41.3500, -106.3167)
        
        assert station is not None
        assert station['name'] == "MEDICINE BOW"
        assert station['triplet'] == "SNOTEL:WY:975"
        assert station['snotelr_site_id'] == 1196
        assert station['distance_km'] < 1.0  # Should be very close
    
    @pytest.mark.unit
    def test_find_nearest_station_no_stations(self, data_dir):
        """Test finding nearest station when no stations loaded."""
        client = SNOTELClient(data_dir=data_dir)
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
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest to return empty (no stations within max_distance)
        mock_result = gpd.GeoDataFrame({
            'index_right': [None],  # No match
            'distance_m': [None]
        })
        mock_sjoin.return_value = mock_result
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location far from all stations (e.g., in ocean)
        station = client._find_nearest_station(0.0, 0.0, max_distance_km=10.0)
        assert station is None
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_find_nearest_station_prioritizes_mapped(self, mock_read_file, data_dir, sample_station_file):
        """Test that _find_nearest_station prioritizes mapped stations over unmapped ones."""
        # Mock geopandas - create scenario where unmapped station is closer but mapped is within range
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['SNOTEL:WY:967', 'SNOTEL:WY:975', 'SNOTEL:WY:964'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'snotelr_site_id': [None, 1196, 419],  # ELKHORN PARK unmapped, others mapped
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location between ELKHORN PARK (closer, unmapped) and MEDICINE BOW (farther, mapped)
        # Should prefer MEDICINE BOW (mapped) even though it's farther
        station = client._find_nearest_station(41.6, -106.37, max_distance_km=100.0)
        
        assert station is not None
        # Should prefer mapped station (MEDICINE BOW) over unmapped (ELKHORN PARK)
        assert station['name'] == "MEDICINE BOW"
        assert station['snotelr_site_id'] == 1196
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_find_nearest_station_unmapped_when_no_mapped(self, mock_read_file, data_dir, sample_station_file):
        """Test that _find_nearest_station uses unmapped station when no mapped stations available."""
        # Mock geopandas - only unmapped stations
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.8350],
            'lon': [-106.4250],
            'elevation_ft': [10200],
            'state': ['WY'],
            'snotelr_site_id': [None],  # Unmapped
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location near ELKHORN PARK (unmapped, but only station available)
        station = client._find_nearest_station(41.8350, -106.4250)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # snotelr_site_id should be None or NaN for unmapped stations
        import pandas as pd
        assert station['snotelr_site_id'] is None or pd.isna(station['snotelr_site_id'])
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    def test_find_nearest_station_unmapped(self, mock_read_file, data_dir, sample_station_file):
        """Test finding nearest station that isn't mapped when it's far from mapped stations."""
        # Mock geopandas - ELKHORN PARK is close, but mapped stations are very far away
        # Location near ELKHORN PARK should still find it if mapped stations are >100km away
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['SNOTEL:WY:967', 'SNOTEL:WY:975', 'SNOTEL:WY:964'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'snotelr_site_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Test location very close to ELKHORN PARK - should find it since mapped stations are farther
        # (Within 100km, ELKHORN PARK is closer than mapped stations)
        station = client._find_nearest_station(41.8350, -106.4250, max_distance_km=50.0)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # snotelr_site_id should be None or NaN for unmapped stations
        import pandas as pd
        assert station['snotelr_site_id'] is None or pd.isna(station['snotelr_site_id'])
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_winter(self):
        """Test elevation-based snow estimation for winter with actual elevation."""
        client = SNOTELClient()
        
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
        client = SNOTELClient()
        
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
        client = SNOTELClient()
        
        # Low elevation (1500 ft) in winter - should be 0 (below threshold)
        result = client._estimate_snow_from_elevation(41.0, -106.0, datetime(2024, 1, 15), elevation_ft=1500.0)
        
        # Winter formula: max(0, (1500 - 6000) / 100) = 0
        assert result['depth'] == 0.0
        assert result['swe'] == 0.0
    
    @pytest.mark.unit
    def test_estimate_snow_from_elevation_uses_provided_elevation(self):
        """Test that provided elevation parameter is used instead of default."""
        client = SNOTELClient()
        
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
        client = SNOTELClient()
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
        client = SNOTELClient()
        
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
        client = SNOTELClient()
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
    @patch('pathlib.Path.exists')
    @patch('rasterio.open')
    def test_estimate_snow_falls_back_to_dem(self, mock_rasterio_open, mock_exists, data_dir):
        """Test that elevation estimation attempts to sample from DEM when elevation not provided."""
        client = SNOTELClient(data_dir=data_dir)
        
        # Mock DEM exists
        mock_exists.return_value = True
        
        # Mock DEM sampling - return elevation in meters
        mock_dem = MagicMock()
        mock_sample = MagicMock()
        mock_sample.__iter__ = lambda self: iter([[2000.0]])  # 2000 meters = ~6562 feet
        mock_dem.sample.return_value = mock_sample
        mock_rasterio_open.return_value.__enter__ = MagicMock(return_value=mock_dem)
        mock_rasterio_open.return_value.__exit__ = MagicMock(return_value=False)
        
        # Call with no elevation provided
        result = client._estimate_snow_from_elevation(43.0, -110.0, datetime(2024, 1, 15), elevation_ft=None)
        
        # Should have attempted to sample DEM
        mock_exists.assert_called()
        
        # Result should use sampled elevation (6562 ft)
        # Winter: (6562 - 6000) / 100 = 5.62 inches
        assert result['depth'] > 0
        assert result['depth'] < 10  # Should be low elevation, minimal snow
    
    @pytest.mark.unit
    def test_estimate_snow_falls_back_to_default(self, data_dir, monkeypatch):
        """Test that elevation estimation falls back to default when DEM not available."""
        import pathlib
        
        client = SNOTELClient(data_dir=data_dir)
        
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
        client = SNOTELClient(data_dir=data_dir)
        # Force no stations by setting _stations_gdf to None
        client._stations_gdf = None
        client.snotelr = None  # Ensure snotelr is disabled
        
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
        client = SNOTELClient(data_dir=data_dir)
        # Force no stations by setting _stations_gdf to None (this makes _find_nearest_station return None)
        client._stations_gdf = None
        client.snotelr = None
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
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = None  # Disable snotelr
        
        # Use location near ELKHORN PARK (unmapped)
        result = client.get_snow_data(41.8350, -106.4250, datetime(2024, 1, 15))
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result


class TestSNOTELClientWithSnotelr:
    """Integration tests with mocked snotelr."""
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    def test_get_snow_data_with_snotelr_mapped(self, mock_sjoin, mock_read_file,
                                               data_dir, sample_station_file):
        """Test getting real SNOTEL data via snotelr for mapped station."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        # Setup mock snotelr
        mock_snotelr = MagicMock()
        mock_snotel_download = MagicMock()
        
        # Create expected pandas DataFrame
        expected_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-04-15', '2024-07-15']),
            'snow_water_equivalent': [373.0, 1018.5, 0.0],  # mm
            'snow_depth': [1849.0, 2410.0, 0.0],  # mm
        })
        
        # Create mock R data frame
        mock_r_data = MagicMock()
        mock_snotel_download.return_value = mock_r_data
        mock_snotelr.snotel_download = mock_snotel_download
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = mock_snotelr
        
        # Mock the rpy2 conversion inside get_snow_data
        # Since rpy2 may not be installed, mock via sys.modules
        import sys
        mock_rpy2 = MagicMock()
        mock_pandas2ri = MagicMock()
        mock_pandas2ri.rpy2py = MagicMock(return_value=expected_df)
        mock_localconverter = MagicMock()
        mock_localconverter.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_localconverter.return_value.__exit__ = MagicMock(return_value=False)
        
        mock_rpy2.robjects.pandas2ri = mock_pandas2ri
        mock_rpy2.robjects.conversion.localconverter = mock_localconverter
        
        # Temporarily add to sys.modules
        sys.modules['rpy2'] = mock_rpy2
        sys.modules['rpy2.robjects'] = mock_rpy2.robjects
        sys.modules['rpy2.robjects.pandas2ri'] = mock_pandas2ri
        sys.modules['rpy2.robjects.conversion'] = MagicMock()
        sys.modules['rpy2.robjects.conversion'].localconverter = mock_localconverter
        
        try:
            # Test location near MEDICINE BOW (mapped)
            result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        finally:
            # Clean up
            for key in ['rpy2', 'rpy2.robjects', 'rpy2.robjects.pandas2ri', 'rpy2.robjects.conversion']:
                if key in sys.modules:
                    del sys.modules[key]
        
        # Verify snotel_download was called
        assert mock_snotel_download.called
        call_args = mock_snotel_download.call_args
        assert call_args[0][0] == 1196  # site_id for MEDICINE BOW
        
        # Verify result structure
        assert 'depth' in result
        assert 'swe' in result
        assert 'crust' in result
        assert 'station' in result
        assert result['station'] == "MEDICINE BOW"
        
        # Verify values are converted from mm to inches
        # Expected: 1849.0 mm = ~72.8 inches, 373.0 mm SWE = ~14.7 inches
        assert result['depth'] > 0
        assert result['swe'] > 0
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_get_snow_data_with_snotelr_unmapped(self, mock_init_r, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test fallback when station isn't mapped."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.8350],
            'lon': [-106.4250],
            'elevation_ft': [10200],
            'state': ['WY'],
            'snotelr_site_id': [None],  # Unmapped
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        # Mock rpy2 initialization to avoid import issues
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = None  # Ensure snotelr is None so it uses fallback
        
        # Use location near ELKHORN PARK (unmapped)
        result = client.get_snow_data(41.8350, -106.4250, datetime(2024, 1, 15))
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result
        assert result.get('station') is None or 'ELKHORN' in result.get('station', '')
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_get_snow_data_no_data_for_date(self, mock_init_r, mock_sjoin, mock_read_file,
                                            data_dir, sample_station_file):
        """Test handling when snotelr returns no data for requested date."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        # Setup mock snotelr
        mock_snotelr = MagicMock()
        mock_snotel_download = MagicMock()
        mock_r_data = MagicMock()
        mock_snotel_download.return_value = mock_r_data
        mock_snotelr.snotel_download = mock_snotel_download
        
        # Create empty DataFrame (no data)
        empty_df = pd.DataFrame(columns=['date', 'snow_water_equivalent', 'snow_depth'])
        
        # Mock rpy2 initialization to avoid import issues
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = mock_snotelr
        
        # Mock the rpy2 conversion inside get_snow_data
        # Since rpy2 may not be installed, mock via sys.modules
        import sys
        mock_rpy2 = MagicMock()
        mock_pandas2ri = MagicMock()
        mock_pandas2ri.rpy2py = MagicMock(return_value=empty_df)
        mock_localconverter = MagicMock()
        mock_localconverter.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_localconverter.return_value.__exit__ = MagicMock(return_value=False)
        
        mock_rpy2.robjects.pandas2ri = mock_pandas2ri
        mock_rpy2.robjects.conversion.localconverter = mock_localconverter
        
        # Temporarily add to sys.modules
        sys.modules['rpy2'] = mock_rpy2
        sys.modules['rpy2.robjects'] = mock_rpy2.robjects
        sys.modules['rpy2.robjects.pandas2ri'] = mock_pandas2ri
        sys.modules['rpy2.robjects.conversion'] = MagicMock()
        sys.modules['rpy2.robjects.conversion'].localconverter = mock_localconverter
        
        try:
            result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        finally:
            # Clean up
            for key in ['rpy2', 'rpy2.robjects', 'rpy2.robjects.pandas2ri', 'rpy2.robjects.conversion']:
                if key in sys.modules:
                    del sys.modules[key]
        
        # Should fall back to elevation estimate
        assert 'depth' in result
        assert 'swe' in result
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_get_snow_data_caching(self, mock_init_r, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test that get_snow_data caches results."""
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        # Mock rpy2 initialization to avoid import issues
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = None  # Ensure it uses fallback
        
        # First call
        result1 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Second call (same location/date)
        result2 = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Results should be identical (cached)
        assert result1 == result2
    
    @pytest.mark.unit
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_station_data_cache_structure(self, mock_init_r, data_dir):
        """Test that station data cache structure is correct."""
        # Mock _init_r_snotelr to avoid rpy2 import issues
        mock_init_r.return_value = None
        client = SNOTELClient(data_dir=data_dir)
        
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


class TestSNOTELClientDataContextIntegration:
    """Integration tests with DataContextBuilder."""
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_data_context_builder_uses_snotel(self, mock_init_r, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test that DataContextBuilder uses SNOTELClient."""
        from src.data.processors import DataContextBuilder
        
        # Mock geopandas for station loading
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        builder = DataContextBuilder(data_dir)
        
        assert builder.snotel_client is not None
        assert isinstance(builder.snotel_client, SNOTELClient)
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_build_context_includes_snow_data(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
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
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization to avoid import issues
        mock_init_r.return_value = None
        
        builder = DataContextBuilder(data_dir)
        
        # Ensure snotelr is set (since we mocked _init_r_snotelr)
        if not hasattr(builder.snotel_client, 'snotelr'):
            builder.snotel_client.snotelr = None
            builder.snotel_client.ro = None
            builder.snotel_client._r_initialized = False
        
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
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_build_context_passes_elevation_to_snotel(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that DataContextBuilder passes elevation to SNOTELClient for better estimation."""
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
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        builder = DataContextBuilder(data_dir)
        builder.snotel_client.snotelr = None  # Force elevation estimate
        
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
            # that elevation is being passed to SNOTELClient, not hardcoded in the client itself.
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_build_context_snow_data_source_tracking(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
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
            'snotelr_site_id': [1196],  # Mapped station
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        builder = DataContextBuilder(data_dir)
        builder.snotel_client.snotelr = None  # Force elevation estimate
        
        location = {"lat": 41.3500, "lon": -106.3167}
        date = "2024-01-15"
        
        context = builder.build_context(location, date)
        
        # Should indicate estimate since snotelr is None
        assert context["snow_data_source"] == "estimate"
        # Station name should be None for estimates
        assert context["snow_station_name"] is None or pd.isna(context["snow_station_name"])
        assert context["snow_station_distance_km"] is None or pd.isna(context["snow_station_distance_km"])
    
    @pytest.mark.integration
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_build_context_with_unmapped_station(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
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
            'snotelr_site_id': [None],  # Unmapped station
            'geometry': [Point(-106.4250, 41.8350)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        builder = DataContextBuilder(data_dir)
        builder.snotel_client.snotelr = None  # Force elevation estimate
        
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
    @patch('geopandas.read_file')
    @patch('geopandas.sjoin_nearest')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_station_has_snotelr_site_id(self, mock_init_r, mock_sjoin, mock_read_file, data_dir, sample_station_file):
        """Test that mapped stations have snotelr_site_id."""
        # Mock geopandas to avoid sandbox issues
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock sjoin_nearest
        mock_result = gpd.GeoDataFrame({
            'index_right': [0],
            'distance_m': [0.0]
        })
        mock_sjoin.return_value = mock_result
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        station = client._find_nearest_station(41.3500, -106.3167)
        
        assert station is not None
        assert 'snotelr_site_id' in station
        assert station['snotelr_site_id'] == 1196
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_station_missing_snotelr_site_id(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that unmapped stations have None for snotelr_site_id."""
        # Mock geopandas - only unmapped station within range
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK', 'MEDICINE BOW', 'COTTONWOOD CREEK'],
            'triplet': ['SNOTEL:WY:967', 'SNOTEL:WY:975', 'SNOTEL:WY:964'],
            'lat': [41.8350, 41.3500, 44.5500],
            'lon': [-106.4250, -106.3167, -109.2833],
            'elevation_ft': [10200, 9700, 8840],
            'state': ['WY', 'WY', 'WY'],
            'snotelr_site_id': [None, 1196, 419],  # ELKHORN PARK unmapped
            'geometry': [Point(-106.4250, 41.8350), Point(-106.3167, 41.3500), 
                        Point(-109.2833, 44.5500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization to avoid import issues
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client._load_stations()
        
        # Use location very close to ELKHORN PARK with small search radius
        # This ensures ELKHORN PARK is found (mapped stations are farther)
        station = client._find_nearest_station(41.8350, -106.4250, max_distance_km=10.0)
        
        assert station is not None
        assert station['name'] == "ELKHORN PARK"
        # snotelr_site_id should be None or NaN for unmapped stations
        assert station['snotelr_site_id'] is None or pd.isna(station['snotelr_site_id'])


class TestSNOTELDataQualityTracking:
    """Tests for data quality tracking fields."""
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_get_snow_data_returns_station_info(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that get_snow_data includes station info in result."""
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = None  # Use elevation estimate
        
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
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_get_snow_data_no_station_returns_no_station_info(self, mock_init_r, mock_read_file, data_dir):
        """Test that get_snow_data returns no station info when no station nearby."""
        client = SNOTELClient(data_dir=data_dir)
        client._stations_gdf = None  # No stations loaded
        client.snotelr = None
        
        result = client.get_snow_data(0.0, 0.0, datetime(2024, 1, 15))
        
        # Should use elevation estimate, no station info
        assert 'depth' in result
        assert 'swe' in result
        assert result.get('station') is None
        assert result.get('station_distance_km') is None
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_elevation_estimates_are_cached(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates are properly cached in request_cache."""
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.5667],
            'lon': [-106.8667],
            'elevation_ft': [10200],
            'state': ['WY'],
            'snotelr_site_id': [None],  # Unmapped station
            'geometry': [Point(-106.8667, 41.5667)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = None  # Simulate snotelr unavailable
        
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
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_unmapped_station_estimates_are_cached(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates for unmapped stations are cached."""
        # Mock geopandas with unmapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['ELKHORN PARK'],
            'triplet': ['SNOTEL:WY:967'],
            'lat': [41.5667],
            'lon': [-106.8667],
            'elevation_ft': [10200],
            'state': ['WY'],
            'snotelr_site_id': [None],  # Unmapped station
            'geometry': [Point(-106.8667, 41.5667)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        # Mock snotelr as available (but station is unmapped)
        mock_snotelr = MagicMock()
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = mock_snotelr  # snotelr available but station unmapped
        
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
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_exception_fallback_estimates_are_cached(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that elevation estimates from exception fallback are cached."""
        # Mock geopandas with mapped station
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],  # Mapped station
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        # Mock snotelr to raise an exception
        mock_snotelr = MagicMock()
        mock_snotelr.snotel_download.side_effect = Exception("Test error")
        
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = mock_snotelr
        
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
        
        # Second call - should return cached result without retrying snotelr
        with patch.object(client, '_estimate_snow_from_elevation') as mock_estimate:
            result2 = client.get_snow_data(lat, lon, date)
            
            # Should return cached result without calling _estimate_snow_from_elevation
            assert result2 == result1
            mock_estimate.assert_not_called()
    
    @pytest.mark.unit
    @patch('geopandas.read_file')
    @patch.object(SNOTELClient, '_init_r_snotelr')
    def test_closest_date_selection_with_non_sequential_index(self, mock_init_r, mock_read_file, data_dir, sample_station_file):
        """Test that closest date selection works correctly with non-sequential DataFrame index."""
        import pandas as pd
        
        # Create a DataFrame with non-sequential index (simulating filtered data)
        # This simulates what happens when df[df['date'].isin(date_range)] filters rows
        date_base = pd.Timestamp('2024-01-01')
        dates = [date_base + pd.Timedelta(days=i) for i in [0, 5, 10, 15, 20]]
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Create DataFrame with non-sequential index (rows 0, 5, 10, 15, 20)
        full_df = pd.DataFrame({
            'date': dates,
            'snow_water_equivalent': [v * 25.4 for v in values],  # mm
            'snow_depth': [v * 25.4 for v in values]  # mm
        }, index=[0, 5, 10, 15, 20])
        
        # Mock geopandas
        mock_gdf = gpd.GeoDataFrame({
            'name': ['MEDICINE BOW'],
            'triplet': ['SNOTEL:WY:975'],
            'lat': [41.3500],
            'lon': [-106.3167],
            'elevation_ft': [9700],
            'state': ['WY'],
            'snotelr_site_id': [1196],
            'geometry': [Point(-106.3167, 41.3500)]
        }, crs='EPSG:4326')
        mock_read_file.return_value = mock_gdf
        
        # Mock rpy2 initialization
        mock_init_r.return_value = None
        
        # Create client and manually set cached DataFrame (bypassing actual R call)
        client = SNOTELClient(data_dir=data_dir)
        client.snotelr = MagicMock()  # Mock snotelr (won't be called due to cache)
        client.station_data_cache[1196] = full_df
        
        # Test: Request date 2024-01-08 (closest to index 5, which is 2024-01-06)
        # When filtered to ±7 days, the dataframe would have indices [0, 5, 10] 
        # idxmin() returns label 5, and we need .loc[5] (not .iloc[5])
        target_date = datetime(2024, 1, 8)
        result = client.get_snow_data(41.3500, -106.3167, target_date)
        
        # Verify result contains expected data
        assert 'depth' in result
        assert 'swe' in result
        
        # The closest date should be 2024-01-06 (index 5), which has value 20.0
        # Converted from mm to inches: 20.0 * 25.4 / 25.4 = 20.0 inches
        expected_swe = 20.0
        assert abs(result['swe'] - expected_swe) < 0.1, f"Expected SWE ~{expected_swe}, got {result['swe']}"
        assert abs(result['depth'] - expected_swe) < 0.1, f"Expected depth ~{expected_swe}, got {result['depth']}"


@pytest.mark.integration
@pytest.mark.slow
class TestSNOTELRealData:
    """Integration tests with real snotelr (requires R and snotelr installed)."""
    
    @pytest.mark.skipif(
        True,  # Skip by default - set to False to run real integration tests
        reason="Requires R and snotelr installed - run manually when needed"
    )
    def test_real_snotel_download(self, data_dir, sample_station_file):
        """Test actual SNOTEL data download (slow test)."""
        client = SNOTELClient(data_dir=data_dir)
        
        if client.snotelr is None:
            pytest.skip("snotelr not available")
        
        # Use location near MEDICINE BOW (mapped)
        result = client.get_snow_data(41.3500, -106.3167, datetime(2024, 1, 15))
        
        # Verify real data was retrieved
        assert 'depth' in result
        assert 'swe' in result
        assert result['station'] == "MEDICINE BOW"
        assert result['depth'] > 0  # Should have snow in January


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

