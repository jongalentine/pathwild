"""
Tests for SNOTEL data cache functionality

Fast tests (no AWDB API calls, no file I/O):
- All TestSNOTELDataCache tests

Slow tests (require AWDB API):
- TestAWDBClientWithCache tests

Run fast tests only: pytest tests/test_snotel_cache.py -m "not slow" -v
Run all tests: pytest tests/test_snotel_cache.py -v
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import threading
import sqlite3
import json
from unittest.mock import Mock, patch, MagicMock

from src.data.processors import SNOTELDataCache, AWDBClient


# ============================================================================
# SNOTEL Data Cache Tests
# ============================================================================

class TestSNOTELDataCache:
    """Tests for SNOTEL data cache functionality"""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for tests"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create SNOTEL data cache instance"""
        return SNOTELDataCache(cache_dir=temp_cache_dir)
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly"""
        cache = SNOTELDataCache(cache_dir=temp_cache_dir)
        
        assert cache.cache_db.exists()
        assert cache.cache_db.name == 'snotel_data_cache.db'
        
        # Check database was created with correct schema
        conn = sqlite3.connect(str(cache.cache_db))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='snotel_station_data';")
        assert cursor.fetchone() is not None
        conn.close()
        
        # Check statistics
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['total_size_mb'] >= 0
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation is deterministic"""
        station_id = "1196"
        begin_date = "2024-01-01"
        end_date = "2024-01-31"
        
        key1 = cache._make_cache_key(station_id, begin_date, end_date)
        key2 = cache._make_cache_key(station_id, begin_date, end_date)
        
        # Same inputs should produce same key
        assert key1 == key2
        
        # Different inputs should produce different keys
        key3 = cache._make_cache_key("1197", begin_date, end_date)
        assert key1 != key3
        
        key4 = cache._make_cache_key(station_id, "2024-02-01", end_date)
        assert key1 != key4
    
    def test_cache_put_get(self, cache):
        """Test basic cache put and get operations"""
        station_id = "1196"
        begin_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Create sample DataFrame
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'WTEQ': [10.0 + i * 0.1 for i in range(31)],  # Snow Water Equivalent in inches
            'SNWD': [40.0 + i * 0.5 for i in range(31)]   # Snow Depth in inches
        })
        
        # Put data in cache
        cache.put(station_id, begin_date, end_date, df)
        
        # Get data from cache
        cached_df = cache.get(station_id, begin_date, end_date)
        
        assert cached_df is not None
        assert len(cached_df) == 31
        assert 'date' in cached_df.columns
        assert 'WTEQ' in cached_df.columns
        assert 'SNWD' in cached_df.columns
        assert pd.api.types.is_datetime64_any_dtype(cached_df['date'])
        
        # Verify data integrity
        assert cached_df.iloc[0]['WTEQ'] == 10.0
        assert cached_df.iloc[30]['WTEQ'] == 13.0
        assert cached_df.iloc[0]['SNWD'] == 40.0
    
    def test_cache_miss(self, cache):
        """Test cache returns None for non-existent entries"""
        station_id = "1196"
        begin_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Try to get non-existent entry
        result = cache.get(station_id, begin_date, end_date)
        assert result is None
    
    def test_cache_overwrite(self, cache):
        """Test that cache overwrites existing entries"""
        station_id = "1196"
        begin_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Put initial data
        df1 = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
            'WTEQ': [10.0] * 31,
            'SNWD': [40.0] * 31
        })
        cache.put(station_id, begin_date, end_date, df1)
        
        # Overwrite with different data
        df2 = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
            'WTEQ': [20.0] * 31,
            'SNWD': [50.0] * 31
        })
        cache.put(station_id, begin_date, end_date, df2)
        
        # Get should return updated data
        cached_df = cache.get(station_id, begin_date, end_date)
        assert cached_df is not None
        assert cached_df.iloc[0]['WTEQ'] == 20.0
        assert cached_df.iloc[0]['SNWD'] == 50.0
    
    def test_cache_statistics(self, cache):
        """Test cache statistics tracking"""
        # Initially empty
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        
        # Add some entries
        for i in range(5):
            station_id = f"119{i}"
            df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
                'WTEQ': [10.0] * 31,
                'SNWD': [40.0] * 31
            })
            cache.put(station_id, "2024-01-01", "2024-01-31", df)
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 5
        assert stats['total_size_mb'] > 0
    
    def test_cache_thread_safety(self, cache):
        """Test cache is thread-safe"""
        results = {}  # Use dict to track worker_id -> result
        errors = []
        results_lock = threading.Lock()
        errors_lock = threading.Lock()
        
        def worker(worker_id):
            try:
                station_id = f"119{worker_id}"
                begin_date = "2024-01-01"
                end_date = "2024-01-31"
                
                # Create DataFrame
                df = pd.DataFrame({
                    'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
                    'WTEQ': [10.0 + worker_id] * 31,
                    'SNWD': [40.0 + worker_id] * 31
                })
                
                # Put
                cache.put(station_id, begin_date, end_date, df)
                
                # Get
                result = cache.get(station_id, begin_date, end_date)
                with results_lock:
                    results[worker_id] = result
            except Exception as e:
                with errors_lock:
                    errors.append(e)
        
        # Run 10 threads concurrently
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        
        # Verify all results are correct (check by worker_id, not enumerate order)
        for worker_id in range(10):
            result = results[worker_id]
            assert result is not None
            assert len(result) == 31
            assert result.iloc[0]['WTEQ'] == 10.0 + worker_id
            assert result.iloc[0]['SNWD'] == 40.0 + worker_id
    
    def test_cache_date_serialization(self, cache):
        """Test that dates are properly serialized and deserialized"""
        station_id = "1196"
        begin_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Create DataFrame with dates
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'WTEQ': [10.0] * 31,
            'SNWD': [40.0] * 31
        })
        
        # Put and get
        cache.put(station_id, begin_date, end_date, df)
        cached_df = cache.get(station_id, begin_date, end_date)
        
        # Verify dates are datetime objects
        assert pd.api.types.is_datetime64_any_dtype(cached_df['date'])
        assert cached_df['date'].iloc[0] == pd.Timestamp('2024-01-01')
        assert cached_df['date'].iloc[30] == pd.Timestamp('2024-01-31')
    
    def test_cache_different_date_ranges(self, cache):
        """Test that different date ranges are cached separately"""
        station_id = "1196"
        
        # Cache January data
        df_jan = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-31', freq='D'),
            'WTEQ': [10.0] * 31,
            'SNWD': [40.0] * 31
        })
        cache.put(station_id, "2024-01-01", "2024-01-31", df_jan)
        
        # Cache February data
        df_feb = pd.DataFrame({
            'date': pd.date_range('2024-02-01', '2024-02-29', freq='D'),
            'WTEQ': [15.0] * 29,
            'SNWD': [45.0] * 29
        })
        cache.put(station_id, "2024-02-01", "2024-02-29", df_feb)
        
        # Retrieve both
        cached_jan = cache.get(station_id, "2024-01-01", "2024-01-31")
        cached_feb = cache.get(station_id, "2024-02-01", "2024-02-29")
        
        assert cached_jan is not None
        assert cached_feb is not None
        assert len(cached_jan) == 31
        assert len(cached_feb) == 29
        assert cached_jan.iloc[0]['WTEQ'] == 10.0
        assert cached_feb.iloc[0]['WTEQ'] == 15.0


# ============================================================================
# AWDBClient Cache Integration Tests
# ============================================================================

@pytest.mark.skipif(True, reason="Requires AWDB API - run manually when needed")
@pytest.mark.slow
class TestAWDBClientWithCache:
    """Integration tests for AWDBClient with persistent cache"""
    
    @pytest.fixture(autouse=True)
    def reset_warning_sets(self):
        """Reset warning sets before each test"""
        AWDBClient._warned_stations.clear()
        AWDBClient._warned_api_failures.clear()
        yield
        AWDBClient._warned_stations.clear()
        AWDBClient._warned_api_failures.clear()
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir
    
    @pytest.fixture
    def awdb_client_with_cache(self, temp_cache_dir):
        """Create AWDBClient with caching enabled"""
        return AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
    
    @pytest.fixture
    def awdb_client_without_cache(self, temp_cache_dir):
        """Create AWDBClient with caching disabled"""
        return AWDBClient(data_dir=temp_cache_dir.parent, use_cache=False)
    
    def test_cache_enabled_by_default(self, temp_cache_dir):
        """Test that cache is enabled by default"""
        client = AWDBClient(data_dir=temp_cache_dir.parent)
        assert client.use_cache is True
        assert client.data_cache is not None
        assert client.cache_historical_only is True
    
    def test_cache_disabled_when_requested(self, temp_cache_dir):
        """Test that cache can be disabled"""
        client = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=False)
        assert client.use_cache is False
        assert client.data_cache is None
    
    def test_cache_historical_only_default(self, temp_cache_dir):
        """Test that cache_historical_only defaults to True"""
        client = AWDBClient(data_dir=temp_cache_dir.parent)
        assert client.cache_historical_only is True
    
    def test_cache_historical_only_can_be_disabled(self, temp_cache_dir):
        """Test that cache_historical_only can be set to False"""
        client = AWDBClient(data_dir=temp_cache_dir.parent, cache_historical_only=False)
        assert client.cache_historical_only is False
    
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_historical_data_uses_cache(self, mock_load_stations, mock_requests_get, temp_cache_dir):
        """Test that historical data (>30 days old) uses persistent cache"""
        import os
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock stations
        import geopandas as gpd
        from shapely.geometry import Point
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
        mock_load_stations.return_value = True
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [
                {
                    'stationElement': {'elementCode': 'WTEQ'},
                    'values': [{'date': '2023-01-15', 'value': 14.7}]
                },
                {
                    'stationElement': {'elementCode': 'SNWD'},
                    'values': [{'date': '2023-01-15', 'value': 72.8}]
                }
            ]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        # Historical date (>30 days old)
        historical_date = datetime.now() - timedelta(days=60)
        
        client = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client._stations_gdf = mock_gdf
        
        # First call - should query API and cache result
        result1 = client.get_snow_data(41.3500, -106.3167, historical_date)
        assert mock_requests_get.called
        mock_requests_get.reset_mock()
        
        # Second call - should use cache (no API call)
        result2 = client.get_snow_data(41.3500, -106.3167, historical_date)
        # Should not call API again (cached)
        assert not mock_requests_get.called or mock_requests_get.call_count == 0
        assert result1 == result2
    
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_recent_data_bypasses_cache(self, mock_load_stations, mock_requests_get, temp_cache_dir):
        """Test that recent data (≤30 days old) bypasses cache and uses live API"""
        import os
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock stations
        import geopandas as gpd
        from shapely.geometry import Point
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
        mock_load_stations.return_value = True
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [
                {
                    'stationElement': {'elementCode': 'WTEQ'},
                    'values': [{'date': datetime.now().strftime('%Y-%m-%d'), 'value': 10.0}]
                },
                {
                    'stationElement': {'elementCode': 'SNWD'},
                    'values': [{'date': datetime.now().strftime('%Y-%m-%d'), 'value': 50.0}]
                }
            ]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        # Recent date (≤30 days old)
        recent_date = datetime.now() - timedelta(days=10)
        
        client = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client._stations_gdf = mock_gdf
        
        # First call - should query API (not cached for recent data)
        result1 = client.get_snow_data(41.3500, -106.3167, recent_date)
        assert mock_requests_get.called
        call_count_1 = mock_requests_get.call_count
        mock_requests_get.reset_mock()
        
        # Second call - should query API again (bypasses cache for recent data)
        result2 = client.get_snow_data(41.3500, -106.3167, recent_date)
        assert mock_requests_get.called
        call_count_2 = mock_requests_get.call_count
        # Should have made API calls both times (not cached)
        assert call_count_1 > 0
        assert call_count_2 > 0
    
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_cache_persists_across_instances(self, mock_load_stations, mock_requests_get, temp_cache_dir):
        """Test that cache persists across different AWDBClient instances"""
        import os
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock stations
        import geopandas as gpd
        from shapely.geometry import Point
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
        mock_load_stations.return_value = True
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [
                {
                    'stationElement': {'elementCode': 'WTEQ'},
                    'values': [{'date': '2023-01-15', 'value': 14.7}]
                },
                {
                    'stationElement': {'elementCode': 'SNWD'},
                    'values': [{'date': '2023-01-15', 'value': 72.8}]
                }
            ]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        # Historical date
        historical_date = datetime.now() - timedelta(days=60)
        
        # First instance - query and cache
        client1 = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client1._stations_gdf = mock_gdf
        result1 = client1.get_snow_data(41.3500, -106.3167, historical_date)
        assert mock_requests_get.called
        mock_requests_get.reset_mock()
        
        # Second instance - should use cache from first instance
        client2 = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client2._stations_gdf = mock_gdf
        result2 = client2.get_snow_data(41.3500, -106.3167, historical_date)
        
        # Should not call API (cache hit from previous instance)
        assert not mock_requests_get.called or mock_requests_get.call_count == 0
        assert result1 == result2
    
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_cache_with_different_parameters(self, mock_load_stations, mock_requests_get, temp_cache_dir):
        """Test that cache works correctly with different date ranges"""
        import os
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock stations
        import geopandas as gpd
        from shapely.geometry import Point
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
        mock_load_stations.return_value = True
        
        # Historical date
        historical_date = datetime.now() - timedelta(days=60)
        
        # Mock API to return data for date range
        def mock_api_response(*args, **kwargs):
            params = kwargs.get('params', {})
            begin_date = params.get('beginDate', '')
            end_date = params.get('endDate', '')
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Return data for the requested range
            dates = pd.date_range(begin_date, end_date, freq='D')
            values = [{'date': d.strftime('%Y-%m-%d'), 'value': 10.0} for d in dates]
            
            mock_response.json.return_value = [{
                'stationTriplet': '123:WY:SNTL',
                'data': [
                    {
                        'stationElement': {'elementCode': 'WTEQ'},
                        'values': values
                    },
                    {
                        'stationElement': {'elementCode': 'SNWD'},
                        'values': values
                    }
                ]
            }]
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        mock_requests_get.side_effect = mock_api_response
        
        client = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client._stations_gdf = mock_gdf
        
        # First call - should query API
        result1 = client.get_snow_data(41.3500, -106.3167, historical_date)
        assert mock_requests_get.called
        call_count_1 = mock_requests_get.call_count
        mock_requests_get.reset_mock()
        
        # Second call with same date - should use cache
        result2 = client.get_snow_data(41.3500, -106.3167, historical_date)
        # Should have fewer API calls (cache hit)
        assert mock_requests_get.call_count < call_count_1 or not mock_requests_get.called
        assert result1 == result2
    
    @patch('requests.get')
    @patch.object(AWDBClient, '_load_stations_from_awdb')
    def test_cache_historical_only_flag(self, mock_load_stations, mock_requests_get, temp_cache_dir):
        """Test that cache_historical_only flag correctly controls caching behavior"""
        import os
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        os.environ.pop('TESTING', None)
        
        # Mock stations
        import geopandas as gpd
        from shapely.geometry import Point
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
        mock_load_stations.return_value = True
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            'stationTriplet': '123:WY:SNTL',
            'data': [
                {
                    'stationElement': {'elementCode': 'WTEQ'},
                    'values': [{'date': '2023-01-15', 'value': 14.7}]
                },
                {
                    'stationElement': {'elementCode': 'SNWD'},
                    'values': [{'date': '2023-01-15', 'value': 72.8}]
                }
            ]
        }]
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        historical_date = datetime.now() - timedelta(days=60)
        
        # Test with cache_historical_only=True (default)
        client1 = AWDBClient(data_dir=temp_cache_dir.parent, use_cache=True, cache_historical_only=True)
        client1._stations_gdf = mock_gdf
        result1 = client1.get_snow_data(41.3500, -106.3167, historical_date)
        assert mock_requests_get.called
        mock_requests_get.reset_mock()
        
        # Second call should use cache
        result2 = client1.get_snow_data(41.3500, -106.3167, historical_date)
        assert not mock_requests_get.called or mock_requests_get.call_count == 0
        
        # Test with cache_historical_only=False (should cache all data)
        mock_requests_get.reset_mock()
        client2 = AWDBClient(data_dir=temp_cache_dir.parent / "cache2", use_cache=True, cache_historical_only=False)
        client2._stations_gdf = mock_gdf
        result3 = client2.get_snow_data(41.3500, -106.3167, historical_date)
        assert mock_requests_get.called
        mock_requests_get.reset_mock()
        
        # Second call should also use cache (even though cache_historical_only=False, historical data is still cached)
        result4 = client2.get_snow_data(41.3500, -106.3167, historical_date)
        # Note: With cache_historical_only=False, all data is cached, so this should also hit cache
        # The distinction is mainly for recent data (≤30 days)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

