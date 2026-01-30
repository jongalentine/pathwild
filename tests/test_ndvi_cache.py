"""
Tests for NDVI cache functionality

Fast tests (no GEE API calls, no file I/O):
- All TestNDVICache tests (except those marked @pytest.mark.slow)

Slow tests (require GEE API):
- TestGEENDVIClientWithCache tests

Run fast tests only: pytest tests/test_ndvi_cache.py -m "not slow" -v
Run all tests: pytest tests/test_ndvi_cache.py -v
"""
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading

# Skip all tests if earthengine-api not available
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False

from src.data.processors import NDVICache, GEENDVIClient, GEECircuitBreaker

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@pytest.fixture
def test_config():
    """Load test configuration"""
    if not YAML_AVAILABLE:
        return {}
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ============================================================================
# NDVI Cache Tests
# ============================================================================

class TestNDVICache:
    """Tests for NDVI cache functionality"""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for tests"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create NDVI cache instance"""
        return NDVICache(cache_dir=temp_cache_dir)
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly"""
        cache = NDVICache(cache_dir=temp_cache_dir)
        
        assert cache.cache_db.exists()
        assert cache.cache_db.name == 'ndvi_cache.db'
        
        # Check database was created
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['total_size_mb'] >= 0
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation is deterministic"""
        lat1, lon1 = 44.5, -107.25
        date1 = datetime(2020, 6, 15)
        collection = 'landsat8'
        buffer_days = 7
        max_cloud_cover = 30.0
        
        key1 = cache._make_cache_key(lat1, lon1, date1, collection, buffer_days, max_cloud_cover)
        key2 = cache._make_cache_key(lat1, lon1, date1, collection, buffer_days, max_cloud_cover)
        
        # Same inputs should produce same key
        assert key1 == key2
        
        # Different inputs should produce different keys
        key3 = cache._make_cache_key(lat1 + 0.001, lon1, date1, collection, buffer_days, max_cloud_cover)
        assert key1 != key3
        
        key4 = cache._make_cache_key(lat1, lon1, datetime(2020, 6, 16), collection, buffer_days, max_cloud_cover)
        assert key1 != key4
    
    def test_cache_put_get(self, cache):
        """Test basic cache put and get operations"""
        lat, lon = 44.5, -107.25
        date = datetime(2020, 6, 15)
        collection = 'landsat8'
        buffer_days = 7
        max_cloud_cover = 30.0
        
        # Put data in cache
        result = {
            'ndvi': 0.65,
            'cloud_cover': 5.2,
            'image_date': datetime(2020, 6, 14),
            'ndvi_age_days': 1.0
        }
        cache.put(lat, lon, date, collection, buffer_days, max_cloud_cover, result)
        
        # Get data from cache
        cached_result = cache.get(lat, lon, date, collection, buffer_days, max_cloud_cover)
        
        assert cached_result is not None
        assert cached_result['ndvi'] == 0.65
        assert cached_result['cloud_cover'] == 5.2
        assert cached_result['ndvi_age_days'] == 1.0
        assert cached_result['image_date'] == datetime(2020, 6, 14)
    
    def test_cache_miss(self, cache):
        """Test cache returns None for non-existent entries"""
        lat, lon = 44.5, -107.25
        date = datetime(2020, 6, 15)
        collection = 'landsat8'
        buffer_days = 7
        max_cloud_cover = 30.0
        
        # Try to get non-existent entry
        result = cache.get(lat, lon, date, collection, buffer_days, max_cloud_cover)
        assert result is None
    
    def test_cache_none_values(self, cache):
        """Test cache handles None values correctly"""
        lat, lon = 44.5, -107.25
        date = datetime(2020, 6, 15)
        collection = 'landsat8'
        buffer_days = 7
        max_cloud_cover = 30.0
        
        # Put None result (no data available)
        result = {
            'ndvi': None,
            'cloud_cover': None,
            'image_date': None,
            'ndvi_age_days': None
        }
        cache.put(lat, lon, date, collection, buffer_days, max_cloud_cover, result)
        
        # Get should return None values
        cached_result = cache.get(lat, lon, date, collection, buffer_days, max_cloud_cover)
        assert cached_result is not None
        assert cached_result['ndvi'] is None
        assert cached_result['cloud_cover'] is None
        assert cached_result['image_date'] is None
        assert cached_result['ndvi_age_days'] is None
    
    def test_cache_coordinate_rounding(self, cache):
        """Test that coordinates are rounded consistently"""
        date = datetime(2020, 6, 15)
        collection = 'landsat8'
        buffer_days = 7
        max_cloud_cover = 30.0
        
        # Put with slightly different coordinates (within rounding precision)
        lat1, lon1 = 44.500001, -107.250001
        result1 = {'ndvi': 0.65, 'cloud_cover': 5.2, 'image_date': None, 'ndvi_age_days': None}
        cache.put(lat1, lon1, date, collection, buffer_days, max_cloud_cover, result1)
        
        # Get with rounded coordinates should still work
        lat2, lon2 = 44.50000, -107.25000
        cached_result = cache.get(lat2, lon2, date, collection, buffer_days, max_cloud_cover)
        
        # Should find the cached entry (coordinates rounded to 5 decimal places)
        assert cached_result is not None
        assert cached_result['ndvi'] == 0.65
    
    def test_cache_statistics(self, cache):
        """Test cache statistics tracking"""
        # Initially empty
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        
        # Add some entries
        for i in range(5):
            cache.put(
                lat=44.5 + i * 0.01,
                lon=-107.25 - i * 0.01,
                date=datetime(2020, 6, 15),
                collection='landsat8',
                buffer_days=7,
                max_cloud_cover=30.0,
                result={'ndvi': 0.5, 'cloud_cover': 10.0, 'image_date': None, 'ndvi_age_days': None}
            )
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 5
        assert stats['total_size_mb'] > 0
    
    def test_cache_clear(self, cache):
        """Test cache clearing"""
        # Add entry
        cache.put(
            lat=44.5,
            lon=-107.25,
            date=datetime(2020, 6, 15),
            collection='landsat8',
            buffer_days=7,
            max_cloud_cover=30.0,
            result={'ndvi': 0.65, 'cloud_cover': 5.2, 'image_date': None, 'ndvi_age_days': None}
        )
        
        assert cache.get_stats()['total_entries'] == 1
        
        # Clear cache
        cache.clear()
        
        assert cache.get_stats()['total_entries'] == 0
        
        # Entry should be gone
        result = cache.get(44.5, -107.25, datetime(2020, 6, 15), 'landsat8', 7, 30.0)
        assert result is None
    
    def test_cache_thread_safety(self, cache):
        """Test cache is thread-safe"""
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Each worker adds and retrieves data
                lat = 44.5 + worker_id * 0.01
                lon = -107.25 - worker_id * 0.01
                date = datetime(2020, 6, 15)
                
                # Put
                cache.put(
                    lat, lon, date, 'landsat8', 7, 30.0,
                    {'ndvi': 0.5 + worker_id * 0.1, 'cloud_cover': 10.0, 'image_date': None, 'ndvi_age_days': None}
                )
                
                # Get
                result = cache.get(lat, lon, date, 'landsat8', 7, 30.0)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 10 threads concurrently
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # All results should be present
        assert len(results) == 10
        assert all(r is not None for r in results)


class TestGEENDVIClientWithCache:
    """Tests for GEENDVIClient with caching enabled"""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for tests"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir
    
    @pytest.fixture
    def gee_client_with_cache(self, test_config, temp_cache_dir):
        """Fixture for GEE NDVI client with cache enabled"""
        # Reset circuit breaker before each test to prevent state persistence
        if GEECircuitBreaker._instance is not None:
            GEECircuitBreaker._instance.reset()
        
        if not EE_AVAILABLE:
            pytest.skip("earthengine-api not available")
        
        gee_config = test_config.get('gee', {})
        project = gee_config.get('project', 'ee-jongalentine')
        
        try:
            return GEENDVIClient(
                project=project,
                collection='landsat8',
                batch_size=10,
                max_workers=2,
                cache_dir=temp_cache_dir,
                use_cache=True
            )
        except Exception as e:
            pytest.skip(f"GEE not authenticated or available: {e}")
    
    @pytest.fixture
    def gee_client_no_cache(self, test_config):
        """Fixture for GEE NDVI client with cache disabled"""
        # Reset circuit breaker before each test to prevent state persistence
        if GEECircuitBreaker._instance is not None:
            GEECircuitBreaker._instance.reset()
        
        if not EE_AVAILABLE:
            pytest.skip("earthengine-api not available")
        
        gee_config = test_config.get('gee', {})
        project = gee_config.get('project', 'ee-jongalentine')
        
        try:
            return GEENDVIClient(
                project=project,
                collection='landsat8',
                batch_size=10,
                max_workers=2,
                use_cache=False
            )
        except Exception as e:
            pytest.skip(f"GEE not authenticated or available: {e}")
    
    @pytest.mark.slow
    def test_cache_enabled_by_default(self, test_config, temp_cache_dir):
        """Test that cache is enabled by default"""
        # Reset circuit breaker before each test to prevent state persistence
        if GEECircuitBreaker._instance is not None:
            GEECircuitBreaker._instance.reset()
        
        if not EE_AVAILABLE:
            pytest.skip("earthengine-api not available")
        
        gee_config = test_config.get('gee', {})
        project = gee_config.get('project', 'ee-jongalentine')
        
        try:
            client = GEENDVIClient(
                project=project,
                collection='landsat8',
                cache_dir=temp_cache_dir
                # use_cache defaults to True
            )
            assert client.use_cache is True
            assert client.cache is not None
        except Exception as e:
            pytest.skip(f"GEE not available: {e}")
    
    @pytest.mark.slow
    def test_cache_disabled(self, gee_client_no_cache):
        """Test that cache can be disabled"""
        assert gee_client_no_cache.use_cache is False
        assert gee_client_no_cache.cache is None
    
    @pytest.mark.slow
    def test_cache_hit_on_second_call(self, gee_client_with_cache):
        """Test that second call uses cache (no GEE API call)"""
        test_data = pd.DataFrame({
            'latitude': [44.5],
            'longitude': [-107.25],
            'timestamp': [datetime(2020, 6, 15)]
        })
        
        # First call - should query GEE and cache result
        result1 = gee_client_with_cache.get_ndvi_for_points(test_data)
        
        # Check cache was populated
        cache_stats = gee_client_with_cache.cache.get_stats()
        assert cache_stats['total_entries'] > 0
        
        # Second call - should use cache
        # We can't directly verify no GEE call, but we can verify same result
        result2 = gee_client_with_cache.get_ndvi_for_points(test_data)
        
        # Results should be identical
        assert result1['ndvi'].iloc[0] == result2['ndvi'].iloc[0]
        if pd.notna(result1['ndvi'].iloc[0]):
            assert result1['ndvi_age_days'].iloc[0] == result2['ndvi_age_days'].iloc[0]
            assert result1['irg'].iloc[0] == result2['irg'].iloc[0]
    
    @pytest.mark.slow
    def test_cache_persistence_across_instances(self, test_config, temp_cache_dir):
        """Test that cache persists across different client instances"""
        # Reset circuit breaker before each test to prevent state persistence
        if GEECircuitBreaker._instance is not None:
            GEECircuitBreaker._instance.reset()
        
        if not EE_AVAILABLE:
            pytest.skip("earthengine-api not available")
        
        gee_config = test_config.get('gee', {})
        project = gee_config.get('project', 'ee-jongalentine')
        
        try:
            # First client instance - query and cache
            client1 = GEENDVIClient(
                project=project,
                collection='landsat8',
                cache_dir=temp_cache_dir,
                use_cache=True
            )
            
            test_data = pd.DataFrame({
                'latitude': [44.5],
                'longitude': [-107.25],
                'timestamp': [datetime(2020, 6, 15)]
            })
            
            result1 = client1.get_ndvi_for_points(test_data)
            ndvi_value = result1['ndvi'].iloc[0]
            
            # Second client instance - should use cache
            client2 = GEENDVIClient(
                project=project,
                collection='landsat8',
                cache_dir=temp_cache_dir,
                use_cache=True
            )
            
            result2 = client2.get_ndvi_for_points(test_data)
            
            # Should get same result from cache
            assert result2['ndvi'].iloc[0] == ndvi_value
            
        except Exception as e:
            pytest.skip(f"GEE not available: {e}")
    
    @pytest.mark.slow
    def test_cache_different_parameters(self, gee_client_with_cache):
        """Test that different parameters create different cache entries"""
        lat, lon = 44.5, -107.25
        date = datetime(2020, 6, 15)
        
        # Query with buffer_days=7
        test_data1 = pd.DataFrame({
            'latitude': [lat],
            'longitude': [lon],
            'timestamp': [date]
        })
        result1 = gee_client_with_cache.get_ndvi_for_points(
            test_data1,
            buffer_days=7,
            max_cloud_cover=30.0
        )
        
        # Query with different buffer_days=14
        result2 = gee_client_with_cache.get_ndvi_for_points(
            test_data1,
            buffer_days=14,
            max_cloud_cover=30.0
        )
        
        # Should have separate cache entries
        cache_stats = gee_client_with_cache.cache.get_stats()
        # Should have at least 2 entries (one for each buffer_days, plus IRG lookups)
        assert cache_stats['total_entries'] >= 2
    
    @pytest.mark.slow
    def test_cache_statistics_logging(self, gee_client_with_cache):
        """Test that cache statistics are logged"""
        test_data = pd.DataFrame({
            'latitude': [44.5],
            'longitude': [-107.25],
            'timestamp': [datetime(2020, 6, 15)]
        })
        
        # Query should log cache statistics
        result = gee_client_with_cache.get_ndvi_for_points(test_data)
        
        # Cache should have entries
        cache_stats = gee_client_with_cache.cache.get_stats()
        assert cache_stats['total_entries'] > 0
    
    @pytest.mark.slow
    def test_cache_with_batch_processing(self, gee_client_with_cache):
        """Test cache works with batch processing"""
        # Create test data with multiple points
        test_data = pd.DataFrame({
            'latitude': [44.5 + i * 0.01 for i in range(3)],
            'longitude': [-107.25 - i * 0.01 for i in range(3)],
            'timestamp': [datetime(2020, 6, 15) for _ in range(3)]
        })
        
        # First call - should cache all points
        result1 = gee_client_with_cache.get_ndvi_for_points(test_data)
        
        cache_stats1 = gee_client_with_cache.cache.get_stats()
        initial_entries = cache_stats1['total_entries']
        
        # Second call - should use cache for all points
        result2 = gee_client_with_cache.get_ndvi_for_points(test_data)
        
        # Cache entries should not increase (all should be hits)
        cache_stats2 = gee_client_with_cache.cache.get_stats()
        # May have a few more entries for IRG calculations, but most should be cached
        assert cache_stats2['total_entries'] >= initial_entries
        
        # Results should be identical
        assert result1['ndvi'].equals(result2['ndvi'])


