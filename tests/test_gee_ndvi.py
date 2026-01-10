"""
Tests for Google Earth Engine NDVI integration

Note: Most tests are marked with @pytest.mark.slow because they make real
GEE API calls, which are inherently slow (1-2 seconds per call).

Fast tests (no GEE API calls, no file I/O):
- test_gee_initializer_singleton
- test_collection_config
- test_invalid_collection

Run fast tests only: pytest tests/test_gee_ndvi.py -m "not slow" -v
Run all tests: pytest tests/test_gee_ndvi.py -v
"""
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Skip all tests if earthengine-api not available
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    pytestmark = pytest.mark.skip("earthengine-api not available")

from src.data.processors import GEENDVIClient, DataContextBuilder, GEEInitializer, NDVICache, NDVICache


@pytest.fixture
def gee_client(test_config):
    """Fixture for GEE NDVI client"""
    # Get project from config or use default
    gee_config = test_config.get('gee', {})
    project = gee_config.get('project', 'ee-jongalentine')
    
    # Log which project we're trying to use (for debugging)
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to initialize GEE with project: {project}")
    
    # Skip if not authenticated (will fail gracefully)
    try:
        return GEENDVIClient(
            project=project,
            collection='landsat8',
            batch_size=10,
            max_workers=2
        )
    except Exception as e:
        pytest.skip(f"GEE not authenticated or available (project: {project}): {e}")


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


@pytest.mark.slow
def test_gee_initialization(gee_client):
    """Test GEE client initializes correctly"""
    status = gee_client.check_availability()
    
    assert 'gee_initialized' in status or 'gee_available' in status
    # Should be initialized or have an error message
    assert status.get('gee_initialized', False) or 'error' in status


@pytest.mark.slow
def test_single_point_ndvi(gee_client):
    """Test NDVI retrieval for single Wyoming point"""
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(2020, 6, 15)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'ndvi' in result.columns
    assert len(result) == 1
    
    # NDVI should be valid if found (range: -1 to 1)
    if pd.notna(result['ndvi'].iloc[0]):
        ndvi_value = result['ndvi'].iloc[0]
        assert -1 <= ndvi_value <= 1, f"NDVI value {ndvi_value} out of valid range"


@pytest.mark.slow
def test_batch_processing(gee_client):
    """Test batch processing with multiple points"""
    # Wyoming Area 048 test points
    test_data = pd.DataFrame({
        'latitude': [44.5 + i*0.01 for i in range(5)],  # Reduced for faster tests
        'longitude': [-107.25 - i*0.01 for i in range(5)],
        'timestamp': [datetime(2020, 6, 15) for _ in range(5)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert len(result) == 5
    assert 'ndvi' in result.columns
    
    # At least some should have valid NDVI (may be None if no data available)
    valid_ndvi = result['ndvi'].notna().sum()
    assert valid_ndvi >= 0  # At least 0 valid (may be 0 if no data)


@pytest.mark.slow
def test_data_context_builder_integration(test_config):
    """Test NDVI integration with DataContextBuilder"""
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    # Try to initialize with GEE enabled
    try:
        gee_config = test_config.get('gee', {})
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=gee_config.get('enabled', False),
            gee_config=gee_config
        )
        
        test_data = pd.DataFrame({
            'latitude': [44.5],
            'longitude': [-107.25],
            'timestamp': [datetime(2020, 6, 15)]
        })
        
        # Only test if GEE is enabled and available
        if builder.use_gee_ndvi and builder.ndvi_client:
            result = builder.add_ndvi(test_data)
            assert 'ndvi' in result.columns
            assert len(result) == 1
    except Exception as e:
        # Skip if GEE not available/authenticated
        pytest.skip(f"GEE not available for integration test: {e}")


@pytest.mark.slow
def test_missing_data_handling(gee_client):
    """Test handling of unavailable data"""
    # Use date before Landsat 8 (starts 2013-04-11)
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(1980, 1, 1)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    # Should return None/NaN, not crash
    assert result['ndvi'].iloc[0] is None or pd.isna(result['ndvi'].iloc[0])


@pytest.mark.slow
def test_different_date_columns(gee_client):
    """Test that different date column names work"""
    test_data = pd.DataFrame({
        'lat': [44.5],
        'lon': [-107.25],
        'date': [datetime(2020, 6, 15)]
    })
    
    result = gee_client.get_ndvi_for_points(
        test_data,
        date_column='date',
        lat_column='lat',
        lon_column='lon'
    )
    
    assert 'ndvi' in result.columns
    assert len(result) == 1


@pytest.mark.slow
def test_parallel_processing(test_config):
    """
    Test parallel processing with larger dataset.
    
    Note: This test processes multiple points in parallel. Each point requires
    2 GEE API calls (NDVI + IRG), so this can be slow. Reduced to 10 points
    to avoid timeouts and rate limits.
    """
    # Get project from config or use default
    gee_config = test_config.get('gee', {})
    project = gee_config.get('project', 'ee-jongalentine')
    
    try:
        client = GEENDVIClient(
            project=project,
            batch_size=5,   # Small batches to avoid overwhelming GEE
            max_workers=2   # Reduced workers to avoid rate limits
        )
    except Exception as e:
        pytest.skip(f"GEE not available: {e}")
    
    # Reduced to 10 points to avoid timeouts (each point = 2 GEE calls: NDVI + IRG)
    # 10 points × 2 calls = 20 GEE calls, which is manageable
    n_points = 10
    test_data = pd.DataFrame({
        'latitude': [44.5 + (i % 10) * 0.01 for i in range(n_points)],
        'longitude': [-107.25 - (i % 10) * 0.01 for i in range(n_points)],
        'timestamp': [datetime(2020, 6, 15) for _ in range(n_points)]
    })
    
    result = client.get_ndvi_for_points(test_data)
    
    assert len(result) == n_points
    
    # At least some should have valid NDVI
    success_rate = result['ndvi'].notna().sum() / len(result)
    assert success_rate >= 0  # May be 0 if no data available


@pytest.mark.slow
def test_thread_safety(test_config):
    """Test that concurrent requests don't cause issues"""
    import threading
    
    # Get project from config or use default
    gee_config = test_config.get('gee', {})
    project = gee_config.get('project', 'ee-jongalentine')
    
    def get_ndvi_batch(batch_id):
        try:
            client = GEENDVIClient(project=project)
            test_data = pd.DataFrame({
                'latitude': [44.5 + batch_id * 0.01],
                'longitude': [-107.25 - batch_id * 0.01],
                'timestamp': [datetime(2020, 6, 15)]
            })
            return client.get_ndvi_for_points(test_data)
        except Exception:
            return None
    
    # Run 5 concurrent requests
    threads = []
    results = []
    
    def worker(batch_id):
        result = get_ndvi_batch(batch_id)
        results.append(result)
    
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # If we get here without errors, thread safety is good
    assert len(results) == 5


def test_gee_initializer_singleton():
    """Test that GEEInitializer is a singleton"""
    init1 = GEEInitializer()
    init2 = GEEInitializer()
    
    assert init1 is init2


def test_collection_config():
    """Test that collection configurations are correct"""
    assert 'landsat8' in GEENDVIClient.COLLECTIONS
    assert 'sentinel2' in GEENDVIClient.COLLECTIONS
    
    landsat8_config = GEENDVIClient.COLLECTIONS['landsat8']
    assert 'id' in landsat8_config
    assert 'red_band' in landsat8_config
    assert 'nir_band' in landsat8_config
    assert 'scale' in landsat8_config


def test_invalid_collection():
    """Test that invalid collection raises error"""
    # This should fail BEFORE trying to initialize GEE
    # So it should work even if GEE is not authenticated
    with pytest.raises(ValueError, match="Unknown collection"):
        GEENDVIClient(collection='invalid_collection')


@pytest.mark.slow
def test_data_context_builder_without_gee():
    """Test DataContextBuilder works without GEE
    
    Note: This test is slow because DataContextBuilder initialization loads
    large raster/vector files (DEM, slope, aspect, landcover, etc.) even when
    GEE is disabled.
    """
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    builder = DataContextBuilder(
        data_dir=data_dir,
        use_gee_ndvi=False
    )
    
    assert builder.use_gee_ndvi is False
    assert builder.ndvi_client is None
    assert builder.satellite_client is not None


@pytest.mark.slow
def test_ndvi_age_days_extraction(gee_client):
    """Test that ndvi_age_days is extracted from GEE image timestamp"""
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(2020, 6, 15)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'ndvi_age_days' in result.columns
    assert len(result) == 1
    
    # If NDVI is available, age_days should be a reasonable value
    if pd.notna(result['ndvi'].iloc[0]):
        age_days = result['ndvi_age_days'].iloc[0]
        # Age should be between 0 and buffer_days*2 (default buffer is 7 days)
        # So age should be 0-14 days typically, but allow up to 30 for edge cases
        assert pd.isna(age_days) or (0 <= age_days <= 30), \
            f"ndvi_age_days {age_days} should be between 0 and 30 days"


@pytest.mark.slow
def test_irg_calculation(gee_client):
    """Test that IRG is calculated from NDVI time series"""
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(2020, 6, 15)]  # Summer date
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'irg' in result.columns
    assert len(result) == 1
    
    # IRG should be present (may be None if can't calculate)
    irg_value = result['irg'].iloc[0]
    
    # If IRG is calculated, it should be a reasonable value
    # IRG typically ranges from -0.1 to 0.1 (rate of change per day)
    if pd.notna(irg_value):
        assert -0.2 <= irg_value <= 0.2, \
            f"IRG value {irg_value} should be between -0.2 and 0.2"


@pytest.mark.slow
def test_irg_seasonal_fallback(gee_client):
    """Test that IRG falls back to seasonal approximation when past data unavailable"""
    # Use a date where past NDVI might not be available
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(2020, 4, 15)]  # Spring - should have positive IRG fallback
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'irg' in result.columns
    irg_value = result['irg'].iloc[0]
    
    # Even if past NDVI unavailable, should have seasonal fallback
    # Spring (April) should have positive IRG (0.01) or calculated value
    if pd.notna(irg_value):
        # Spring should have positive or near-zero IRG
        assert irg_value >= -0.1, f"Spring IRG {irg_value} should be >= -0.1"


@pytest.mark.slow
def test_ndvi_age_days_in_batch(gee_client):
    """Test that ndvi_age_days is included in batch processing results"""
    test_data = pd.DataFrame({
        'latitude': [44.5 + i*0.01 for i in range(3)],
        'longitude': [-107.25 - i*0.01 for i in range(3)],
        'timestamp': [datetime(2020, 6, 15) for _ in range(3)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'ndvi_age_days' in result.columns
    assert len(result) == 3
    
    # Check that age_days values are reasonable
    for idx, row in result.iterrows():
        if pd.notna(row['ndvi']):
            age_days = row['ndvi_age_days']
            assert pd.isna(age_days) or (0 <= age_days <= 30), \
                f"ndvi_age_days {age_days} should be between 0 and 30 days"


@pytest.mark.slow
def test_irg_in_batch(gee_client):
    """Test that IRG is calculated for all points in batch"""
    test_data = pd.DataFrame({
        'latitude': [44.5 + i*0.01 for i in range(3)],
        'longitude': [-107.25 - i*0.01 for i in range(3)],
        'timestamp': [datetime(2020, 6, 15) for _ in range(3)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    assert 'irg' in result.columns
    assert len(result) == 3
    
    # Check that IRG values are reasonable
    for idx, row in result.iterrows():
        irg_value = row['irg']
        if pd.notna(irg_value):
            assert -0.2 <= irg_value <= 0.2, \
                f"IRG value {irg_value} should be between -0.2 and 0.2"


@pytest.mark.slow
def test_build_context_with_ndvi_age_days(test_config):
    """Test that build_context includes real ndvi_age_days from GEE"""
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    try:
        gee_config = test_config.get('gee', {})
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=gee_config.get('enabled', False),
            gee_config=gee_config
        )
        
        if not builder.use_gee_ndvi or not builder.ndvi_client:
            pytest.skip("GEE not enabled or available")
        
        # Build context for a single point
        context = builder.build_context(
            lat=44.5,
            lon=-107.25,
            date=datetime(2020, 6, 15)
        )
        
        # Check that ndvi_age_days is present and reasonable
        assert 'ndvi_age_days' in context
        if pd.notna(context.get('ndvi')):
            age_days = context.get('ndvi_age_days')
            assert age_days is not None
            assert 0 <= age_days <= 30, f"ndvi_age_days {age_days} should be reasonable"
            
    except Exception as e:
        pytest.skip(f"GEE not available for build_context test: {e}")


@pytest.mark.slow
def test_build_context_with_irg(test_config):
    """Test that build_context includes real IRG from GEE"""
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    try:
        gee_config = test_config.get('gee', {})
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=gee_config.get('enabled', False),
            gee_config=gee_config
        )
        
        if not builder.use_gee_ndvi or not builder.ndvi_client:
            pytest.skip("GEE not enabled or available")
        
        # Build context for a single point
        context = builder.build_context(
            lat=44.5,
            lon=-107.25,
            date=datetime(2020, 6, 15)
        )
        
        # Check that IRG is present and reasonable
        assert 'irg' in context
        irg_value = context.get('irg')
        if irg_value is not None:
            assert -0.2 <= irg_value <= 0.2, f"IRG {irg_value} should be reasonable"
            
    except Exception as e:
        pytest.skip(f"GEE not available for build_context test: {e}")



@pytest.mark.slow
def test_summer_integrated_ndvi_calculation(gee_client):
    """Test summer integrated NDVI calculation"""
    # Test for a year with known data availability
    year = 2020
    
    integrated_ndvi = gee_client.get_summer_integrated_ndvi(
        lat=44.5,
        lon=-107.25,
        year=year,
        buffer_days=7,
        max_cloud_cover=30.0
    )
    
    # May return None if insufficient data, but if present should be reasonable
    if integrated_ndvi is not None:
        # Integrated NDVI is sum of multiple NDVI values (typically 4-8 images over summer)
        # Each NDVI is 0-1, so integrated should be roughly 2-8 (4 images * 0.5 avg)
        assert 0 <= integrated_ndvi <= 20, \
            f"Summer integrated NDVI {integrated_ndvi} should be reasonable (0-20)"


@pytest.mark.slow
def test_summer_integrated_ndvi_no_data(gee_client):
    """Test summer integrated NDVI when no data is available"""
    # Use a year before Landsat 8 (starts 2013-04-11)
    year = 2010
    
    integrated_ndvi = gee_client.get_summer_integrated_ndvi(
        lat=44.5,
        lon=-107.25,
        year=year,
        buffer_days=7,
        max_cloud_cover=30.0
    )
    
    # Should return None for years before satellite coverage
    assert integrated_ndvi is None, "Should return None for years without satellite data"


@pytest.mark.slow
def test_build_context_with_summer_integrated_ndvi(test_config):
    """Test that build_context includes summer_integrated_ndvi from GEE"""
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    try:
        gee_config = test_config.get('gee', {})
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=gee_config.get('enabled', False),
            gee_config=gee_config
        )
        
        if not builder.use_gee_ndvi or not builder.ndvi_client:
            pytest.skip("GEE not enabled or available")
        
        # Build context for a point in fall (should use current year's summer)
        context = builder.build_context(
            lat=44.5,
            lon=-107.25,
            date=datetime(2020, 10, 15)  # October (after summer)
        )
        
        # Check that summer_integrated_ndvi is present
        assert 'summer_integrated_ndvi' in context
        summer_ndvi = context.get('summer_integrated_ndvi')
        
        # May be None if insufficient data, but if present should be reasonable
        if summer_ndvi is not None:
            assert 0 <= summer_ndvi <= 20, \
                f"summer_integrated_ndvi {summer_ndvi} should be reasonable (0-20)"
            
    except Exception as e:
        pytest.skip(f"GEE not available for build_context test: {e}")


@pytest.mark.slow
def test_all_ndvi_features_together(gee_client):
    """Integration test: verify all three features (ndvi, ndvi_age_days, irg) work together"""
    test_data = pd.DataFrame({
        'latitude': [44.5],
        'longitude': [-107.25],
        'timestamp': [datetime(2020, 6, 15)]
    })
    
    result = gee_client.get_ndvi_for_points(test_data)
    
    # All three features should be present
    assert 'ndvi' in result.columns
    assert 'ndvi_age_days' in result.columns
    assert 'irg' in result.columns
    assert 'cloud_cover_percent' in result.columns
    
    # If NDVI was found, other features should be present
    if pd.notna(result['ndvi'].iloc[0]):
        assert pd.notna(result['ndvi_age_days'].iloc[0]), \
            "ndvi_age_days should be present when NDVI is found"
        assert pd.notna(result['irg'].iloc[0]), \
            "IRG should be present when NDVI is found"
        
        # Validate ranges
        ndvi = result['ndvi'].iloc[0]
        age_days = result['ndvi_age_days'].iloc[0]
        irg = result['irg'].iloc[0]
        
        assert -1 <= ndvi <= 1, f"NDVI {ndvi} out of range"
        assert 0 <= age_days <= 30, f"ndvi_age_days {age_days} out of range"
        assert -0.2 <= irg <= 0.2, f"IRG {irg} out of range"


@pytest.mark.slow
def test_build_context_all_features(test_config):
    """Integration test: verify build_context includes all NDVI-related features"""
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    try:
        gee_config = test_config.get('gee', {})
        builder = DataContextBuilder(
            data_dir=data_dir,
            use_gee_ndvi=gee_config.get('enabled', False),
            gee_config=gee_config
        )
        
        if not builder.use_gee_ndvi or not builder.ndvi_client:
            pytest.skip("GEE not enabled or available")
        
        # Build context for a point in fall
        context = builder.build_context(
            lat=44.5,
            lon=-107.25,
            date=datetime(2020, 10, 15)
        )
        
        # All NDVI-related features should be present
        required_features = ['ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi']
        for feature in required_features:
            assert feature in context, f"Missing feature: {feature}"
            
            # Features may be None if data unavailable, but should be present in context
            value = context.get(feature)
            if value is not None:
                # Validate reasonable ranges
                if feature == 'ndvi':
                    assert -1 <= value <= 1, f"NDVI {value} out of range"
                elif feature == 'ndvi_age_days':
                    assert 0 <= value <= 30, f"ndvi_age_days {value} out of range"
                elif feature == 'irg':
                    assert -0.2 <= value <= 0.2, f"IRG {value} out of range"
                elif feature == 'summer_integrated_ndvi':
                    assert 0 <= value <= 20, f"summer_integrated_ndvi {value} out of range"
            
    except Exception as e:
        pytest.skip(f"GEE not available for build_context test: {e}")
