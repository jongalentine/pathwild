import pytest
from pathlib import Path
from src.data.processors import DataContextBuilder

class TestDataContextBuilder:
    
    @pytest.fixture
    def builder(self, tmp_path):
        """Create test builder with mock data"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create directory structure
        (data_dir / "dem").mkdir()
        (data_dir / "terrain").mkdir()
        (data_dir / "landcover").mkdir()
        (data_dir / "hydrology").mkdir()
        
        builder = DataContextBuilder(data_dir)
        return builder
    
    def test_context_structure(self, builder):
        """Test that context has expected structure"""
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        context = builder.build_context(location, date)
        
        # Check required fields
        assert "elevation" in context
        assert "slope_degrees" in context
        assert "water_distance_miles" in context
        assert "snow_depth_inches" in context
        assert "temperature_f" in context
        assert "ndvi" in context
        assert "wolves_per_1000_elk" in context
        assert "security_habitat_percent" in context
    
    def test_context_values_reasonable(self, builder):
        """Test that context values are in reasonable ranges"""
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        context = builder.build_context(location, date)
        
        # Check ranges
        assert 5000 <= context["elevation"] <= 14000  # Wyoming elevations
        assert 0 <= context["slope_degrees"] <= 90
        assert 0 <= context["water_distance_miles"] <= 20
        assert 0 <= context["snow_depth_inches"] <= 200
        assert -40 <= context["temperature_f"] <= 100
        assert 0 <= context["ndvi"] <= 1.0
        assert 0 <= context["wolves_per_1000_elk"] <= 50
        assert 0 <= context["security_habitat_percent"] <= 100
    
    def test_temporal_data_varies_by_date(self, builder):
        """Test that temporal data changes with date"""
        location = {"lat": 43.0, "lon": -110.0}
        
        winter_context = builder.build_context(location, "2026-01-15")
        summer_context = builder.build_context(location, "2026-07-15")
        
        # Snow should differ
        assert winter_context["snow_depth_inches"] != summer_context["snow_depth_inches"]
        
        # NDVI should differ
        assert winter_context["ndvi"] != summer_context["ndvi"]
