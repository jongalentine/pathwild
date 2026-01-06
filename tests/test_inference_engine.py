import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from src.inference.engine import ElkPredictionEngine

class TestElkPredictionEngine:
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context that DataContextBuilder.build_context returns"""
        return {
            "elevation": 8500.0,
            "slope": 15.0,
            "aspect": 180.0,
            "landcover": "Mixed Forest",
            "nlcd_code": 43,
            "canopy_cover": 45.0,
            "water_distance_miles": 0.93,  # ~1.5 km in miles
            "water_reliability": 1.0,  # High reliability (0-1 scale)
            "road_distance_miles": 1.86,  # ~3.0 km in miles
            "trail_distance_miles": 1.24,  # ~2.0 km in miles
            "ndvi": 0.65,
            "ndvi_age_days": 5,
            "irg": 0.01,
            "summer_integrated_ndvi": 70.0,
            "temperature_f": 50.0,
            "temp_high": 60.0,
            "temp_low": 40.0,
            "precip_last_7_days_inches": 0.5,
            "cloud_cover_percent": 30,
            "snow_depth_inches": 0.0,
            "snow_water_equiv_inches": 0.0,
            "snow_crust_detected": False,
            "snow_data_source": "snotel",
            "snow_station_name": "TEST_STATION",
            "snow_station_distance_km": 5.0
        }
    
    @pytest.fixture
    def engine(self, tmp_path, mock_context):
        """Create test engine with mock data directory and mocked DataContextBuilder"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create minimal directory structure
        (data_dir / "dem").mkdir()
        (data_dir / "terrain").mkdir()
        (data_dir / "landcover").mkdir()
        
        engine = ElkPredictionEngine(str(data_dir))
        
        # Mock the DataContextBuilder's build_context method to return quickly
        engine.data_builder.build_context = Mock(return_value=mock_context)
        
        return engine
    
    def test_simple_prediction(self, engine):
        """Test basic prediction request"""
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.72, 43.08],
                    [-110.68, 43.08],
                    [-110.68, 43.05],
                    [-110.72, 43.05],
                    [-110.72, 43.08]
                ]]
            },
            "date_range": {
                "start": "2026-10-27",
                "end": "2026-10-27",
                "find_best_days": False
            }
        }
        
        response = engine.predict(request)
        
        # Check response structure
        assert "query" in response
        assert "overall" in response
        assert "hotspots" in response
        assert "factor_breakdown" in response
        assert "recommendations" in response
        assert "metadata" in response
        
        # Check overall metrics
        assert 0 <= response["overall"]["score"] <= 100
        assert response["overall"]["estimated_population"] > 0
        assert response["overall"]["confidence_level"] in ["low", "medium", "high"]

    def test_date_range_prediction(self, engine):
        """Test prediction across date range"""
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.72, 43.08],
                    [-110.68, 43.08],
                    [-110.68, 43.05],
                    [-110.72, 43.05],
                    [-110.72, 43.08]
                ]]
            },
            "date_range": {
                "start": "2026-10-27",
                "end": "2026-10-31",
                "find_best_days": True
            }
        }
        
        response = engine.predict(request)
        
        # Should have daily predictions
        assert "daily_predictions" in response
        assert len(response["daily_predictions"]) == 5  # 5 days
        
        # Check daily predictions structure
        for pred in response["daily_predictions"]:
            assert "date" in pred
            assert "score" in pred
            assert "rank" in pred
            assert "reason" in pred
        
        # Check rankings
        ranks = [p["rank"] for p in response["daily_predictions"]]
        assert sorted(ranks) == list(range(1, 6))

    def test_hotspot_identification(self, engine):
        """Test that hotspots are identified"""
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.72, 43.08],
                    [-110.68, 43.08],
                    [-110.68, 43.05],
                    [-110.72, 43.05],
                    [-110.72, 43.08]
                ]]
            },
            "date_range": {
                "start": "2026-10-27",
                "end": "2026-10-27",
                "find_best_days": False
            }
        }
        
        response = engine.predict(request)
        
        # Should have hotspots
        assert len(response["hotspots"]) > 0
        
        # Check hotspot structure
        for hotspot in response["hotspots"]:
            assert "id" in hotspot
            assert "rank" in hotspot
            assert "center" in hotspot
            assert "score" in hotspot
            assert "estimated_elk" in hotspot
            assert "min" in hotspot["estimated_elk"]
            assert "max" in hotspot["estimated_elk"]
