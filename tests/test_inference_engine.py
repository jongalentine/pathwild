import pytest
from pathlib import Path
from src.inference.engine import ElkPredictionEngine

class TestElkPredictionEngine:
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create test engine with mock data directory"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create minimal directory structure
        (data_dir / "dem").mkdir()
        (data_dir / "terrain").mkdir()
        (data_dir / "landcover").mkdir()
        
        engine = ElkPredictionEngine(str(data_dir))
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
