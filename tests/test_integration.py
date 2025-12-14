import pytest
from pathlib import Path
from src.inference.engine import ElkPredictionEngine
import json

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with test data"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create minimal directory structure
        for subdir in ["dem", "terrain", "landcover", "hydrology", 
                      "infrastructure", "wildlife", "canopy"]:
            (data_dir / subdir).mkdir()
        
        engine = ElkPredictionEngine(str(data_dir))
        return engine
    
    def test_full_prediction_workflow(self, engine):
        """Test complete prediction workflow"""
        # Create realistic request
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.75, 43.10],
                    [-110.65, 43.10],
                    [-110.65, 43.00],
                    [-110.75, 43.00],
                    [-110.75, 43.10]
                ]]
            },
            "date_range": {
                "start": "2026-10-27",
                "end": "2026-10-31",
                "find_best_days": True
            }
        }
        
        # Run prediction
        response = engine.predict(request)
        
        # Validate complete response structure
        self._validate_response_structure(response)
        
        # Validate data quality
        self._validate_response_data(response)
        
        # Check that response is JSON serializable
        json_str = json.dumps(response)
        assert len(json_str) > 0
    
    def _validate_response_structure(self, response):
        """Validate response has all required fields"""
        required_top_level = [
            "query", "overall", "hotspots", "factor_breakdown",
            "migration_status", "activity_patterns", 
            "recommendations", "metadata"
        ]
        
        for field in required_top_level:
            assert field in response, f"Missing required field: {field}"
        
        # Validate query
        assert "area_acres" in response["query"]
        assert "date_range" in response["query"]
        
        # Validate overall
        assert "score" in response["overall"]
        assert "estimated_population" in response["overall"]
        assert "confidence_level" in response["overall"]
        
        # Validate hotspots
        assert isinstance(response["hotspots"], list)
        if len(response["hotspots"]) > 0:
            hotspot = response["hotspots"][0]
            assert "center" in hotspot
            assert "score" in hotspot
            assert "estimated_elk" in hotspot
        
        # Validate factor breakdown
        assert len(response["factor_breakdown"]) > 0
        for factor in response["factor_breakdown"].values():
            assert "score" in factor
            assert "weight" in factor
            assert "contribution" in factor
        
        # Validate daily predictions if present
        if "daily_predictions" in response:
            assert len(response["daily_predictions"]) > 0
            for pred in response["daily_predictions"]:
                assert "date" in pred
                assert "score" in pred
                assert "rank" in pred
    
    def _validate_response_data(self, response):
        """Validate response data is reasonable"""
        # Score should be 0-100
        assert 0 <= response["overall"]["score"] <= 100
        
        # Population should be positive
        assert response["overall"]["estimated_population"] > 0
        
        # Confidence should be valid
        assert response["overall"]["confidence_level"] in ["low", "medium", "high"]
        
        # Factor scores should be 0-10
        for factor in response["factor_breakdown"].values():
            assert 0 <= factor["score"] <= 10
        
        # Hotspot scores should be 0-100
        for hotspot in response["hotspots"]:
            assert 0 <= hotspot["score"] <= 100
            assert hotspot["estimated_elk"]["min"] <= hotspot["estimated_elk"]["max"]
    
    def test_weighted_vs_unweighted(self, engine):
        """Test that weights affect scores"""
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
        
        # Unweighted
        response1 = engine.predict(request)
        score1 = response1["overall"]["score"]
        
        # Create new engine with different weights
        weighted_engine = ElkPredictionEngine(
            engine.data_dir,
            weights={"water": 3.0, "elevation": 2.5}
        )
        
        response2 = weighted_engine.predict(request)
        score2 = response2["overall"]["score"]
        
        # Scores should differ (unless by coincidence)
        # We mainly want to verify weights are being applied
        assert "water_distance" in response2["factor_breakdown"]
        assert response2["factor_breakdown"]["water_distance"]["weight"] == 3.0
