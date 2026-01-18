import pytest
import numpy as np
from src.inference.engine import ElkPredictionEngine
from pathlib import Path
from unittest.mock import Mock

class TestValidation:
    """Tests for validating predictions against known data"""
    
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
            "water_distance_miles": 0.93,
            "water_reliability": 1.0,
            "road_distance_miles": 1.86,
            "trail_distance_miles": 1.24,
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
        """Create test engine with mocked DataContextBuilder"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        for subdir in ["dem", "terrain", "landcover", "hydrology"]:
            (data_dir / subdir).mkdir()
        
        engine = ElkPredictionEngine(str(data_dir))
        
        # Mock the DataContextBuilder's build_context method to return quickly
        engine.data_builder.build_context = Mock(return_value=mock_context)
        
        return engine
    
    def test_known_good_habitat_scores_high(self, engine):
        """Test that known good elk habitat scores highly"""
        # Example: Area near Granite Creek, WY (known elk habitat)
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.52, 43.35],
                    [-110.48, 43.35],
                    [-110.48, 43.32],
                    [-110.52, 43.32],
                    [-110.52, 43.35]
                ]]
            },
            "date_range": {
                "start": "2026-10-15",
                "end": "2026-10-15",
                "find_best_days": False
            }
        }
        
        response = engine.predict(request)
        
        # Should score reasonably well (this is just checking system works)
        # In production, compare against actual GPS collar data
        assert response["overall"]["score"] > 0
    
    def test_known_poor_habitat_scores_low(self, engine):
        """Test that known poor habitat scores lower"""
        # Example: Low elevation, far from water, etc.
        # This would need real coordinates for true validation
        
        # For now, just verify engine handles various inputs
        request = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.90, 42.50],
                    [-110.85, 42.50],
                    [-110.85, 42.45],
                    [-110.90, 42.45],
                    [-110.90, 42.50]
                ]]
            },
            "date_range": {
                "start": "2026-10-15",
                "end": "2026-10-15",
                "find_best_days": False
            }
        }
        
        response = engine.predict(request)
        
        # Should produce valid response
        assert 0 <= response["overall"]["score"] <= 100
    
    def test_seasonal_variation(self, engine):
        """Test that scores vary appropriately by season"""
        location = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.70, 43.05],
                    [-110.65, 43.05],
                    [-110.65, 43.00],
                    [-110.70, 43.00],
                    [-110.70, 43.05]
                ]]
            }
        }
        
        # Test different seasons
        seasons = [
            ("2026-07-15", "summer"),
            ("2026-10-15", "fall"),
            ("2026-01-15", "winter")
        ]
        
        scores = {}
        for date, season in seasons:
            request = {
                **location,
                "date_range": {
                    "start": date,
                    "end": date,
                    "find_best_days": False
                }
            }
            response = engine.predict(request)
            scores[season] = response["overall"]["score"]
        
        # Scores should differ by season (exact relationships depend on location)
        # Main point: system responds to seasonal changes
        assert len(set(scores.values())) > 1  # Not all the same
    
    @pytest.mark.parametrize("month", range(1, 13))
    def test_monthly_seasonal_validation(self, engine, month):
        """Test seasonal validation across all 12 months"""
        location = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.70, 43.05],
                    [-110.65, 43.05],
                    [-110.65, 43.00],
                    [-110.70, 43.00],
                    [-110.70, 43.05]
                ]]
            }
        }
        
        # Create date for the 15th of each month in 2026
        date = f"2026-{month:02d}-15"
        
        request = {
            **location,
            "date_range": {
                "start": date,
                "end": date,
                "find_best_days": False
            }
        }
        
        response = engine.predict(request)
        
        # Validate response structure
        assert "overall" in response
        assert "score" in response["overall"]
        assert 0 <= response["overall"]["score"] <= 100
        
        # Validate that system produces valid predictions for all months
        assert response["overall"]["score"] >= 0
        assert response["overall"]["confidence_level"] in ["low", "medium", "high"]
        
        # Store month name for reporting
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        # Log the score for this month (visible in verbose test output)
        print(f"\n{month_names[month]} ({date}): Score = {response['overall']['score']:.2f}")
    
    def test_seasonal_patterns_all_months(self, engine):
        """Test that seasonal patterns are consistent across all months"""
        location = {
            "location": {
                "type": "polygon",
                "coordinates": [[
                    [-110.70, 43.05],
                    [-110.65, 43.05],
                    [-110.65, 43.00],
                    [-110.70, 43.00],
                    [-110.70, 43.05]
                ]]
            }
        }
        
        # Test all 12 months
        monthly_scores = {}
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        for month in range(1, 13):
            date = f"2026-{month:02d}-15"
            request = {
                **location,
                "date_range": {
                    "start": date,
                    "end": date,
                    "find_best_days": False
                }
            }
            response = engine.predict(request)
            monthly_scores[month] = {
                "month_name": month_names[month],
                "date": date,
                "score": response["overall"]["score"],
                "confidence": response["overall"]["confidence_level"]
            }
        
        # Validate that we have scores for all months
        assert len(monthly_scores) == 12
        
        # Validate that scores vary (system responds to seasonal changes)
        scores = [v["score"] for v in monthly_scores.values()]
        assert len(set(scores)) > 1, "Scores should vary across months"
        
        # Validate that all scores are in valid range
        assert all(0 <= s <= 100 for s in scores), "All scores should be 0-100"
        
        # Print summary for analysis
        print("\n=== Seasonal Validation Summary ===")
        for month, data in monthly_scores.items():
            print(f"{data['month_name']:12} ({data['date']}): Score={data['score']:6.2f}, Confidence={data['confidence']}")
        
        # Check for expected seasonal patterns
        # Winter months (Dec, Jan, Feb, Mar) might have different patterns than summer
        winter_scores = [monthly_scores[m]["score"] for m in [12, 1, 2, 3]]
        summer_scores = [monthly_scores[m]["score"] for m in [6, 7, 8]]
        
        # At minimum, ensure system produces different scores for different seasons
        # (exact relationships depend on location and heuristics)
        assert len(set(winter_scores + summer_scores)) > 1, \
            "Winter and summer should show some variation"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
