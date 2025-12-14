import pytest
from src.scoring.aggregator import ScoreAggregator
from src.scoring.heuristics import ElevationHeuristic, WaterDistanceHeuristic

class TestScoreAggregator:
    
    def test_additive_aggregation(self):
        """Test additive score aggregation"""
        heuristics = [
            ElevationHeuristic(weight=2.0),
            WaterDistanceHeuristic(weight=3.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="additive")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            "water_distance_miles": 0.5,
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)
        
        # Should have positive score
        assert 0 <= result.total_score <= 100
        
        # Should have factor breakdown
        assert "elevation" in result.factor_scores
        assert "water_distance" in result.factor_scores
        
        # Check contributions
        elev_contrib = result.factor_scores["elevation"]["contribution"]
        water_contrib = result.factor_scores["water_distance"]["contribution"]
        assert elev_contrib > 0
        assert water_contrib > 0
    
    def test_multiplicative_aggregation(self):
        """Test multiplicative score aggregation"""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            WaterDistanceHeuristic(weight=1.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="multiplicative")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            "water_distance_miles": 0.5,
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)
        
        assert 0 <= result.total_score <= 100
    
    def test_limiting_factor_identification(self):
        """Test that limiting factor is correctly identified"""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            WaterDistanceHeuristic(weight=1.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="additive")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,  # Good
            "water_distance_miles": 3.5,  # Poor
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)
        
        # Water should be limiting factor
        assert result.limiting_factor == "water_distance"
