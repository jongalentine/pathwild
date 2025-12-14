import pytest
from datetime import datetime
from src.scoring.heuristics import (
    ElevationHeuristic,
    SnowConditionsHeuristic,
    WaterDistanceHeuristic,
    VegetationQualityHeuristic,
    HuntingPressureHeuristic,
    SecurityHabitatHeuristic,
    PredationRiskHeuristic,
    NutritionalConditionHeuristic,
    WinterSeverityHeuristic
)

class TestElevationHeuristic:
    
    def test_optimal_elevation_october(self):
        """Test that optimal October elevation (8500-9500) scores 10"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {"elevation": 9000.0}
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score == 10.0
        assert result.status == "excellent"
        assert "optimal" in result.note.lower()
    
    def test_low_elevation_october(self):
        """Test that low elevation in October scores poorly"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {"elevation": 6000.0}  # Too low
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score < 7.0
        assert result.status in ["critical", "poor"]
    
    def test_seasonal_variation(self):
        """Test that same elevation scores differently by season"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {"elevation": 7000.0}
        
        # Winter - should be good
        winter_result = heuristic.calculate(location, "2026-01-15", context)
        
        # Summer - should be poor
        summer_result = heuristic.calculate(location, "2026-07-15", context)
        
        assert winter_result.score > summer_result.score


class TestSnowConditionsHeuristic:
    
    def test_minimal_snow_scores_high(self):
        """Test that minimal snow scores well"""
        heuristic = SnowConditionsHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {"snow_depth_inches": 3.0}
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score >= 9.0
        assert result.status == "excellent"
    
    def test_deep_snow_scores_low(self):
        """Test that deep snow scores poorly"""
        heuristic = SnowConditionsHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {"snow_depth_inches": 35.0}
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score < 4.0
        assert result.status in ["poor", "critical"]
    
    def test_snow_crust_penalty(self):
        """Test that crusted snow reduces score"""
        heuristic = SnowConditionsHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-03-15"
        
        # Same depth, with and without crust
        no_crust = heuristic.calculate(location, date, 
                                      {"snow_depth_inches": 15.0, 
                                       "snow_crust_detected": False})
        
        with_crust = heuristic.calculate(location, date,
                                        {"snow_depth_inches": 15.0,
                                         "snow_crust_detected": True})
        
        assert with_crust.score < no_crust.score


class TestWaterDistanceHeuristic:
    
    def test_close_water_scores_high(self):
        """Test that proximity to water scores well"""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "water_distance_miles": 0.3,
            "water_reliability": 1.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score >= 9.0
        assert result.status == "good"
    
    def test_distant_water_scores_low(self):
        """Test that distance from water scores poorly"""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "water_distance_miles": 3.0,
            "water_reliability": 1.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score < 3.0
        assert result.status in ["poor", "critical"]
    
    def test_water_reliability_adjustment(self):
        """Test that ephemeral water reduces score"""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Permanent water
        permanent = heuristic.calculate(location, date,
                                       {"water_distance_miles": 0.5,
                                        "water_reliability": 1.0})
        
        # Ephemeral water
        ephemeral = heuristic.calculate(location, date,
                                       {"water_distance_miles": 0.5,
                                        "water_reliability": 0.4})
        
        assert ephemeral.score < permanent.score


class TestHuntingPressureHeuristic:
    
    def test_remote_during_season_scores_high(self):
        """Test that remote areas during hunting season score well"""
        heuristic = HuntingPressureHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"  # During rifle season
        context = {
            "road_distance_miles": 3.0,
            "trail_distance_miles": 2.0,
            "security_habitat_percent": 45
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score >= 8.0
        assert "low hunting pressure" in result.note.lower()
    
    def test_accessible_during_season_scores_low(self):
        """Test that accessible areas during hunting score poorly"""
        heuristic = HuntingPressureHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"  # During rifle season
        context = {
            "road_distance_miles": 0.2,
            "trail_distance_miles": 0.1,
            "security_habitat_percent": 20
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score < 6.0
        assert "heavy" in result.note.lower() or "high" in result.note.lower()
    
    def test_off_season_less_sensitive(self):
        """Test that accessibility matters less outside hunting season"""
        heuristic = HuntingPressureHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {
            "road_distance_miles": 0.3,
            "trail_distance_miles": 0.2,
            "security_habitat_percent": 30
        }
        
        # During season
        during = heuristic.calculate(location, "2026-10-15", context)
        
        # Off season
        off_season = heuristic.calculate(location, "2026-08-15", context)
        
        assert off_season.score > during.score


class TestPredationRiskHeuristic:
    
    def test_low_predator_density_scores_high(self):
        """Test that areas with few predators score well"""
        heuristic = PredationRiskHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "wolves_per_1000_elk": 2.0,
            "bear_activity_distance_miles": 8.0,
            "snow_depth_inches": 5.0,
            "security_habitat_percent": 40
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score >= 7.0
        assert "moderate" in result.note.lower()
    
    def test_high_wolf_density_reduces_score(self):
        """Test that high wolf density reduces score"""
        heuristic = PredationRiskHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "wolves_per_1000_elk": 8.0,  # Above threshold
            "bear_activity_distance_miles": 10.0,
            "snow_depth_inches": 5.0,
            "security_habitat_percent": 30
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert "wolf" in result.note.lower()
        assert result.score < 7.0
    
    def test_calving_season_bear_risk(self):
        """Test that bears near calving areas reduce score in May-June"""
        heuristic = PredationRiskHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {
            "wolves_per_1000_elk": 2.0,
            "bear_activity_distance_miles": 2.0,  # Close
            "snow_depth_inches": 0.0,
            "security_habitat_percent": 35
        }
        
        # During calving
        may_result = heuristic.calculate(location, "2026-05-15", context)
        
        # Not calving season
        oct_result = heuristic.calculate(location, "2026-10-15", context)
        
        assert may_result.score < oct_result.score
        assert "calving" in may_result.note.lower()


class TestWinterSeverityHeuristic:
    
    def test_mild_winter_scores_high(self):
        """Test that mild winters score well (low mortality)"""
        heuristic = WinterSeverityHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-04-30"  # End of winter
        context = {}  # Will use mild defaults
        
        # Mock mild winter WSI
        heuristic._calculate_cumulative_wsi = lambda *args: 1500
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score >= 9.0
        assert "mild" in result.note.lower()
    
    def test_severe_winter_scores_low(self):
        """Test that severe winters score poorly (high mortality)"""
        heuristic = WinterSeverityHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-04-30"
        context = {}
        
        # Mock severe winter WSI
        heuristic._calculate_cumulative_wsi = lambda *args: 5000
        
        result = heuristic.calculate(location, date, context)
        
        assert result.score <= 5.0
        assert "severe" in result.note.lower()
