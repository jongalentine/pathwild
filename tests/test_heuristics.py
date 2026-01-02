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


# Edge case tests for better coverage

class TestBaseHeuristic:
    """Test BaseHeuristic utility methods."""
    
    def test_normalize_score_basic(self):
        """Test basic normalization."""
        heuristic = ElevationHeuristic()
        
        # Test normal case
        score = heuristic.normalize_score(5.0, 0.0, 10.0, inverse=False)
        assert score == 5.0  # Middle of range = 5.0
        
        # Test minimum
        score = heuristic.normalize_score(0.0, 0.0, 10.0, inverse=False)
        assert score == 0.0
        
        # Test maximum
        score = heuristic.normalize_score(10.0, 0.0, 10.0, inverse=False)
        assert score == 10.0
    
    def test_normalize_score_clamping(self):
        """Test that values outside range are clamped."""
        heuristic = ElevationHeuristic()
        
        # Below minimum
        score = heuristic.normalize_score(-5.0, 0.0, 10.0, inverse=False)
        assert score == 0.0
        
        # Above maximum
        score = heuristic.normalize_score(15.0, 0.0, 10.0, inverse=False)
        assert score == 10.0
    
    def test_normalize_score_inverse(self):
        """Test inverse normalization (higher raw = lower score)."""
        heuristic = ElevationHeuristic()
        
        # Inverse: high raw value = low score
        score = heuristic.normalize_score(10.0, 0.0, 10.0, inverse=True)
        assert score == 0.0
        
        score = heuristic.normalize_score(0.0, 0.0, 10.0, inverse=True)
        assert score == 10.0
        
        score = heuristic.normalize_score(5.0, 0.0, 10.0, inverse=True)
        assert score == 5.0
    
    def test_get_status_boundaries(self):
        """Test status boundaries at exact thresholds."""
        heuristic = ElevationHeuristic()
        
        # Test exact boundaries
        assert heuristic.get_status(9.0) == "excellent"
        assert heuristic.get_status(8.99) == "good"
        assert heuristic.get_status(7.0) == "good"
        assert heuristic.get_status(6.99) == "fair"
        assert heuristic.get_status(5.0) == "fair"
        assert heuristic.get_status(4.99) == "poor"
        assert heuristic.get_status(3.0) == "poor"
        assert heuristic.get_status(2.99) == "critical"
        assert heuristic.get_status(0.0) == "critical"
        assert heuristic.get_status(10.0) == "excellent"


class TestWaterDistanceHeuristicEdgeCases:
    """Test WaterDistanceHeuristic edge cases and internal methods."""
    
    def test_water_distance_optimal_boundary(self):
        """Test optimal distance boundary (0.25 miles)."""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Exactly at optimal distance
        result = heuristic.calculate(location, date, {
            "water_distance_miles": 0.25,
            "water_reliability": 1.0
        })
        assert result.score == 10.0
        assert result.status == "excellent"
        
        # Just above optimal
        result = heuristic.calculate(location, date, {
            "water_distance_miles": 0.251,
            "water_reliability": 1.0
        })
        assert result.score < 10.0
        assert result.status == "good"
    
    def test_water_distance_calculation_with_sources(self):
        """Test _calculate_water_distance method with water sources."""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {
            "water_sources": [
                {"lat": 43.0, "lon": -110.0},  # Same location
                {"lat": 43.1, "lon": -110.1},  # ~8 miles away
            ]
        }
        
        distance = heuristic._calculate_water_distance(location, context)
        assert distance == 0.0  # Should find the closest (same location)
    
    def test_water_distance_missing_sources(self):
        """Test error handling when water sources are missing."""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {}  # No water sources
        
        with pytest.raises(ValueError, match="No water source data"):
            heuristic._calculate_water_distance(location, context)
    
    def test_haversine_distance_accuracy(self):
        """Test haversine distance calculation accuracy."""
        heuristic = WaterDistanceHeuristic()
        
        # Test known distance: same point
        dist = heuristic._haversine_distance(43.0, -110.0, 43.0, -110.0)
        assert abs(dist) < 0.01  # Should be ~0
        
        # Test known distance: ~1 degree lat = ~69 miles
        dist = heuristic._haversine_distance(43.0, -110.0, 44.0, -110.0)
        assert 68 < dist < 70  # Approximately 69 miles
        
        # Test cross-hemisphere (should still work)
        dist = heuristic._haversine_distance(43.0, -110.0, 43.0, -111.0)
        assert dist > 0
    
    def test_water_reliability_edge_cases(self):
        """Test water reliability adjustments at boundaries."""
        heuristic = WaterDistanceHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Perfect reliability
        result = heuristic.calculate(location, date, {
            "water_distance_miles": 0.5,
            "water_reliability": 1.0
        })
        base_score = result.score
        
        # Zero reliability (ephemeral)
        result = heuristic.calculate(location, date, {
            "water_distance_miles": 0.5,
            "water_reliability": 0.0
        })
        assert result.score == 0.0
        
        # Partial reliability
        result = heuristic.calculate(location, date, {
            "water_distance_miles": 0.5,
            "water_reliability": 0.5
        })
        assert result.score < base_score
        assert result.score > 0.0


class TestElevationHeuristicEdgeCases:
    """Test ElevationHeuristic edge cases."""
    
    def test_elevation_seasonal_transitions(self):
        """Test elevation scoring at month boundaries."""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        elevation = 8500.0
        
        # Test month transitions
        jan_result = heuristic.calculate(location, "2026-01-31", {"elevation": elevation})
        feb_result = heuristic.calculate(location, "2026-02-01", {"elevation": elevation})
        # Should be similar (both winter)
        assert abs(jan_result.score - feb_result.score) < 1.0
        
        # Spring transition
        mar_result = heuristic.calculate(location, "2026-03-31", {"elevation": elevation})
        apr_result = heuristic.calculate(location, "2026-04-01", {"elevation": elevation})
        # April should score better for 8500ft
        assert apr_result.score >= mar_result.score
    
    def test_elevation_extreme_values(self):
        """Test very high and very low elevations."""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Very low elevation
        result = heuristic.calculate(location, date, {"elevation": 1000.0})
        assert result.score < 3.0
        assert result.status in ["poor", "critical"]
        
        # Very high elevation
        result = heuristic.calculate(location, date, {"elevation": 15000.0})
        assert result.score < 3.0
        assert result.status in ["poor", "critical"]
    
    def test_elevation_optimal_ranges(self):
        """Test that optimal ranges score 10.0."""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        
        # October optimal: 8500-9500
        result = heuristic.calculate(location, "2026-10-15", {"elevation": 9000.0})
        assert result.score == 10.0
        
        # July optimal: 9500-10500
        result = heuristic.calculate(location, "2026-07-15", {"elevation": 10000.0})
        assert result.score == 10.0


class TestNutritionHeuristicEdgeCases:
    """Test NutritionalConditionHeuristic edge cases."""
    
    def test_nutrition_summer_ndvi_estimation(self):
        """Test fallback when summer NDVI not available."""
        heuristic = NutritionalConditionHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # No summer NDVI, should estimate from current
        result = heuristic.calculate(location, date, {
            "ndvi": 0.5,  # Current NDVI
            "predation_score": 7.0
        })
        
        assert result.score > 0
        assert "summer_integrated_ndvi" in result.metadata
    
    def test_nutrition_predation_stress(self):
        """Test that predation stress reduces body condition."""
        heuristic = NutritionalConditionHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Low predation stress
        low_stress = heuristic.calculate(location, date, {
            "summer_integrated_ndvi": 60.0,
            "predation_score": 9.0  # Low predation
        })
        
        # High predation stress
        high_stress = heuristic.calculate(location, date, {
            "summer_integrated_ndvi": 60.0,
            "predation_score": 2.0  # High predation
        })
        
        assert high_stress.score < low_stress.score
    
    def test_nutrition_body_fat_edge_cases(self):
        """Test body fat estimation at boundaries."""
        heuristic = NutritionalConditionHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Excellent threshold (13.7%)
        result = heuristic.calculate(location, date, {
            "summer_integrated_ndvi": 73.5,  # Should give ~13.7% body fat
            "predation_score": 7.0
        })
        assert result.status == "excellent"
        
        # Critical threshold (7.9%)
        result = heuristic.calculate(location, date, {
            "summer_integrated_ndvi": 44.5,  # Should give ~7.9% body fat
            "predation_score": 7.0
        })
        assert result.status in ["fair", "poor"]


class TestSecurityHabitatHeuristicEdgeCases:
    """Test SecurityHabitatHeuristic edge cases."""
    
    def test_security_habitat_detection(self):
        """Test security habitat detection with multiple criteria."""
        heuristic = SecurityHabitatHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Steep slope (security)
        result = heuristic.calculate(location, date, {
            "slope_degrees": 45.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 30.0
        })
        assert "steep slope" in result.note.lower()
        assert result.metadata["is_security_habitat"] is True
        
        # Dense cover (security)
        result = heuristic.calculate(location, date, {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 75.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 30.0
        })
        assert "dense cover" in result.note.lower()
        assert result.metadata["is_security_habitat"] is True
        
        # Remote (security)
        result = heuristic.calculate(location, date, {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 3.0,
            "trail_distance_miles": 2.0,
            "security_habitat_percent": 30.0
        })
        assert "remote" in result.note.lower()
        assert result.metadata["is_security_habitat"] is True
    
    def test_security_percentage_calculations(self):
        """Test security percentage scoring."""
        heuristic = SecurityHabitatHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Optimal security (40%+)
        result = heuristic.calculate(location, date, {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 45.0
        })
        assert result.score == 10.0
        assert result.status == "excellent"
        
        # Critical security (<20%)
        result = heuristic.calculate(location, date, {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 15.0
        })
        assert result.score < 4.0
        assert result.status == "poor"


class TestPredationRiskHeuristicEdgeCases:
    """Test PredationRiskHeuristic edge cases."""
    
    def test_predation_wolf_density_edge_cases(self):
        """Test wolf density at boundaries."""
        heuristic = PredationRiskHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        
        # Zero wolves
        result = heuristic.calculate(location, date, {
            "wolves_per_1000_elk": 0.0,
            "bear_activity_distance_miles": 10.0,
            "snow_depth_inches": 5.0,
            "security_habitat_percent": 35.0
        })
        assert result.score >= 7.0  # Should score well with no wolves
        
        # Very high wolf density
        result = heuristic.calculate(location, date, {
            "wolves_per_1000_elk": 15.0,
            "bear_activity_distance_miles": 10.0,
            "snow_depth_inches": 5.0,
            "security_habitat_percent": 35.0
        })
        assert result.score < 5.0
    
    def test_predation_bear_activity(self):
        """Test bear activity distance calculations during calving season."""
        heuristic = PredationRiskHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        
        # Close bear activity during calving season (May)
        close = heuristic.calculate(location, "2026-05-15", {
            "wolves_per_1000_elk": 2.0,
            "bear_activity_distance_miles": 1.0,
            "snow_depth_inches": 0.0,
            "security_habitat_percent": 35.0
        })
        
        # Distant bear activity during calving season
        distant = heuristic.calculate(location, "2026-05-15", {
            "wolves_per_1000_elk": 2.0,
            "bear_activity_distance_miles": 10.0,
            "snow_depth_inches": 0.0,
            "security_habitat_percent": 35.0
        })
        
        # During calving season, closer bears should reduce score
        assert close.score <= distant.score


class TestVegetationQualityHeuristicEdgeCases:
    """Test VegetationQualityHeuristic edge cases."""
    
    def test_vegetation_ndvi_calculations(self):
        """Test NDVI scoring at thresholds."""
        heuristic = VegetationQualityHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"  # Fall
        
        # Test fall thresholds: poor=0.3, good=0.5, excellent=0.65
        result = heuristic.calculate(location, date, {"ndvi": 0.65})
        assert result.score == 10.0
        assert result.status == "excellent"
        
        result = heuristic.calculate(location, date, {"ndvi": 0.5})
        assert result.score == 7.0
        assert result.status == "good"
        
        result = heuristic.calculate(location, date, {"ndvi": 0.3})
        assert result.score == 4.0
        assert result.status == "fair"
    
    def test_vegetation_seasonal_patterns(self):
        """Test seasonal NDVI variations."""
        heuristic = VegetationQualityHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        ndvi = 0.6
        
        # Spring (thresholds: poor=0.3, good=0.5, excellent=0.7)
        # 0.6 is between good and excellent, so scores 7.0 + (0.1/0.2)*3.0 = 8.5
        spring = heuristic.calculate(location, "2026-04-15", {"ndvi": ndvi})
        
        # Summer (thresholds: poor=0.5, good=0.7, excellent=0.85)
        # 0.6 is between poor and good, so scores 4.0 + (0.1/0.2)*3.0 = 5.5
        summer = heuristic.calculate(location, "2026-07-15", {"ndvi": ndvi})
        
        # Same NDVI scores differently by season due to different thresholds
        # Spring should score higher for 0.6 NDVI
        assert spring.score > summer.score
    
    def test_vegetation_irg_adjustments(self):
        """Test IRG (green-up rate) adjustments."""
        heuristic = VegetationQualityHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        
        # Spring with positive IRG (greening)
        spring_greening = heuristic.calculate(location, "2026-04-15", {
            "ndvi": 0.5,
            "irg": 0.05  # Actively greening
        })
        
        # Spring with no IRG
        spring_static = heuristic.calculate(location, "2026-04-15", {
            "ndvi": 0.5,
            "irg": 0.0
        })
        
        assert spring_greening.score > spring_static.score
