import pytest
from datetime import datetime
from src.scoring.heuristics import (
    RutBehaviorHeuristic,
    ElevationHeuristic,
    SecurityHabitatHeuristic,
    HuntingPressureHeuristic
)

class TestRutBehaviorHeuristic:
    """Tests for RutBehaviorHeuristic"""
    
    def test_rut_phase_detection_pre_rut(self):
        """Test rut phase detection for pre-rut period"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-10"  # Mid-September (pre-rut)
        context = {
            "elevation": 9000.0,
            "canopy_cover_percent": 50.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "pre_rut"
        assert result.metadata["is_rut_period"] is True
        assert "pre rut" in result.note.lower()
    
    def test_rut_phase_detection_peak_rut(self):
        """Test rut phase detection for peak rut period"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-20"  # Late September (peak rut)
        context = {
            "elevation": 9000.0,
            "canopy_cover_percent": 50.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "peak_rut"
        assert result.metadata["is_rut_period"] is True
        assert "peak" in result.note.lower()
    
    def test_rut_phase_detection_post_rut(self):
        """Test rut phase detection for post-rut period"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-20"  # Late October (post-rut)
        context = {
            "elevation": 9000.0,
            "canopy_cover_percent": 50.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "post_rut"
        assert result.metadata["is_rut_period"] is True
        assert "post rut" in result.note.lower()
    
    def test_rut_phase_detection_outside_rut(self):
        """Test rut phase detection for non-rut period"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-07-15"  # July (not rut period)
        context = {
            "elevation": 9000.0,
            "canopy_cover_percent": 50.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] is None
        assert result.metadata["is_rut_period"] is False
        assert "outside rut period" in result.note.lower()
    
    def test_terrain_boost_during_rut(self):
        """Test terrain feature boosts during rut"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-20"  # Peak rut
        context = {
            "elevation": 9000.0,
            "slope_degrees": 20.0,  # Moderate slope (potential ridgeline)
            "canopy_cover_percent": 50.0,  # Ecotone: partial cover
            "nlcd_code": 43,  # Mixed forest
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        # Should have terrain boost during peak rut
        assert result.metadata["terrain_boost"] > 0
        assert result.score > 7.0  # Base score is 7.0, should be boosted
    
    def test_elevation_boost_during_rut(self):
        """Test elevation preference during rut"""
        heuristic = RutBehaviorHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-20"  # Peak rut
        
        # Optimal elevation during rut (9,000-10,000 ft)
        context_optimal = {
            "elevation": 9500.0,
            "security_habitat_percent": 40.0
        }
        result_optimal = heuristic.calculate(location, date, context_optimal)
        
        # Suboptimal elevation
        context_low = {
            "elevation": 7000.0,
            "security_habitat_percent": 40.0
        }
        result_low = heuristic.calculate(location, date, context_low)
        
        # Optimal elevation should score higher during rut
        assert result_optimal.score > result_low.score


class TestElevationHeuristicRutIntegration:
    """Tests for ElevationHeuristic rut-aware adjustments"""
    
    def test_elevation_adjustment_during_pre_rut(self):
        """Test elevation range adjustment during pre-rut"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-10"  # Pre-rut
        context = {"elevation": 9500.0}  # Higher elevation preferred during rut
        
        result = heuristic.calculate(location, date, context)
        
        # Should include rut phase in metadata
        assert "rut_phase" in result.metadata
        assert result.metadata["rut_phase"] == "pre_rut"
        
        # Optimal range should be adjusted upward (9,000-10,000 ft during rut)
        opt_low, opt_high = result.metadata["optimal_range"]
        assert opt_low >= 9000  # Adjusted upward from normal September range
    
    def test_elevation_adjustment_during_peak_rut(self):
        """Test elevation range adjustment during peak rut"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-25"  # Peak rut
        context = {"elevation": 9500.0}
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "peak_rut"
        opt_low, opt_high = result.metadata["optimal_range"]
        assert opt_low >= 9000  # Higher elevations preferred during peak rut
    
    def test_elevation_adjustment_during_early_october_rut(self):
        """Test elevation adjustment during early October (still peak rut)"""
        heuristic = ElevationHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-05"  # Early October (peak rut)
        context = {"elevation": 9500.0}
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "peak_rut"
        opt_low, opt_high = result.metadata["optimal_range"]
        assert opt_low >= 9000  # Still adjusted during early October rut


class TestSecurityHabitatHeuristicRutIntegration:
    """Tests for SecurityHabitatHeuristic rut-aware adjustments"""
    
    def test_security_boost_during_pre_rut(self):
        """Test security habitat importance increase during pre-rut"""
        heuristic = SecurityHabitatHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-10"  # Pre-rut
        context = {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "pre_rut"
        assert "pre-rut" in result.note.lower() or "increased importance" in result.note.lower()
        # Security score should be boosted during rut
        assert result.score >= 7.0
    
    def test_security_boost_during_peak_rut(self):
        """Test security habitat importance increase during peak rut"""
        heuristic = SecurityHabitatHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-09-25"  # Peak rut
        context = {
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 40.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "peak_rut"
        assert "peak" in result.note.lower() or "heightened" in result.note.lower()
        # Peak rut should have higher boost than pre-rut
        assert result.score >= 7.0


class TestHuntingPressureHeuristicRutIntegration:
    """Tests for HuntingPressureHeuristic rut-aware adjustments"""
    
    def test_hunting_pressure_modifier_during_peak_rut(self):
        """Test hunting pressure modifier during peak rut"""
        heuristic = HuntingPressureHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-05"  # Peak rut during hunting season
        context = {
            "road_distance_miles": 1.0,  # Moderate distance
            "trail_distance_miles": 0.8,
            "security_habitat_percent": 30.0
        }
        
        result = heuristic.calculate(location, date, context)
        
        assert result.metadata["rut_phase"] == "peak_rut"
        # Should note rut period in metadata
        assert result.metadata["in_hunting_season"] is True
        
        # During peak rut with moderate access, should have slight boost
        # (still avoid very close access)
        if result.metadata["min_access_distance"] >= 0.5:
            assert "peak rut" in result.note.lower() or result.score > 5.0


class TestRutIntegration:
    """Integration tests for rut behavior across heuristics"""
    
    def test_rut_phase_consistency(self):
        """Test that rut phases are consistent across heuristics"""
        date = "2026-09-20"  # Peak rut
        
        rut_heuristic = RutBehaviorHeuristic()
        elevation_heuristic = ElevationHeuristic()
        security_heuristic = SecurityHabitatHeuristic()
        hunting_heuristic = HuntingPressureHeuristic()
        
        location = {"lat": 43.0, "lon": -110.0}
        context = {
            "elevation": 9000.0,
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "trail_distance_miles": 0.8,
            "security_habitat_percent": 40.0
        }
        
        rut_result = rut_heuristic.calculate(location, date, context)
        elevation_result = elevation_heuristic.calculate(location, date, context)
        security_result = security_heuristic.calculate(location, date, context)
        hunting_result = hunting_heuristic.calculate(location, date, context)
        
        # All should detect the same rut phase
        assert rut_result.metadata["rut_phase"] == "peak_rut"
        assert elevation_result.metadata["rut_phase"] == "peak_rut"
        assert security_result.metadata["rut_phase"] == "peak_rut"
        assert hunting_result.metadata["rut_phase"] == "peak_rut"
    
    def test_rut_vs_non_rut_scoring(self):
        """Test that rut period produces different scores than non-rut"""
        location = {"lat": 43.0, "lon": -110.0}
        context = {
            "elevation": 9000.0,
            "slope_degrees": 20.0,
            "canopy_cover_percent": 50.0,
            "road_distance_miles": 1.0,
            "security_habitat_percent": 40.0
        }
        
        # Peak rut date
        rut_date = "2026-09-25"
        
        # Non-rut date (summer)
        non_rut_date = "2026-07-15"
        
        rut_heuristic = RutBehaviorHeuristic()
        
        rut_result = rut_heuristic.calculate(location, rut_date, context)
        non_rut_result = rut_heuristic.calculate(location, non_rut_date, context)
        
        # Rut period should have different behavior
        assert rut_result.metadata["is_rut_period"] is True
        assert non_rut_result.metadata["is_rut_period"] is False
        
        # Scores may differ due to rut adjustments
        # (rut period should have terrain/elevation boosts)
        if rut_result.metadata["terrain_boost"] > 0:
            assert rut_result.score != non_rut_result.score
