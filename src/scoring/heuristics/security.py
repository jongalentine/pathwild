from typing import Dict
import numpy as np
from .base import BaseHeuristic, HeuristicScore

class SecurityHabitatHeuristic(BaseHeuristic):
    """
    Heuristic 10: Security Habitat Ratio
    
    Elk need escape terrain: steep slopes, dense cover, or remote areas.
    Populations remain stable when 30%+ of range is security habitat.
    """
    
    OPTIMAL_SECURITY_PCT = 40.0
    MINIMUM_SECURITY_PCT = 30.0
    CRITICAL_SECURITY_PCT = 20.0
    
    def __init__(self, weight: float = 1.0):
        super().__init__("security_habitat", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        # Security habitat criteria:
        # - Slope > 40 degrees, OR
        # - Canopy cover > 70%, OR
        # - Distance to road > 2.5 miles AND distance to trail > 1.5 miles
        
        slope = context.get("slope_degrees", 0)
        canopy_cover = context.get("canopy_cover_percent", 0)
        road_distance = context.get("road_distance_miles", 0)
        trail_distance = context.get("trail_distance_miles", road_distance)
        
        # Check if this location IS security habitat
        is_security = (
            slope > 40 or
            canopy_cover > 70 or
            (road_distance > 2.5 and trail_distance > 1.5)
        )
        
        # Get security habitat percentage in surrounding area (1km radius)
        security_pct = context.get("security_habitat_percent", 
                                   50 if is_security else 20)
        
        # Calculate score based on security percentage in area
        if security_pct >= self.OPTIMAL_SECURITY_PCT:
            score = 10.0
            status = "excellent"
            note = f"High security habitat ({security_pct:.0f}% in area)"
        elif security_pct >= self.MINIMUM_SECURITY_PCT:
            fraction = (security_pct - self.MINIMUM_SECURITY_PCT) / \
                      (self.OPTIMAL_SECURITY_PCT - self.MINIMUM_SECURITY_PCT)
            score = 7.0 + (fraction * 3.0)
            status = "good"
            note = f"Adequate security habitat ({security_pct:.0f}% in area)"
        elif security_pct >= self.CRITICAL_SECURITY_PCT:
            fraction = (security_pct - self.CRITICAL_SECURITY_PCT) / \
                      (self.MINIMUM_SECURITY_PCT - self.CRITICAL_SECURITY_PCT)
            score = 4.0 + (fraction * 3.0)
            status = "fair"
            note = f"Marginal security habitat ({security_pct:.0f}% in area)"
        else:
            score = max(1.0, 4.0 * (security_pct / self.CRITICAL_SECURITY_PCT))
            status = "poor"
            note = f"Low security habitat ({security_pct:.0f}% in area)"
        
        # Bonus if this specific location is security habitat
        if is_security:
            score = min(10.0, score * 1.1)
            note += " [location is secure]"
        
        # Add qualifiers about what makes it secure
        security_features = []
        if slope > 40:
            security_features.append(f"steep slope ({slope:.0f}°)")
        if canopy_cover > 70:
            security_features.append(f"dense cover ({canopy_cover:.0f}%)")
        if road_distance > 2.5:
            security_features.append("remote")
        
        if security_features:
            note += f" [{', '.join(security_features)}]"
        
        confidence = 0.85  # Terrain data is reliable
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=security_pct,
            metadata={
                "security_percent": security_pct,
                "is_security_habitat": is_security,
                "slope_degrees": slope,
                "canopy_cover": canopy_cover,
                "road_distance": road_distance,
                "security_features": security_features
            }
        )