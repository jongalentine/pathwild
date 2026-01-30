from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
from .base import BaseHeuristic, HeuristicScore

class RutBehaviorHeuristic(BaseHeuristic):
    """
    Heuristic 12: Rut Behavior Patterns
    
    During rut (mating season), elk behavior changes significantly:
    - Bulls increase movement and range widely to form harems
    - Preference for elevated display sites (ridgelines, meadow edges)
    - Increased activity at dawn/dusk
    - Less predictable patterns due to social behavior
    
    Rut timing (Wyoming-specific):
    - Pre-rut: September 1-15
    - Peak rut: September 15 - October 10
    - Post-rut: October 10-31
    """
    
    # Rut timing windows (Wyoming-specific)
    RUT_PHASES = {
        "pre_rut": (9, 1, 9, 15),   # month_start, day_start, month_end, day_end
        "peak_rut": (9, 15, 10, 10),
        "post_rut": (10, 10, 10, 31)
    }
    
    # Rut-preferred terrain feature boosts (multipliers)
    RUT_TERRAIN_BOOSTS = {
        "meadow_edge": 1.2,      # 20% score boost
        "ridgeline": 1.15,        # 15% score boost
        "elevated_viewpoint": 1.1, # 10% score boost
        "aspen_stand": 1.1,       # 10% score boost
        "ecotone": 1.15           # 15% score boost
    }
    
    def __init__(self, weight: float = 1.0):
        super().__init__("rut_behavior", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        """
        Calculate rut behavior score based on rut phase and terrain features.
        
        Args:
            location: Location dict with lat/lon
            date: ISO format date string
            context: Context dict with terrain, vegetation, and other features
        
        Returns:
            HeuristicScore with rut behavior adjustments
        """
        # Parse date
        dt = datetime.fromisoformat(date)
        month = dt.month
        day = dt.day
        
        # Determine rut phase
        rut_phase = self._get_rut_phase(month, day)
        
        # If not in rut period, return neutral score
        if rut_phase is None:
            return HeuristicScore(
                score=7.0,  # Neutral
                confidence=0.9,
                status="good",
                note="Outside rut period - no rut behavior adjustments",
                raw_value=None,
                metadata={
                    "rut_phase": None,
                    "is_rut_period": False
                }
            )
        
        # Base score starts neutral
        base_score = 7.0
        adjustments = []
        
        # Get terrain features that favor rut behavior
        terrain_boost = self._calculate_terrain_boost(context)
        
        if terrain_boost > 0:
            # Apply terrain boost based on rut phase intensity
            if rut_phase == "peak_rut":
                boost_multiplier = 1.0  # Full boost during peak
            elif rut_phase == "pre_rut":
                boost_multiplier = 0.7  # Partial boost during pre-rut
            else:  # post_rut
                boost_multiplier = 0.5  # Reduced boost post-rut
            
            score_adjustment = terrain_boost * boost_multiplier
            base_score += score_adjustment
            adjustments.append(f"rut terrain boost ({terrain_boost:.1f})")
        
        # Account for elevation preference during rut (higher elevations preferred)
        elevation = context.get("elevation")
        if elevation:
            elevation_boost = self._calculate_elevation_boost(elevation, rut_phase)
            if elevation_boost != 0:
                base_score += elevation_boost
                adjustments.append(f"rut elevation preference")
        
        # Account for security habitat importance during rut (elk more vulnerable)
        security_pct = context.get("security_habitat_percent", 0)
        security_boost = self._calculate_security_boost(security_pct, rut_phase)
        if security_boost != 0:
            base_score += security_boost
            adjustments.append("security habitat during rut")
        
        # Ensure score stays in bounds
        score = max(0.0, min(10.0, base_score))
        
        # Determine status and note
        status = self.get_status(score)
        note = f"{rut_phase.replace('_', ' ').title()}"
        if adjustments:
            note += f" - {', '.join(adjustments)}"
        else:
            note += " - standard rut period"
        
        # Confidence is high for rut timing, moderate for terrain detection
        confidence = 0.85 if rut_phase else 0.9
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=rut_phase,
            metadata={
                "rut_phase": rut_phase,
                "is_rut_period": True,
                "terrain_boost": terrain_boost,
                "elevation_boost": elevation_boost if elevation else None,
                "security_boost": security_boost
            }
        )
    
    def _get_rut_phase(self, month: int, day: int) -> Optional[str]:
        """Determine rut phase based on date"""
        if month == 9:  # September
            if day >= 1 and day < 15:
                return "pre_rut"
            elif day >= 15:
                return "peak_rut"
        elif month == 10:  # October
            if day < 10:
                return "peak_rut"
            elif day >= 10:
                return "post_rut"
        return None
    
    def _calculate_terrain_boost(self, context: Dict) -> float:
        """
        Calculate terrain boost based on rut-preferred features.
        
        Checks for:
        - Meadow edges (proximity to meadow-forest boundary)
        - Ridgelines (elevated terrain features)
        - Elevated viewpoints (high elevation with good visibility)
        - Aspen stands (preferred vegetation type)
        - Ecotones (edge habitats)
        """
        boost = 0.0
        
        # Check for meadow edge (would need landcover data to detect)
        # For now, use open areas (low canopy) near forest (high canopy)
        canopy_cover = context.get("canopy_cover_percent", 0)
        if 30 <= canopy_cover <= 60:  # Ecotone: partial cover
            boost += 0.8  # Partial boost for ecotone/meadow edge
        
        # Check for ridgeline (would need slope/aspect analysis)
        # Elevated areas with moderate slopes suggest ridgelines
        elevation = context.get("elevation", 0)
        slope = context.get("slope_degrees", 0)
        if elevation > 8500 and 15 <= slope <= 35:  # Elevated, moderate slope
            boost += 0.6  # Partial boost for potential ridgeline
        
        # Check for aspen stands (would need vegetation type data)
        # For now, check if in mixed forest (often includes aspen)
        nlcd_code = context.get("nlcd_code")
        if nlcd_code in [41, 42, 43]:  # Deciduous, evergreen, mixed forest
            boost += 0.3  # Small boost for forested areas
        
        # Cap total boost at reasonable level
        return min(2.5, boost)
    
    def _calculate_elevation_boost(self, elevation: float, rut_phase: str) -> float:
        """
        Calculate elevation boost during rut.
        
        During rut, bulls prefer higher elevations (8,500-10,000 ft)
        for display sites with better visibility and sound propagation.
        """
        # Optimal elevation range during rut
        optimal_low = 8500
        optimal_high = 10000
        
        if optimal_low <= elevation <= optimal_high:
            # Within optimal range
            if rut_phase == "peak_rut":
                return 0.5  # Small boost for optimal elevation during peak
            return 0.3
        elif 8000 <= elevation < optimal_low or optimal_high < elevation <= 10500:
            # Near optimal range
            if rut_phase == "peak_rut":
                return 0.2
            return 0.1
        
        return 0.0
    
    def _calculate_security_boost(self, security_pct: float, rut_phase: str) -> float:
        """
        Calculate security habitat boost during rut.
        
        During rut, elk are more vulnerable and predictable, so security
        habitat becomes even more important.
        """
        if security_pct >= 40:
            if rut_phase == "peak_rut":
                return 0.6  # Higher boost during peak rut vulnerability
            return 0.4
        elif security_pct >= 30:
            if rut_phase == "peak_rut":
                return 0.3
            return 0.2
        
        return 0.0
