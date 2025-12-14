from datetime import datetime
from typing import Dict
from .base import BaseHeuristic, HeuristicScore

class SnowConditionsHeuristic(BaseHeuristic):
    """
    Heuristic 2: Snow Depth Threshold
    
    Snow depth affects elk accessibility and movement:
    - 0-6 inches: Optimal, easy access to forage
    - 6-18 inches: Acceptable, some difficulty
    - 18-30 inches: Difficult, triggers migration
    - >30 inches: Critical, elk avoid or cannot access
    """
    
    OPTIMAL_MAX = 6.0  # inches
    ACCEPTABLE_MAX = 18.0
    DIFFICULT_MAX = 30.0
    
    def __init__(self, weight: float = 1.0):
        super().__init__("snow_conditions", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        snow_depth = context.get("snow_depth_inches", 0.0)
        
        # Get elevation for context
        elevation = context.get("elevation", 0)
        
        # Calculate base score
        if snow_depth <= self.OPTIMAL_MAX:
            score = 10.0
            status = "excellent"
            note = f"Minimal snow ({snow_depth:.1f}in), easy forage access"
        elif snow_depth <= self.ACCEPTABLE_MAX:
            # Linear decrease from 10 to 7
            fraction = (snow_depth - self.OPTIMAL_MAX) / \
                      (self.ACCEPTABLE_MAX - self.OPTIMAL_MAX)
            score = 10.0 - (fraction * 3.0)
            status = "good"
            note = f"Moderate snow ({snow_depth:.1f}in), some difficulty accessing forage"
        elif snow_depth <= self.DIFFICULT_MAX:
            # Linear decrease from 7 to 3
            fraction = (snow_depth - self.ACCEPTABLE_MAX) / \
                      (self.DIFFICULT_MAX - self.ACCEPTABLE_MAX)
            score = 7.0 - (fraction * 4.0)
            status = "fair"
            note = f"Deep snow ({snow_depth:.1f}in), likely triggering migration"
        else:
            # Critical snow depth
            score = max(0, 3.0 - ((snow_depth - self.DIFFICULT_MAX) / 10.0))
            status = "poor" if score > 1 else "critical"
            note = f"Extreme snow ({snow_depth:.1f}in), area likely abandoned"
        
        # Check for snow crusting (from context)
        snow_crust = context.get("snow_crust_detected", False)
        if snow_crust and snow_depth > self.OPTIMAL_MAX:
            score *= 0.7  # 30% penalty for crusting
            note += " [crusting detected]"
        
        # Confidence depends on forecast horizon
        dt = datetime.fromisoformat(date)
        today = datetime.now().date()
        days_out = (dt.date() - today).days
        
        if days_out <= 3:
            confidence = 0.90
        elif days_out <= 7:
            confidence = 0.75
        else:
            confidence = 0.60
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=snow_depth,
            metadata={
                "snow_depth_inches": snow_depth,
                "elevation_ft": elevation,
                "snow_crust": snow_crust,
                "forecast_days_out": days_out
            }
        )