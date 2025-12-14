from typing import Dict
from datetime import datetime
from .base import BaseHeuristic, HeuristicScore

class PredationRiskHeuristic(BaseHeuristic):
    """
    Heuristics 8, 9, 11: Predation Patterns
    
    Combines:
    - Wolf pack density (Heuristic 8)
    - Calving ground vulnerability to bears (Heuristic 9)
    - Predator hunting efficiency with snow (Heuristic 11)
    """
    
    WOLF_THRESHOLD = 5.0  # wolves per 1000 elk
    BEAR_BUFFER_MILES = 3.0
    OPTIMAL_SNOW_FOR_WOLVES = (12, 18)  # inches
    
    def __init__(self, weight: float = 1.0):
        super().__init__("predation_risk", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        # Get predator data
        wolf_density = context.get("wolves_per_1000_elk", 0.0)
        bear_distance = context.get("bear_activity_distance_miles", 10.0)
        snow_depth = context.get("snow_depth_inches", 0.0)
        
        # Date-specific risks
        dt = datetime.fromisoformat(date)
        month = dt.month
        
        # Start with neutral score
        base_score = 7.0
        risk_factors = []
        
        # 1. Wolf predation risk
        if wolf_density > self.WOLF_THRESHOLD:
            wolf_penalty = min(3.0, (wolf_density - self.WOLF_THRESHOLD) * 0.3)
            base_score -= wolf_penalty
            risk_factors.append(f"high wolf density ({wolf_density:.1f}/1000)")
        elif wolf_density > self.WOLF_THRESHOLD * 0.5:
            wolf_penalty = 0.5
            base_score -= wolf_penalty
            risk_factors.append(f"moderate wolves ({wolf_density:.1f}/1000)")
        
        # 2. Wolf hunting efficiency based on snow depth
        if self.OPTIMAL_SNOW_FOR_WOLVES[0] <= snow_depth <= self.OPTIMAL_SNOW_FOR_WOLVES[1]:
            snow_penalty = 1.5
            base_score -= snow_penalty
            risk_factors.append(f"optimal snow for wolf hunting ({snow_depth:.0f}in)")
        
        # 3. Bear predation risk (calving season: May-June)
        if month in [5, 6]:
            if bear_distance < self.BEAR_BUFFER_MILES:
                bear_penalty = 2.0 * (1 - bear_distance / self.BEAR_BUFFER_MILES)
                base_score -= bear_penalty
                risk_factors.append(f"near bear activity ({bear_distance:.1f}mi) during calving")
        
        # 4. Security habitat reduces risk
        security_pct = context.get("security_habitat_percent", 30)
        if security_pct > 40:
            security_bonus = 1.0
            base_score = min(10.0, base_score + security_bonus)
        
        # Ensure score stays in bounds
        score = max(0.0, min(10.0, base_score))
        
        # Determine status
        if score >= 8.0:
            status = "excellent"
            note = "Low predation risk"
        elif score >= 6.0:
            status = "good"
            note = "Moderate predation risk"
        elif score >= 4.0:
            status = "fair"
            note = "Elevated predation risk"
        else:
            status = "poor"
            note = "High predation risk"
        
        if risk_factors:
            note += f": {', '.join(risk_factors)}"
        
        # Confidence depends on predator monitoring data quality
        wolf_data_quality = context.get("wolf_data_quality", 0.7)
        bear_data_quality = context.get("bear_data_quality", 0.6)
        confidence = (wolf_data_quality + bear_data_quality) / 2
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value={"wolves": wolf_density, "bears": bear_distance},
            metadata={
                "wolf_density": wolf_density,
                "bear_distance_miles": bear_distance,
                "snow_depth_inches": snow_depth,
                "security_percent": security_pct,
                "is_calving_season": month in [5, 6],
                "risk_factors": risk_factors
            }
        )