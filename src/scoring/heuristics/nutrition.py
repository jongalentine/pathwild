from typing import Dict
from datetime import datetime
from .base import BaseHeuristic, HeuristicScore

class NutritionalConditionHeuristic(BaseHeuristic):
    """
    Heuristic 14: Nutritional Plane Threshold
    
    Predicts pre-winter body condition based on summer forage quality.
    Poor summer forage → low body fat → higher winterkill risk.
    """
    
    # Body fat thresholds (percent)
    EXCELLENT_THRESHOLD = 13.7
    GOOD_THRESHOLD = 11.0
    CRITICAL_THRESHOLD = 7.9
    
    def __init__(self, weight: float = 1.0):
        super().__init__("nutritional_condition", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        # Predict body condition from summer forage quality
        # This uses integrated NDVI from June-September
        
        dt = datetime.fromisoformat(date)
        
        # Get summer NDVI (integrated from Jun-Sep)
        summer_ndvi = context.get("summer_integrated_ndvi")
        if summer_ndvi is None:
            # Estimate from current NDVI if summer data not available
            current_ndvi = context.get("ndvi", 0.5)
            summer_ndvi = current_ndvi * 90  # Rough approximation
        
        # Get predation stress (affects body condition)
        predation_score = context.get("predation_score", 7.0)
        predation_stress = (10.0 - predation_score) / 10.0  # 0-1, higher = more stress
        
        # Pregnancy rate (typically 85-95%)
        pregnancy_rate = context.get("pregnancy_rate", 0.90)
        
        # Estimate body condition score (BCS) as proxy for body fat %
        # Scale: summer_ndvi correlates with fall body condition
        # Typical range: summer iNDVI 45-75 → body fat 9-15%
        
        estimated_body_fat = 9.0 + (summer_ndvi - 45) * 0.2
        estimated_body_fat *= (1 - predation_stress * 0.15)  # Stress reduces condition
        
        # Calculate score
        if estimated_body_fat >= self.EXCELLENT_THRESHOLD:
            score = 10.0
            status = "excellent"
            calf_survival = 80
            cow_survival = 97
            note = f"Excellent body condition (~{estimated_body_fat:.1f}% fat), " \
                   f"low winterkill risk"
        elif estimated_body_fat >= self.GOOD_THRESHOLD:
            fraction = (estimated_body_fat - self.GOOD_THRESHOLD) / \
                      (self.EXCELLENT_THRESHOLD - self.GOOD_THRESHOLD)
            score = 7.0 + (fraction * 3.0)
            status = "good"
            calf_survival = 65
            cow_survival = 93
            note = f"Good body condition (~{estimated_body_fat:.1f}% fat), " \
                   f"moderate winterkill risk"
        elif estimated_body_fat >= self.CRITICAL_THRESHOLD:
            fraction = (estimated_body_fat - self.CRITICAL_THRESHOLD) / \
                      (self.GOOD_THRESHOLD - self.CRITICAL_THRESHOLD)
            score = 4.0 + (fraction * 3.0)
            status = "fair"
            calf_survival = 45
            cow_survival = 88
            note = f"Fair body condition (~{estimated_body_fat:.1f}% fat), " \
                   f"elevated winterkill risk"
        else:
            score = max(1.0, 4.0 * (estimated_body_fat / self.CRITICAL_THRESHOLD))
            status = "poor"
            calf_survival = 35
            cow_survival = 85
            note = f"Poor body condition (~{estimated_body_fat:.1f}% fat), " \
                   f"high winterkill risk"
        
        # This affects population estimation (calf recruitment)
        recruitment_rate = (pregnancy_rate * calf_survival) / 100
        
        # Confidence depends on time of year
        if dt.month in [9, 10, 11]:  # Fall - best time to assess
            confidence = 0.75
        elif dt.month in [12, 1, 2, 3]:  # Winter - showing effects
            confidence = 0.80
        else:  # Spring/summer - harder to predict fall condition
            confidence = 0.60
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=estimated_body_fat,
            metadata={
                "estimated_body_fat_pct": estimated_body_fat,
                "summer_integrated_ndvi": summer_ndvi,
                "predation_stress": predation_stress,
                "pregnancy_rate": pregnancy_rate,
                "predicted_calf_survival": calf_survival,
                "predicted_cow_survival": cow_survival,
                "recruitment_rate": recruitment_rate
            }
        )