from datetime import datetime, timedelta
from typing import Dict
import numpy as np
from .base import BaseHeuristic, HeuristicScore

class VegetationQualityHeuristic(BaseHeuristic):
    """
    Heuristic 4: Green Wave Timing
    
    Elk track vegetation phenology (green-up in spring, quality in fall).
    Uses NDVI (Normalized Difference Vegetation Index) and IRG 
    (Instantaneous Rate of Green-up).
    """
    
    # NDVI thresholds by season
    NDVI_THRESHOLDS = {
        "spring": (0.3, 0.5, 0.7),    # poor, good, excellent
        "summer": (0.5, 0.7, 0.85),
        "fall": (0.3, 0.5, 0.65),
        "winter": (0.2, 0.3, 0.4)
    }
    
    def __init__(self, weight: float = 1.0):
        super().__init__("vegetation_quality", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        ndvi = context.get("ndvi")
        irg = context.get("irg", 0.0)  # Rate of change
        
        if ndvi is None:
            raise ValueError("NDVI data required for vegetation heuristic")
        
        # Get season
        dt = datetime.fromisoformat(date)
        season = self._get_season(dt.month)
        
        # Get thresholds for this season
        poor_thresh, good_thresh, excellent_thresh = self.NDVI_THRESHOLDS[season]
        
        # Calculate base score from NDVI
        if ndvi >= excellent_thresh:
            base_score = 10.0
            quality = "excellent"
        elif ndvi >= good_thresh:
            fraction = (ndvi - good_thresh) / (excellent_thresh - good_thresh)
            base_score = 7.0 + (fraction * 3.0)
            quality = "good"
        elif ndvi >= poor_thresh:
            fraction = (ndvi - poor_thresh) / (good_thresh - poor_thresh)
            base_score = 4.0 + (fraction * 3.0)
            quality = "fair"
        else:
            base_score = max(0, 4.0 * (ndvi / poor_thresh))
            quality = "poor"
        
        # Adjust for green wave timing (spring/fall)
        if season in ["spring", "fall"]:
            # Positive IRG (greening) is good in spring
            # Negative IRG (browning) is expected in fall
            if season == "spring" and irg > 0.01:
                base_score = min(10.0, base_score * 1.15)  # 15% bonus
                note = f"NDVI {ndvi:.2f}, actively greening (IRG: {irg:.3f})"
            elif season == "fall" and irg < -0.01:
                # Rapid browning is bad
                base_score *= 0.9
                note = f"NDVI {ndvi:.2f}, rapidly senescing"
            else:
                note = f"NDVI {ndvi:.2f}, {quality} forage quality"
        else:
            note = f"NDVI {ndvi:.2f}, {quality} forage quality"
        
        # Check for recent precipitation (improves forage)
        recent_precip = context.get("precip_last_7_days_inches", 0.0)
        if recent_precip > 0.5 and season in ["spring", "summer", "fall"]:
            base_score = min(10.0, base_score * 1.1)
            note += f" [recent precip: {recent_precip:.1f}in]"
        
        # Get land cover type for context
        land_cover = context.get("land_cover_type", "unknown")
        
        # Confidence depends on cloud cover and data recency
        cloud_cover = context.get("cloud_cover_percent", 0)
        data_age_days = context.get("ndvi_age_days", 0)
        
        confidence = 0.85
        if cloud_cover > 30:
            confidence -= 0.15
        if data_age_days > 16:  # Landsat revisit is 16 days
            confidence -= 0.10
        
        status = self._quality_to_status(quality)
        
        return HeuristicScore(
            score=base_score,
            confidence=max(0.5, confidence),
            status=status,
            note=note,
            raw_value=ndvi,
            metadata={
                "ndvi": ndvi,
                "irg": irg,
                "season": season,
                "land_cover": land_cover,
                "recent_precipitation": recent_precip,
                "cloud_cover": cloud_cover,
                "data_age_days": data_age_days
            }
        )
    
    def _get_season(self, month: int) -> str:
        """Determine season from month"""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"
    
    def _quality_to_status(self, quality: str) -> str:
        """Convert quality descriptor to status"""
        mapping = {
            "excellent": "excellent",
            "good": "good",
            "fair": "fair",
            "poor": "poor"
        }
        return mapping.get(quality, "fair")