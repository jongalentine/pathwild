from datetime import datetime, timedelta
from typing import Dict
from .base import BaseHeuristic, HeuristicScore

class WinterSeverityHeuristic(BaseHeuristic):
    """
    Heuristic 12: Cumulative Winter Severity Index (WSI)
    
    Predicts winterkill risk based on cumulative snow depth × temperature stress.
    Higher WSI = more winterkill = lower population carrying capacity.
    """
    
    # WSI thresholds
    WSI_MINIMAL = 2000
    WSI_MODERATE = 4000
    WSI_HIGH = 6000
    
    def __init__(self, weight: float = 1.0):
        super().__init__("winter_severity", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        dt = datetime.fromisoformat(date)
        
        # Only calculate WSI during/after winter (Nov-Apr)
        if dt.month < 5 or dt.month > 10:
            # Calculate cumulative WSI from Nov 1 to current date
            wsi = self._calculate_cumulative_wsi(location, dt, context)
        else:
            # Use previous winter's WSI for summer/fall predictions
            # This affects current population size (winterkill already happened)
            previous_winter_end = datetime(dt.year, 4, 30)
            wsi = self._calculate_cumulative_wsi(location, previous_winter_end, context)
        
        # Convert WSI to score
        # Low WSI = high score (good for elk)
        # High WSI = low score (high mortality)
        if wsi < self.WSI_MINIMAL:
            score = 10.0
            status = "excellent"
            mortality_estimate = "<5%"
            note = f"Mild winter (WSI: {wsi:.0f}), minimal winterkill expected"
        elif wsi < self.WSI_MODERATE:
            # Linear from 10 to 7
            fraction = (wsi - self.WSI_MINIMAL) / (self.WSI_MODERATE - self.WSI_MINIMAL)
            score = 10.0 - (fraction * 3.0)
            status = "good"
            mortality_estimate = "5-15%"
            note = f"Moderate winter (WSI: {wsi:.0f}), some winterkill expected"
        elif wsi < self.WSI_HIGH:
            # Linear from 7 to 3
            fraction = (wsi - self.WSI_MODERATE) / (self.WSI_HIGH - self.WSI_MODERATE)
            score = 7.0 - (fraction * 4.0)
            status = "fair"
            mortality_estimate = "15-30%"
            note = f"Severe winter (WSI: {wsi:.0f}), high winterkill expected"
        else:
            # Catastrophic winter
            score = max(0, 3.0 - ((wsi - self.WSI_HIGH) / 2000) * 3.0)
            status = "poor" if score > 1 else "critical"
            mortality_estimate = ">30%"
            note = f"Extreme winter (WSI: {wsi:.0f}), severe winterkill expected"
        
        # Confidence depends on how far into winter we are
        if dt.month in [11, 12, 1, 2, 3, 4]:
            # During winter - historical + short-term forecast
            confidence = 0.80
        else:
            # Using previous winter - historical data
            confidence = 0.95
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=wsi,
            metadata={
                "wsi": wsi,
                "mortality_estimate": mortality_estimate,
                "winter_severity": self._classify_winter(wsi)
            }
        )
    
    def _calculate_cumulative_wsi(self, location: Dict, end_date: datetime, 
                                   context: Dict) -> float:
        """Calculate WSI from Nov 1 to end_date"""
        start_date = datetime(end_date.year if end_date.month >= 11 else end_date.year - 1, 11, 1)
        
        cumulative_wsi = 0.0
        current_date = start_date
        
        while current_date <= end_date:
            # Get weather data for this day
            daily_snow_depth = self._get_snow_depth(location, current_date, context)
            daily_temp = self._get_temperature(location, current_date, context)
            
            # Calculate temperature severity multiplier
            if daily_temp > 20:
                temp_severity = 0
            elif daily_temp >= 0:
                temp_severity = (20 - daily_temp) / 20
            else:  # Below zero
                temp_severity = 2.0
            
            # Daily WSI = snow depth × temperature severity
            daily_wsi = daily_snow_depth * temp_severity
            cumulative_wsi += daily_wsi
            
            current_date += timedelta(days=1)
        
        return cumulative_wsi
    
    def _classify_winter(self, wsi: float) -> str:
        """Classify winter severity"""
        if wsi < self.WSI_MINIMAL:
            return "mild"
        elif wsi < self.WSI_MODERATE:
            return "moderate"
        elif wsi < self.WSI_HIGH:
            return "severe"
        else:
            return "extreme"
    
    def _get_snow_depth(self, location, date, context):
        """Get snow depth for location/date (inches)"""
        # Placeholder - would query SNOTEL or snow grid
        return 12.0
    
    def _get_temperature(self, location, date, context):
        """Get temperature for location/date (°F)"""
        # Placeholder - would query weather data
        return 25.0