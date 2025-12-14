from datetime import datetime
from typing import Dict
from .base import BaseHeuristic, HeuristicScore

class ElevationHeuristic(BaseHeuristic):
    """
    Heuristic 1: Elevation Band Rule
    
    Elk prefer different elevations by season:
    - Summer (Jun-Aug): 9,000-11,000 ft (optimal: 9,500-10,500)
    - Fall (Sep-Nov): 8,000-10,000 ft (optimal: 8,500-9,500)
    - Winter (Dec-Mar): 6,000-8,000 ft (optimal: 6,500-7,500)
    - Spring (Apr-May): 7,000-9,000 ft (optimal: 7,500-8,500)
    """
    
    # Optimal elevation ranges by month (in feet)
    MONTHLY_RANGES = {
        1: (6500, 7500, 6000, 8000),   # Jan: optimal_low, optimal_high, min, max
        2: (6500, 7500, 6000, 8000),   # Feb
        3: (7000, 8000, 6500, 8500),   # Mar
        4: (7500, 8500, 7000, 9000),   # Apr
        5: (8000, 9000, 7500, 9500),   # May
        6: (9500, 10500, 9000, 11000), # Jun
        7: (9500, 10500, 9000, 11000), # Jul
        8: (9500, 10500, 9000, 11000), # Aug
        9: (9000, 10000, 8500, 10500), # Sep
        10: (8500, 9500, 8000, 10000), # Oct - YOUR HUNT!
        11: (8000, 9000, 7500, 9500),  # Nov
        12: (7000, 8000, 6500, 8500),  # Dec
    }
    
    def __init__(self, weight: float = 1.0):
        super().__init__("elevation", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        # Get elevation at location
        elevation = context.get("elevation")
        if elevation is None:
            # Fall back to DEM lookup if not provided
            elevation = self._get_elevation_from_dem(location, context)
        
        # Parse date to get month
        dt = datetime.fromisoformat(date)
        month = dt.month
        
        # Get optimal range for this month
        opt_low, opt_high, min_accept, max_accept = self.MONTHLY_RANGES[month]
        
        # Calculate score
        if opt_low <= elevation <= opt_high:
            # In optimal range
            score = 10.0
            status = "excellent"
            note = f"Elevation {elevation:.0f}ft is optimal for {dt.strftime('%B')}"
        elif min_accept <= elevation <= max_accept:
            # In acceptable range
            # Score decreases linearly as you move away from optimal
            if elevation < opt_low:
                distance = opt_low - elevation
                max_distance = opt_low - min_accept
            else:
                distance = elevation - opt_high
                max_distance = max_accept - opt_high
            
            score = 10.0 - (distance / max_distance) * 3.0  # Score 7-10
            status = "good"
            note = f"Elevation {elevation:.0f}ft is acceptable for {dt.strftime('%B')}"
        else:
            # Outside acceptable range
            if elevation < min_accept:
                distance = min_accept - elevation
                score = max(0, 7.0 - (distance / 1000) * 5)  # Decreases with distance
                note = f"Elevation {elevation:.0f}ft is below preferred range"
            else:
                distance = elevation - max_accept
                score = max(0, 7.0 - (distance / 1000) * 5)
                note = f"Elevation {elevation:.0f}ft is above preferred range"
            
            status = "poor" if score > 3 else "critical"
        
        # Confidence is high for elevation (it doesn't change!)
        confidence = 0.95
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=elevation,
            metadata={
                "elevation_ft": elevation,
                "month": dt.strftime('%B'),
                "optimal_range": (opt_low, opt_high),
                "acceptable_range": (min_accept, max_accept)
            }
        )
    
    def _get_elevation_from_dem(self, location: Dict, context: Dict) -> float:
        """Look up elevation from Digital Elevation Model"""
        dem = context.get("dem_grid")
        if dem is None:
            raise ValueError("No elevation data available")
        
        # Bilinear interpolation to get elevation at exact lat/lon
        lat, lon = location["lat"], location["lon"]
        elevation = self._bilinear_interpolate(dem, lat, lon)
        
        return elevation
    
    def _bilinear_interpolate(self, grid, lat, lon):
        """Simple bilinear interpolation (you'd use rasterio in practice)"""
        # Simplified placeholder - use rasterio.sample() in real implementation
        return 8500.0  # Placeholder