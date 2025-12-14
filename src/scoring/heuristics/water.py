import numpy as np
from typing import Dict
from .base import BaseHeuristic, HeuristicScore

class WaterDistanceHeuristic(BaseHeuristic):
    """
    Heuristic 3: Distance to Water Rule
    
    In October, elk will be within 1 mile of permanent water 90% of the time.
    Score decreases with distance, becomes critical beyond 2 miles.
    """
    
    OPTIMAL_DISTANCE = 0.25  # miles (quarter mile is ideal)
    ACCEPTABLE_DISTANCE = 1.0  # miles
    CRITICAL_DISTANCE = 2.0  # miles
    
    def __init__(self, weight: float = 1.0):
        super().__init__("water_distance", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        # Get distance to nearest water
        water_distance = context.get("water_distance_miles")
        if water_distance is None:
            water_distance = self._calculate_water_distance(location, context)
        
        # Get water source reliability (permanent vs ephemeral)
        water_reliability = context.get("water_reliability", 1.0)  # 0-1 scale
        
        # Calculate base score
        if water_distance <= self.OPTIMAL_DISTANCE:
            base_score = 10.0
            status = "excellent"
            note = f"Within {water_distance:.2f} miles of water (optimal)"
        elif water_distance <= self.ACCEPTABLE_DISTANCE:
            # Linear decrease from 10 to 7
            fraction = (water_distance - self.OPTIMAL_DISTANCE) / \
                      (self.ACCEPTABLE_DISTANCE - self.OPTIMAL_DISTANCE)
            base_score = 10.0 - (fraction * 3.0)
            status = "good"
            note = f"Within {water_distance:.2f} miles of water (acceptable)"
        elif water_distance <= self.CRITICAL_DISTANCE:
            # Linear decrease from 7 to 3
            fraction = (water_distance - self.ACCEPTABLE_DISTANCE) / \
                      (self.CRITICAL_DISTANCE - self.ACCEPTABLE_DISTANCE)
            base_score = 7.0 - (fraction * 4.0)
            status = "fair"
            note = f"{water_distance:.2f} miles from water (marginal)"
        else:
            # Beyond critical distance
            base_score = max(0, 3.0 - (water_distance - self.CRITICAL_DISTANCE) * 1.0)
            status = "poor" if base_score > 1 else "critical"
            note = f"{water_distance:.2f} miles from water (too far)"
        
        # Adjust score based on water reliability
        # Ephemeral water sources reduce score
        adjusted_score = base_score * water_reliability
        
        # Confidence depends on water source mapping quality
        confidence = 0.85 if water_reliability > 0.8 else 0.70
        
        return HeuristicScore(
            score=adjusted_score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=water_distance,
            metadata={
                "distance_miles": water_distance,
                "water_reliability": water_reliability,
                "base_score": base_score,
                "reliability_adjusted": adjusted_score != base_score
            }
        )
    
    def _calculate_water_distance(self, location: Dict, context: Dict) -> float:
        """Calculate distance to nearest water source"""
        water_sources = context.get("water_sources")
        if water_sources is None:
            raise ValueError("No water source data available")
        
        # Calculate distance to each water source
        lat, lon = location["lat"], location["lon"]
        min_distance = float('inf')
        
        for water in water_sources:
            dist = self._haversine_distance(
                lat, lon, 
                water["lat"], water["lon"]
            )
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth (in miles)"""
        R = 3959  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c