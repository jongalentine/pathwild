# src/scoring/heuristics/access.py

from datetime import datetime
from typing import Dict
import numpy as np
from .base import BaseHeuristic, HeuristicScore

class HuntingPressureHeuristic(BaseHeuristic):
    """
    Heuristic 6: Hunt Pressure Displacement
    
    During hunting season, elk avoid accessible areas and concentrate
    in security habitat far from roads and trails.
    """
    
    ROAD_BUFFER_OPTIMAL = 1.5  # miles - elk feel safe
    ROAD_BUFFER_MINIMUM = 0.5   # miles - elk avoid if possible
    
    # Hunting season dates (Wyoming example)
    HUNTING_SEASONS = {
        "archery": (datetime(2026, 9, 1), datetime(2026, 9, 30)),
        "rifle_1": (datetime(2026, 10, 1), datetime(2026, 10, 14)),
        "rifle_2": (datetime(2026, 10, 15), datetime(2026, 10, 31)),
        "rifle_3": (datetime(2026, 11, 1), datetime(2026, 11, 14))
    }
    
    def __init__(self, weight: float = 1.0):
        super().__init__("hunting_pressure", weight)
    
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        road_distance = context.get("road_distance_miles")
        trail_distance = context.get("trail_distance_miles")
        
        if road_distance is None:
            raise ValueError("Road distance data required")
        
        # Determine if we're in hunting season
        dt = datetime.fromisoformat(date)
        in_hunting_season, season_name = self._check_hunting_season(dt)
        
        # Calculate accessibility score (inverse of what elk want during hunting)
        min_access_distance = min(road_distance, trail_distance if trail_distance else road_distance)
        
        if in_hunting_season:
            # During hunting: farther from access = better
            if min_access_distance >= self.ROAD_BUFFER_OPTIMAL:
                score = 10.0
                status = "excellent"
                note = f"Remote location ({min_access_distance:.1f}mi from access), " \
                       f"low hunting pressure [{season_name}]"
            elif min_access_distance >= self.ROAD_BUFFER_MINIMUM:
                fraction = (min_access_distance - self.ROAD_BUFFER_MINIMUM) / \
                          (self.ROAD_BUFFER_OPTIMAL - self.ROAD_BUFFER_MINIMUM)
                score = 5.0 + (fraction * 5.0)  # 5-10 range
                status = "good" if score > 7 else "fair"
                note = f"Moderate access ({min_access_distance:.1f}mi), " \
                       f"moderate hunting pressure [{season_name}]"
            else:
                # Too close to roads during hunting
                score = max(1.0, 5.0 * (min_access_distance / self.ROAD_BUFFER_MINIMUM))
                status = "poor"
                note = f"High access ({min_access_distance:.1f}mi from road), " \
                       f"heavy hunting pressure [{season_name}]"
        else:
            # Outside hunting season: accessibility matters less
            # But elk still prefer some security
            if min_access_distance >= 0.5:
                score = 9.0
                status = "good"
                note = f"Good security ({min_access_distance:.1f}mi from access), " \
                       f"no hunting pressure"
            else:
                score = 7.0
                status = "fair"
                note = f"Near access ({min_access_distance:.1f}mi), " \
                       f"but no hunting pressure"
        
        # Factor in security habitat availability
        security_cover = context.get("security_habitat_percent", 50)
        if security_cover < 30 and in_hunting_season:
            score *= 0.8  # 20% penalty for low security cover
            note += " [limited escape terrain]"
        
        # Weekend vs weekday (during hunting season)
        if in_hunting_season and dt.weekday() in [5, 6]:  # Sat, Sun
            score *= 0.85  # 15% penalty for weekend pressure
            note += " [weekend]"
        
        confidence = 0.80  # Generally reliable predictor
        
        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=min_access_distance,
            metadata={
                "road_distance_miles": road_distance,
                "trail_distance_miles": trail_distance,
                "min_access_distance": min_access_distance,
                "in_hunting_season": in_hunting_season,
                "season_name": season_name if in_hunting_season else "closed",
                "security_cover_percent": security_cover,
                "is_weekend": dt.weekday() in [5, 6]
            }
        )
    
    def _check_hunting_season(self, date: datetime) -> tuple:
        """Check if date falls in hunting season"""
        for season_name, (start, end) in self.HUNTING_SEASONS.items():
            if start <= date <= end:
                return True, season_name
        return False, None