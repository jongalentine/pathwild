from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class HeuristicScore:
    """Standard output from any heuristic"""
    score: float  # 0-10 scale
    confidence: float  # 0-1 scale
    status: str  # "excellent", "good", "fair", "poor", "critical"
    note: str  # Human-readable explanation
    raw_value: Any  # Original value before scoring
    metadata: Dict[str, Any]  # Additional context
    
class BaseHeuristic(ABC):
    """Base class for all heuristics"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.default_weight = weight
    
    @abstractmethod
    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        """Calculate the heuristic score for a location and date
        
        Args:
            location: {"lat": float, "lon": float} or pixel coordinates
            date: ISO format date string
            context: Additional data (elevation grid, weather, etc.)
        
        Returns:
            HeuristicScore object
        """
        pass
    
    def normalize_score(self, raw_score: float, min_val: float, max_val: float, 
                       inverse: bool = False) -> float:
        """Normalize raw value to 0-10 scale
        
        Args:
            raw_score: The raw value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            inverse: If True, higher raw values = lower scores
        """
        if raw_score < min_val:
            raw_score = min_val
        if raw_score > max_val:
            raw_score = max_val
        
        normalized = (raw_score - min_val) / (max_val - min_val)
        
        if inverse:
            normalized = 1 - normalized
        
        return normalized * 10
    
    def get_status(self, score: float) -> str:
        """Convert score to categorical status"""
        if score >= 9.0:
            return "excellent"
        elif score >= 7.0:
            return "good"
        elif score >= 5.0:
            return "fair"
        elif score >= 3.0:
            return "poor"
        else:
            return "critical"
