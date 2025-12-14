from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class AggregateScore:
    """Combined score from all heuristics"""
    total_score: float  # 0-100 scale
    confidence: float  # 0-1 scale
    factor_scores: Dict[str, Dict]  # Breakdown by heuristic
    limiting_factor: str  # Which factor is holding back the score
    best_feature: str  # Which factor is driving the score
    habitat_quality: str  # excellent, good, fair, poor

class ScoreAggregator:
    """Combines multiple heuristic scores into aggregate prediction"""
    
    def __init__(self, heuristics: List, method: str = "additive"):
        """
        Args:
            heuristics: List of BaseHeuristic instances
            method: "additive" or "multiplicative"
        """
        self.heuristics = heuristics
        self.method = method
        self.total_weight = sum(h.weight for h in heuristics)
    
    def calculate_aggregate(self, location: Dict, date: str, 
                           context: Dict) -> AggregateScore:
        """Calculate aggregate score from all heuristics"""
        
        # Calculate each heuristic
        heuristic_results = {}
        for h in self.heuristics:
            result = h.calculate(location, date, context)
            heuristic_results[h.name] = {
                "score": result.score,
                "weight": h.weight,
                "confidence": result.confidence,
                "status": result.status,
                "note": result.note,
                "raw_value": result.raw_value,
                "metadata": result.metadata
            }
        
        # Aggregate scores
        if self.method == "additive":
            total_score = self._additive_aggregation(heuristic_results)
        else:
            total_score = self._multiplicative_aggregation(heuristic_results)
        
        # Calculate factor contributions
        for name, result in heuristic_results.items():
            if self.method == "additive":
                contribution = (result["score"] * result["weight"])
                result["contribution"] = contribution
                result["contribution_pct"] = (contribution / total_score) * 100
            else:
                # For multiplicative, contribution is harder to define
                # Use partial derivative approximation
                result["contribution"] = result["score"]
        
        # Find limiting and best factors
        limiting_factor = min(heuristic_results.items(), 
                            key=lambda x: x[1]["score"])[0]
        best_feature = max(heuristic_results.items(), 
                          key=lambda x: x[1]["score"])[0]
        
        # Overall confidence (weighted average)
        total_confidence = sum(
            r["confidence"] * r["weight"] 
            for r in heuristic_results.values()
        ) / self.total_weight
        
        # Classify habitat quality based on total score
        habitat_quality = self._classify_habitat(total_score)
        
        return AggregateScore(
            total_score=total_score,
            confidence=total_confidence,
            factor_scores=heuristic_results,
            limiting_factor=limiting_factor,
            best_feature=best_feature,
            habitat_quality=habitat_quality
        )
    
    def _additive_aggregation(self, results: Dict) -> float:
        """Sum of (score × weight) for each heuristic"""
        total = sum(
            r["score"] * r["weight"] 
            for r in results.values()
        )
        # Normalize to 0-100 scale
        max_possible = self.total_weight * 10  # Max score per heuristic is 10
        normalized = (total / max_possible) * 100
        return normalized
    
    def _multiplicative_aggregation(self, results: Dict) -> float:
        """Product of normalized scores"""
        # Normalize each score to 0-1
        normalized_scores = [r["score"] / 10.0 for r in results.values()]
        
        # Apply weights as exponents
        weighted_product = 1.0
        for h, score_norm in zip(self.heuristics, normalized_scores):
            weighted_product *= score_norm ** (h.weight / self.total_weight)
        
        # Scale to 0-100
        return weighted_product * 100
    
    def _classify_habitat(self, score: float) -> str:
        """Convert score to quality rating"""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"