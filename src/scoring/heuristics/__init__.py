# Import base classes from base module
from .base import BaseHeuristic, HeuristicScore

# Import all heuristic classes
from .elevation import ElevationHeuristic
from .snow import SnowConditionsHeuristic
from .water import WaterDistanceHeuristic
from .vegetation import VegetationQualityHeuristic
from .access import HuntingPressureHeuristic
from .security import SecurityHabitatHeuristic
from .predation import PredationRiskHeuristic
from .nutrition import NutritionalConditionHeuristic
from .winterkill import WinterSeverityHeuristic

__all__ = [
    "BaseHeuristic",
    "HeuristicScore",
    "ElevationHeuristic",
    "SnowConditionsHeuristic",
    "WaterDistanceHeuristic",
    "VegetationQualityHeuristic",
    "HuntingPressureHeuristic",
    "SecurityHabitatHeuristic",
    "PredationRiskHeuristic",
    "NutritionalConditionHeuristic",
    "WinterSeverityHeuristic",
]    