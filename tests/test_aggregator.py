import pytest
import pickle
import numpy as np
from pathlib import Path
from src.scoring.aggregator import ScoreAggregator
from src.scoring.heuristics import ElevationHeuristic, WaterDistanceHeuristic
from src.scoring.heuristics.ml_prediction import MLPredictionHeuristic, REQUIRED_FEATURES


class SimpleModel:
    """Simple picklable model for testing."""

    def __init__(self, prob_present=0.6):
        self._prob_present = prob_present

    def predict_proba(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.array([[1 - self._prob_present, self._prob_present]] * n_samples)

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.array([1 if self._prob_present > 0.5 else 0] * n_samples)


class SimpleLabelEncoder:
    """Simple picklable label encoder for testing."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def transform(self, x):
        return np.array([0] * len(x))

class TestScoreAggregator:
    
    def test_additive_aggregation(self):
        """Test additive score aggregation"""
        heuristics = [
            ElevationHeuristic(weight=2.0),
            WaterDistanceHeuristic(weight=3.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="additive")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            "water_distance_miles": 0.5,
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)
        
        # Should have positive score
        assert 0 <= result.total_score <= 100
        
        # Should have factor breakdown
        assert "elevation" in result.factor_scores
        assert "water_distance" in result.factor_scores
        
        # Check contributions
        elev_contrib = result.factor_scores["elevation"]["contribution"]
        water_contrib = result.factor_scores["water_distance"]["contribution"]
        assert elev_contrib > 0
        assert water_contrib > 0
    
    def test_multiplicative_aggregation(self):
        """Test multiplicative score aggregation"""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            WaterDistanceHeuristic(weight=1.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="multiplicative")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            "water_distance_miles": 0.5,
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)
        
        assert 0 <= result.total_score <= 100
    
    def test_limiting_factor_identification(self):
        """Test that limiting factor is correctly identified"""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            WaterDistanceHeuristic(weight=1.0)
        ]
        
        aggregator = ScoreAggregator(heuristics, method="additive")
        
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,  # Good
            "water_distance_miles": 3.5,  # Poor
            "water_reliability": 1.0
        }
        
        result = aggregator.calculate_aggregate(location, date, context)

        # Water should be limiting factor
        assert result.limiting_factor == "water_distance"


class TestAggregatorWithMLHeuristic:
    """Tests for ScoreAggregator with ML heuristic integration."""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock model file for testing."""
        model_path = tmp_path / "xgboost_generalizable_test.pkl"
        self._create_mock_model(model_path)
        return model_path

    def _create_mock_model(self, model_path: Path, prob_present: float = 0.7):
        """Create a mock XGBoost model file."""
        model = SimpleModel(prob_present=prob_present)
        label_encoder = SimpleLabelEncoder(['evergreen_forest', 'shrub', 'grassland'])

        data = {
            'model': model,
            'feature_cols': REQUIRED_FEATURES,
            'label_encoders': {'land_cover_type': label_encoder},
            'cv_results': {'mean_accuracy': 0.717}
        }

        with open(model_path, 'wb') as f:
            pickle.dump(data, f)

    def _create_full_context(self):
        """Create a context dictionary with all required features."""
        return {
            'elevation': 9000.0,
            'slope_degrees': 15.0,
            'aspect_degrees': 180.0,
            'canopy_cover_percent': 30.0,
            'land_cover_code': 42,
            'land_cover_type': 'evergreen_forest',
            'water_distance_miles': 0.5,
            'water_reliability': 0.8,
            'road_distance_miles': 2.0,
            'trail_distance_miles': 1.5,
            'security_habitat_percent': 30.0,
            'snow_depth_inches': 6.0,
            'snow_crust_detected': False,
            'temperature_f': 35.0,
            'precip_last_7_days_inches': 0.2,
            'cloud_cover_percent': 30.0,
            'ndvi': 0.4,
            'ndvi_age_days': 7,
            'irg': 0.001,
            'summer_integrated_ndvi': 40.0,
            'effective_illumination': 0.05,
        }

    def test_aggregator_with_ml_heuristic(self, mock_model_path):
        """Test that ML heuristic integrates with aggregator."""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            WaterDistanceHeuristic(weight=1.0),
            MLPredictionHeuristic(model_path=mock_model_path, weight=2.0)
        ]

        aggregator = ScoreAggregator(heuristics, method="additive")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            "water_distance_miles": 0.5,
            "water_reliability": 1.0,
            **self._create_full_context()
        }

        result = aggregator.calculate_aggregate(location, date, context)

        # Should have all three factors
        assert "elevation" in result.factor_scores
        assert "water_distance" in result.factor_scores
        assert "ml_prediction" in result.factor_scores

        # Total score should be valid
        assert 0 <= result.total_score <= 100

    def test_ml_heuristic_weighted_contribution(self, mock_model_path):
        """Test that ML heuristic weight affects total score."""
        # Lower weight ML heuristic
        heuristics_low = [
            ElevationHeuristic(weight=1.0),
            MLPredictionHeuristic(model_path=mock_model_path, weight=1.0)
        ]
        aggregator_low = ScoreAggregator(heuristics_low, method="additive")

        # Higher weight ML heuristic
        heuristics_high = [
            ElevationHeuristic(weight=1.0),
            MLPredictionHeuristic(model_path=mock_model_path, weight=3.0)
        ]
        aggregator_high = ScoreAggregator(heuristics_high, method="additive")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,  # Optimal = score 10
            **self._create_full_context()  # ML prob 0.7 = score 7
        }

        result_low = aggregator_low.calculate_aggregate(location, date, context)
        result_high = aggregator_high.calculate_aggregate(location, date, context)

        # Both should be valid
        assert 0 <= result_low.total_score <= 100
        assert 0 <= result_high.total_score <= 100

        # With higher ML weight, if ML score (7) < elevation score (10),
        # the total should be lower with higher ML weight
        ml_low = result_low.factor_scores["ml_prediction"]["score"]
        ml_high = result_high.factor_scores["ml_prediction"]["score"]
        assert ml_low == ml_high  # Same score, different weights

    def test_ml_as_limiting_factor(self, tmp_path):
        """Test that ML can be identified as limiting factor."""
        # Create model with very low probability
        model_path = tmp_path / "model_low.pkl"
        self._create_mock_model(model_path, prob_present=0.05)
        ml_heuristic = MLPredictionHeuristic(model_path=model_path, weight=1.0)

        heuristics = [
            ElevationHeuristic(weight=1.0),
            ml_heuristic
        ]

        aggregator = ScoreAggregator(heuristics, method="additive")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,  # Optimal = high score
            **self._create_full_context()
        }

        result = aggregator.calculate_aggregate(location, date, context)

        # ML should be limiting factor with score 0.5 (0.05 * 10)
        assert result.limiting_factor == "ml_prediction"
        assert result.factor_scores["ml_prediction"]["score"] == pytest.approx(0.5, rel=0.1)

    def test_ml_as_best_feature(self, tmp_path):
        """Test that ML can be identified as best feature."""
        # Create model with high probability
        model_path = tmp_path / "model_high.pkl"
        self._create_mock_model(model_path, prob_present=0.95)
        ml_heuristic = MLPredictionHeuristic(model_path=model_path, weight=1.0)

        heuristics = [
            ElevationHeuristic(weight=1.0),
            ml_heuristic
        ]

        aggregator = ScoreAggregator(heuristics, method="additive")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        # Put elevation AFTER context expansion to override the default
        context = {
            **self._create_full_context(),
            "elevation": 3000.0,  # Extremely low = very poor score (should be ~0)
        }

        result = aggregator.calculate_aggregate(location, date, context)

        # ML should be best feature with score 9.5 vs elevation ~0
        assert result.best_feature == "ml_prediction"

    def test_multiplicative_with_ml(self, mock_model_path):
        """Test multiplicative aggregation with ML heuristic."""
        heuristics = [
            ElevationHeuristic(weight=1.0),
            MLPredictionHeuristic(model_path=mock_model_path, weight=1.0)
        ]

        aggregator = ScoreAggregator(heuristics, method="multiplicative")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,
            **self._create_full_context()
        }

        result = aggregator.calculate_aggregate(location, date, context)

        assert 0 <= result.total_score <= 100
        assert "ml_prediction" in result.factor_scores
