"""
Tests for MLPredictionHeuristic - ML-based elk presence prediction.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pickle
import tempfile
import os

from src.scoring.heuristics.ml_prediction import (
    MLPredictionHeuristic,
    find_latest_generalizable_model,
    REQUIRED_FEATURES,
    CONTEXT_TO_FEATURE_MAP
)
from src.scoring.heuristics import HeuristicScore
from src.scoring.aggregator import ScoreAggregator


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


class TestFindLatestGeneralizableModel:
    """Tests for the find_latest_generalizable_model function."""

    def test_find_model_in_directory(self, tmp_path):
        """Test finding model when it exists."""
        # Create mock model files
        model_file = tmp_path / "xgboost_generalizable_20260117_143423.pkl"
        model_file.touch()

        result = find_latest_generalizable_model(tmp_path)
        assert result == model_file

    def test_find_latest_model_by_mtime(self, tmp_path):
        """Test that most recent model is returned."""
        # Create older model
        old_model = tmp_path / "xgboost_generalizable_20260115_100000.pkl"
        old_model.touch()

        # Create newer model
        import time
        time.sleep(0.01)  # Ensure different mtime
        new_model = tmp_path / "xgboost_generalizable_20260117_143423.pkl"
        new_model.touch()

        result = find_latest_generalizable_model(tmp_path)
        assert result == new_model

    def test_excludes_with_coords_models(self, tmp_path):
        """Test that with_coords models are excluded."""
        # Create with_coords model (should be excluded)
        coords_model = tmp_path / "xgboost_generalizable_with_coords_20260117.pkl"
        coords_model.touch()

        # Create generalizable model (should be found)
        gen_model = tmp_path / "xgboost_generalizable_20260115.pkl"
        gen_model.touch()

        result = find_latest_generalizable_model(tmp_path)
        assert result == gen_model

    def test_returns_none_when_no_models(self, tmp_path):
        """Test that None is returned when no models exist."""
        result = find_latest_generalizable_model(tmp_path)
        assert result is None

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        """Test that None is returned for nonexistent directory."""
        result = find_latest_generalizable_model(tmp_path / "nonexistent")
        assert result is None


class TestMLPredictionHeuristicInitialization:
    """Tests for MLPredictionHeuristic initialization."""

    def test_initialization_with_model(self, mock_model_path):
        """Test successful initialization with a model."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        assert heuristic.name == "ml_prediction"
        assert heuristic.weight == 2.0
        assert heuristic.model is not None
        assert heuristic.feature_cols is not None
        assert len(heuristic.feature_cols) > 0

    def test_initialization_custom_weight(self, mock_model_path):
        """Test initialization with custom weight."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path, weight=3.0)
        assert heuristic.weight == 3.0

    def test_initialization_custom_confidence_threshold(self, mock_model_path):
        """Test initialization with custom confidence threshold."""
        heuristic = MLPredictionHeuristic(
            model_path=mock_model_path,
            confidence_threshold=0.8
        )
        assert heuristic.confidence_threshold == 0.8

    def test_initialization_no_model_raises_error(self, tmp_path):
        """Test that FileNotFoundError is raised when no model exists."""
        with pytest.raises(FileNotFoundError):
            MLPredictionHeuristic(model_path=tmp_path / "nonexistent.pkl")

    def test_initialization_invalid_model_path_string(self, tmp_path):
        """Test initialization with string path."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path)

        heuristic = MLPredictionHeuristic(model_path=str(model_path))
        assert heuristic.model is not None

    def test_get_model_info(self, mock_model_path):
        """Test get_model_info method."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)
        info = heuristic.get_model_info()

        assert "model_path" in info
        assert "n_features" in info
        assert "features" in info
        assert "weight" in info
        assert info["weight"] == 2.0


class TestMLPredictionHeuristicCalculate:
    """Tests for MLPredictionHeuristic.calculate method."""

    def test_calculate_high_probability_present(self, tmp_path):
        """Test that high probability results in high score."""
        model_path = tmp_path / "model_high.pkl"
        _create_mock_model(model_path, prob_present=0.8)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = _create_full_context()

        result = heuristic.calculate(location, date, context)

        assert result.score == 8.0  # 0.8 * 10
        assert result.status in ["excellent", "good"]
        assert result.raw_value == 0.8
        assert result.metadata["prediction"] == "Present"

    def test_calculate_low_probability_absent(self, tmp_path):
        """Test that low probability results in low score."""
        model_path = tmp_path / "model_low.pkl"
        _create_mock_model(model_path, prob_present=0.1)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = _create_full_context()

        result = heuristic.calculate(location, date, context)

        assert result.score == 1.0  # 0.1 * 10
        assert result.status == "critical"
        assert result.raw_value == 0.1
        assert result.metadata["prediction"] == "Absent"

    def test_calculate_uncertain_probability(self, tmp_path):
        """Test that probability near 0.5 results in low confidence."""
        model_path = tmp_path / "model_uncertain.pkl"
        _create_mock_model(model_path, prob_present=0.52)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = _create_full_context()

        result = heuristic.calculate(location, date, context)

        assert result.score == pytest.approx(5.2, rel=0.01)
        # Confidence should be low (close to 0) when prob is near 0.5
        assert result.confidence < 0.1  # |0.52 - 0.5| * 2 = 0.04

    def test_calculate_with_temporal_features_from_date(self, mock_model_path):
        """Test that temporal features are extracted from date."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-07-15"  # July 15
        context = _create_full_context()

        result = heuristic.calculate(location, date, context)

        # Should extract month=7 from date
        assert result.metadata["prediction_available"] is True

    def test_calculate_missing_features_under_threshold(self, mock_model_path):
        """Test that missing features under threshold still work."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        # Only provide some features (less than 30% missing)
        context = {
            "elevation": 9000.0,
            "slope_degrees": 15.0,
            "aspect_degrees": 180.0,
            "canopy_cover_percent": 30.0,
            "land_cover_code": 42,
            "land_cover_type": "evergreen_forest",
            "water_distance_miles": 0.5,
            "water_reliability": 0.8,
            "road_distance_miles": 2.0,
            "trail_distance_miles": 1.5,
            "security_habitat_percent": 30.0,
            "snow_depth_inches": 6.0,
            "snow_crust_detected": False,
            "temperature_f": 35.0,
            "precip_last_7_days_inches": 0.2,
            "cloud_cover_percent": 30.0,
            "ndvi": 0.4,
            "ndvi_age_days": 7,
            # Missing: irg, summer_integrated_ndvi, effective_illumination
        }

        result = heuristic.calculate(location, date, context)

        assert result.metadata["prediction_available"] is True
        assert len(result.metadata["missing_features"]) > 0

    def test_calculate_many_missing_features_low_confidence(self, mock_model_path):
        """Test that many missing features result in low confidence."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        # Very sparse context (more than 30% missing)
        context = {
            "elevation": 9000.0,
            "slope_degrees": 15.0,
        }

        result = heuristic.calculate(location, date, context)

        assert result.confidence <= 0.1
        assert result.score == 5.0  # Neutral score
        assert result.metadata["prediction_available"] is False

    def test_calculate_uses_context_key_mapping(self, mock_model_path):
        """Test that context keys are properly mapped to feature names."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        # Use alternative context key names
        context = {
            "elevation": 9000.0,
            "slope": 15.0,  # Maps to slope_degrees
            "aspect": 180.0,  # Maps to aspect_degrees
            "canopy_cover": 30.0,  # Maps to canopy_cover_percent
            "land_cover_code": 42,
            "land_cover_type": "evergreen_forest",
            "water_distance": 0.5,  # Maps to water_distance_miles
            "water_reliability": 0.8,
            "road_distance": 2.0,  # Maps to road_distance_miles
            "trail_distance": 1.5,  # Maps to trail_distance_miles
            "security_habitat": 30.0,  # Maps to security_habitat_percent
            "snow_depth": 6.0,  # Maps to snow_depth_inches
            "snow_crust": False,  # Maps to snow_crust_detected
            "temperature": 35.0,  # Maps to temperature_f
            "precip_7day": 0.2,  # Maps to precip_last_7_days_inches
            "cloud_cover": 30.0,  # Maps to cloud_cover_percent
            "ndvi": 0.4,
            "ndvi_age": 7,  # Maps to ndvi_age_days
            "irg": 0.001,
            "summer_ndvi": 40.0,  # Maps to summer_integrated_ndvi
            "illumination": 0.05,  # Maps to effective_illumination
        }

        result = heuristic.calculate(location, date, context)

        assert result.metadata["prediction_available"] is True


class TestMLPredictionHeuristicConfidence:
    """Tests for confidence calculation."""

    def test_confidence_at_boundaries(self, tmp_path):
        """Test confidence calculation at probability boundaries."""
        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = _create_full_context()

        # Probability = 1.0 -> confidence = 1.0
        model_path = tmp_path / "model_1.pkl"
        _create_mock_model(model_path, prob_present=1.0)
        heuristic = MLPredictionHeuristic(model_path=model_path)
        result = heuristic.calculate(location, date, context)
        assert result.confidence == 1.0

        # Probability = 0.0 -> confidence = 1.0
        model_path = tmp_path / "model_0.pkl"
        _create_mock_model(model_path, prob_present=0.0)
        heuristic = MLPredictionHeuristic(model_path=model_path)
        result = heuristic.calculate(location, date, context)
        assert result.confidence == 1.0

        # Probability = 0.5 -> confidence = 0.0
        model_path = tmp_path / "model_05.pkl"
        _create_mock_model(model_path, prob_present=0.5)
        heuristic = MLPredictionHeuristic(model_path=model_path)
        result = heuristic.calculate(location, date, context)
        assert result.confidence == 0.0

    def test_confidence_reduced_with_missing_features(self, tmp_path):
        """Test that confidence is reduced when features are missing."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path, prob_present=0.8)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"

        # Full context - high confidence
        full_context = _create_full_context()
        full_result = heuristic.calculate(location, date, full_context)

        # Partial context - reduced confidence
        partial_context = _create_full_context()
        del partial_context["irg"]
        del partial_context["summer_integrated_ndvi"]

        partial_result = heuristic.calculate(location, date, partial_context)

        assert partial_result.confidence < full_result.confidence


class TestMLPredictionHeuristicStatus:
    """Tests for status determination."""

    def test_status_excellent_high_probability(self, tmp_path):
        """Test excellent status for high probability."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path, prob_present=0.9)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.status == "excellent"

    def test_status_good_moderate_probability(self, tmp_path):
        """Test good status for moderate probability."""
        model_path = tmp_path / "model.pkl"
        # Use 0.65 to ensure confidence is above 0.2 threshold (confidence = |0.65-0.5|*2 = 0.3)
        _create_mock_model(model_path, prob_present=0.65)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.status == "good"

    def test_status_fair_low_confidence(self, tmp_path):
        """Test fair status for low confidence predictions."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path, prob_present=0.51)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.status == "fair"

    def test_status_critical_very_low_probability(self, tmp_path):
        """Test critical status for very low probability."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path, prob_present=0.05)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.status == "critical"


class TestMLPredictionHeuristicMetadata:
    """Tests for metadata in HeuristicScore."""

    def test_metadata_contains_probabilities(self, tmp_path):
        """Test that metadata contains probability values."""
        model_path = tmp_path / "model.pkl"
        _create_mock_model(model_path, prob_present=0.7)
        heuristic = MLPredictionHeuristic(model_path=model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )

        assert "probability_present" in result.metadata
        assert "probability_absent" in result.metadata
        assert result.metadata["probability_present"] == 0.7
        assert result.metadata["probability_absent"] == pytest.approx(0.3, rel=0.01)

    def test_metadata_contains_prediction_label(self, tmp_path):
        """Test that metadata contains prediction label."""
        # Present prediction
        model_path = tmp_path / "model_present.pkl"
        _create_mock_model(model_path, prob_present=0.7)
        heuristic = MLPredictionHeuristic(model_path=model_path)
        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.metadata["prediction"] == "Present"

        # Absent prediction
        model_path = tmp_path / "model_absent.pkl"
        _create_mock_model(model_path, prob_present=0.3)
        heuristic = MLPredictionHeuristic(model_path=model_path)
        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )
        assert result.metadata["prediction"] == "Absent"

    def test_metadata_contains_model_info(self, mock_model_path):
        """Test that metadata contains model information."""
        heuristic = MLPredictionHeuristic(model_path=mock_model_path)

        result = heuristic.calculate(
            {"lat": 43.0, "lon": -110.0},
            "2026-10-15",
            _create_full_context()
        )

        assert "model_path" in result.metadata
        assert "prediction_available" in result.metadata


class TestMLPredictionHeuristicIntegration:
    """Integration tests with ScoreAggregator."""

    def test_integration_with_aggregator(self, mock_model_path):
        """Test that ML heuristic works with ScoreAggregator."""
        from src.scoring.heuristics import ElevationHeuristic, WaterDistanceHeuristic

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
            **_create_full_context()
        }

        result = aggregator.calculate_aggregate(location, date, context)

        assert result.total_score >= 0
        assert result.total_score <= 100
        assert "ml_prediction" in result.factor_scores

    def test_ml_can_be_limiting_factor(self, tmp_path):
        """Test that ML heuristic can be identified as limiting factor."""
        from src.scoring.heuristics import ElevationHeuristic

        # Create model with very low probability
        model_path = tmp_path / "model_low.pkl"
        _create_mock_model(model_path, prob_present=0.05)
        ml_heuristic = MLPredictionHeuristic(model_path=model_path, weight=1.0)

        heuristics = [
            ElevationHeuristic(weight=1.0),
            ml_heuristic
        ]

        aggregator = ScoreAggregator(heuristics, method="additive")

        location = {"lat": 43.0, "lon": -110.0}
        date = "2026-10-15"
        context = {
            "elevation": 9000.0,  # Optimal elevation = high score
            **_create_full_context()
        }

        result = aggregator.calculate_aggregate(location, date, context)

        # ML should be limiting factor since it has very low score
        assert result.limiting_factor == "ml_prediction"


# Fixtures and helpers

@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model file for testing."""
    model_path = tmp_path / "xgboost_generalizable_test.pkl"
    _create_mock_model(model_path)
    return model_path


def _create_mock_model(model_path: Path, prob_present: float = 0.6):
    """Create a mock XGBoost model file."""
    model = SimpleModel(prob_present=prob_present)

    # Create simple label encoder for land_cover_type
    label_encoder = SimpleLabelEncoder(['evergreen_forest', 'shrub', 'grassland'])

    data = {
        'model': model,
        'feature_cols': REQUIRED_FEATURES,
        'label_encoders': {'land_cover_type': label_encoder},
        'cv_results': {'mean_accuracy': 0.717}
    }

    with open(model_path, 'wb') as f:
        pickle.dump(data, f)


def _create_full_context():
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
