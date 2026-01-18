"""
ML-based elk presence prediction heuristic using XGBoost model.

This heuristic wraps the trained XGBoost model to provide data-driven
predictions that integrate with the existing heuristics framework.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
import pickle
import numpy as np
import pandas as pd

from .base import BaseHeuristic, HeuristicScore


# Required features for the generalizable model (no lat/lon/year)
REQUIRED_FEATURES = [
    'elevation',
    'slope_degrees',
    'aspect_degrees',
    'canopy_cover_percent',
    'land_cover_code',
    'land_cover_type',
    'water_distance_miles',
    'water_reliability',
    'road_distance_miles',
    'trail_distance_miles',
    'security_habitat_percent',
    'snow_depth_inches',
    'snow_crust_detected',
    'temperature_f',
    'precip_last_7_days_inches',
    'cloud_cover_percent',
    'ndvi',
    'ndvi_age_days',
    'irg',
    'summer_integrated_ndvi',
    'effective_illumination',
    'month',
    'day_of_year_sin',
    'day_of_year_cos',
]

# Mapping from context keys to model feature names
CONTEXT_TO_FEATURE_MAP = {
    'elevation': 'elevation',
    'slope': 'slope_degrees',
    'slope_degrees': 'slope_degrees',
    'aspect': 'aspect_degrees',
    'aspect_degrees': 'aspect_degrees',
    'canopy_cover': 'canopy_cover_percent',
    'canopy_cover_percent': 'canopy_cover_percent',
    'land_cover_code': 'land_cover_code',
    'land_cover_type': 'land_cover_type',
    'water_distance_miles': 'water_distance_miles',
    'water_distance': 'water_distance_miles',
    'water_reliability': 'water_reliability',
    'road_distance_miles': 'road_distance_miles',
    'road_distance': 'road_distance_miles',
    'trail_distance_miles': 'trail_distance_miles',
    'trail_distance': 'trail_distance_miles',
    'security_habitat_percent': 'security_habitat_percent',
    'security_habitat': 'security_habitat_percent',
    'snow_depth_inches': 'snow_depth_inches',
    'snow_depth': 'snow_depth_inches',
    'snow_crust_detected': 'snow_crust_detected',
    'snow_crust': 'snow_crust_detected',
    'temperature_f': 'temperature_f',
    'temperature': 'temperature_f',
    'precip_last_7_days_inches': 'precip_last_7_days_inches',
    'precip_7day': 'precip_last_7_days_inches',
    'precipitation': 'precip_last_7_days_inches',
    'cloud_cover_percent': 'cloud_cover_percent',
    'cloud_cover': 'cloud_cover_percent',
    'ndvi': 'ndvi',
    'ndvi_age_days': 'ndvi_age_days',
    'ndvi_age': 'ndvi_age_days',
    'irg': 'irg',
    'summer_integrated_ndvi': 'summer_integrated_ndvi',
    'summer_ndvi': 'summer_integrated_ndvi',
    'effective_illumination': 'effective_illumination',
    'illumination': 'effective_illumination',
}


def find_latest_generalizable_model(models_dir: Path = None) -> Optional[Path]:
    """
    Find the most recent generalizable model file.

    Args:
        models_dir: Directory containing model files. Defaults to 'models/'.

    Returns:
        Path to the most recent generalizable model, or None if not found.
    """
    if models_dir is None:
        models_dir = Path('models')

    if not models_dir.exists():
        return None

    # Find generalizable models (excluding with_coords versions)
    pkl_files = list(models_dir.glob('xgboost_generalizable_*.pkl'))
    pkl_files = [f for f in pkl_files if 'with_coords' not in f.name]

    if not pkl_files:
        return None

    # Sort by modification time, most recent first
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pkl_files[0]


class MLPredictionHeuristic(BaseHeuristic):
    """
    ML-based elk presence prediction heuristic using XGBoost model.

    This heuristic wraps the trained XGBoost generalizable model to provide
    data-driven predictions. It integrates with the existing heuristics
    framework and provides:

    - Score: probability_present * 10 (maps 0-1 probability to 0-10 scale)
    - Confidence: Based on how far the probability is from 0.5 (uncertainty)
    - Graceful degradation: Returns low confidence when features are missing

    Attributes:
        model: The loaded XGBoost model
        feature_cols: List of feature column names the model expects
        label_encoders: Dict of label encoders for categorical features
        confidence_threshold: Minimum probability distance from 0.5 to trust
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        weight: float = 2.0,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the ML prediction heuristic.

        Args:
            model_path: Path to the trained model .pkl file. If None, will
                        attempt to find the latest generalizable model.
            weight: Weight for this heuristic in score aggregation. Default 2.0
                    gives ML predictions more influence than individual heuristics.
            confidence_threshold: Minimum confidence (probability distance from 0.5)
                                  to consider the prediction reliable. Default 0.6.

        Raises:
            FileNotFoundError: If no model file is found.
            ValueError: If the model file is invalid.
        """
        super().__init__("ml_prediction", weight)

        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_cols = None
        self.label_encoders = None
        self.model_path = None
        self.cv_accuracy = None

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[Path] = None) -> None:
        """Load the XGBoost model from disk."""
        if model_path is None:
            model_path = find_latest_generalizable_model()

        if model_path is None:
            raise FileNotFoundError(
                "No generalizable model found. Train a model first with: "
                "python scripts/train_generalizable_model.py"
            )

        if isinstance(model_path, str):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.label_encoders = data.get('label_encoders', {})
        self.model_path = model_path

        # Store CV accuracy if available
        if 'cv_results' in data:
            self.cv_accuracy = data['cv_results'].get('mean_accuracy')

    def calculate(self, location: Dict, date: str, context: Dict) -> HeuristicScore:
        """
        Calculate ML prediction score for a location and date.

        Args:
            location: {"lat": float, "lon": float} - Used for logging only,
                      not for prediction (generalizable model excludes coords).
            date: ISO format date string (e.g., "2026-10-15").
            context: Dictionary containing environmental features. Keys can use
                     either context names or model feature names.

        Returns:
            HeuristicScore with:
            - score: probability_present * 10 (0-10 scale)
            - confidence: How certain the model is (0-1 scale)
            - status: "excellent", "good", "fair", "poor", or "critical"
            - note: Human-readable explanation
            - raw_value: The raw probability of elk presence
            - metadata: Model details, feature values, probability breakdown
        """
        # Extract features from context
        features, missing_features = self._extract_features(context, date)

        # Handle missing features gracefully
        if missing_features:
            missing_pct = len(missing_features) / len(self.feature_cols)

            # If too many features missing, return low confidence score
            if missing_pct > 0.3:  # More than 30% missing
                return HeuristicScore(
                    score=5.0,  # Neutral score
                    confidence=0.1,  # Very low confidence
                    status="fair",
                    note=f"Insufficient data for ML prediction ({len(missing_features)} features missing)",
                    raw_value=0.5,
                    metadata={
                        "missing_features": missing_features,
                        "missing_percent": missing_pct * 100,
                        "prediction_available": False
                    }
                )

            # Fill missing features with defaults
            features = self._fill_missing_features(features, missing_features)

        # Prepare features for model
        try:
            X = self._prepare_features(features)
        except Exception as e:
            return HeuristicScore(
                score=5.0,
                confidence=0.1,
                status="fair",
                note=f"Error preparing features for ML: {str(e)}",
                raw_value=0.5,
                metadata={
                    "error": str(e),
                    "prediction_available": False
                }
            )

        # Make prediction
        probabilities = self.model.predict_proba(X)[0]
        prob_absent = probabilities[0]
        prob_present = probabilities[1]

        # Calculate score (0-10 scale)
        score = prob_present * 10.0

        # Calculate confidence (distance from 0.5, scaled to 0-1)
        confidence = abs(prob_present - 0.5) * 2.0

        # Reduce confidence if features were missing
        if missing_features:
            confidence *= (1.0 - len(missing_features) / len(self.feature_cols))

        # Determine status
        status = self._get_prediction_status(prob_present, confidence)

        # Generate note
        note = self._generate_note(prob_present, confidence, missing_features)

        return HeuristicScore(
            score=score,
            confidence=confidence,
            status=status,
            note=note,
            raw_value=float(prob_present),  # Convert numpy to native Python
            metadata={
                "probability_present": float(prob_present),
                "probability_absent": float(prob_absent),
                "prediction": "Present" if prob_present > 0.5 else "Absent",
                "model_confidence": float(confidence),
                "missing_features": missing_features or [],
                "prediction_available": True,
                "model_path": str(self.model_path) if self.model_path else None,
                "cv_accuracy": float(self.cv_accuracy) if self.cv_accuracy is not None else None,
            }
        )

    def _extract_features(self, context: Dict, date: str) -> tuple:
        """
        Extract model features from context dictionary.

        Returns:
            Tuple of (features_dict, missing_features_list)
        """
        features = {}
        missing = []

        # Parse date for temporal features
        dt = datetime.fromisoformat(date)

        # Add temporal features
        day_of_year = dt.timetuple().tm_yday
        features['month'] = dt.month
        features['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        features['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Extract other features from context
        for feature in self.feature_cols:
            if feature in ['month', 'day_of_year_sin', 'day_of_year_cos']:
                continue  # Already handled

            # Try to find feature in context using various key names
            value = None

            # Direct match
            if feature in context:
                value = context[feature]
            else:
                # Try mapped names
                for ctx_key, feat_name in CONTEXT_TO_FEATURE_MAP.items():
                    if feat_name == feature and ctx_key in context:
                        value = context[ctx_key]
                        break

            if value is not None:
                features[feature] = value
            else:
                missing.append(feature)

        return features, missing

    def _fill_missing_features(self, features: Dict, missing: List[str]) -> Dict:
        """Fill missing features with reasonable defaults."""
        defaults = {
            'elevation': 8500.0,  # Mid-range elk elevation
            'slope_degrees': 15.0,
            'aspect_degrees': 180.0,  # South-facing
            'canopy_cover_percent': 30.0,
            'land_cover_code': 42,  # Evergreen forest
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

        for feature in missing:
            if feature in defaults:
                features[feature] = defaults[feature]

        return features

    def _prepare_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features for model prediction."""
        df = pd.DataFrame([features])

        # Apply label encoders to categorical columns
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].fillna('unknown').astype(str)
                known_classes = set(le.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Ensure all required columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Select and order features
        return df[self.feature_cols]

    def _get_prediction_status(self, prob_present: float, confidence: float) -> str:
        """Determine status based on probability and confidence."""
        if confidence < 0.2:
            return "fair"  # Low confidence = uncertain

        if prob_present >= 0.7:
            return "excellent"
        elif prob_present >= 0.5:
            return "good"
        elif prob_present >= 0.3:
            return "fair"
        elif prob_present >= 0.15:
            return "poor"
        else:
            return "critical"

    def _generate_note(
        self,
        prob_present: float,
        confidence: float,
        missing_features: Optional[List[str]]
    ) -> str:
        """Generate human-readable note for the prediction."""
        if prob_present >= 0.7:
            base_note = f"ML model predicts high elk presence likelihood ({prob_present:.0%})"
        elif prob_present >= 0.5:
            base_note = f"ML model predicts moderate elk presence likelihood ({prob_present:.0%})"
        elif prob_present >= 0.3:
            base_note = f"ML model predicts low elk presence likelihood ({prob_present:.0%})"
        else:
            base_note = f"ML model predicts very low elk presence likelihood ({prob_present:.0%})"

        # Add confidence qualifier
        if confidence < 0.3:
            base_note += " (low confidence - uncertain prediction)"
        elif confidence < 0.6:
            base_note += " (moderate confidence)"

        # Mention missing features if any
        if missing_features:
            base_note += f" - {len(missing_features)} features estimated"

        return base_note

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "n_features": len(self.feature_cols) if self.feature_cols else 0,
            "features": self.feature_cols,
            "cv_accuracy": self.cv_accuracy,
            "weight": self.weight,
            "confidence_threshold": self.confidence_threshold,
        }
