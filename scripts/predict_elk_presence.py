#!/usr/bin/env python3
"""
Predict elk presence for new locations using trained XGBoost model.

This script provides multiple ways to make predictions:
1. Single location with pre-computed features
2. Batch predictions from CSV file with features
3. Interactive mode for quick predictions

Usage:
    # Predict from CSV file with features
    python scripts/predict_elk_presence.py --input locations.csv --output predictions.csv

    # Single prediction with pre-computed features (JSON)
    python scripts/predict_elk_presence.py --features '{"latitude": 43.5, "longitude": -109.5, ...}'

    # List required features
    python scripts/predict_elk_presence.py --list-features

    # Use specific model
    python scripts/predict_elk_presence.py --model models/xgboost_elk_presence_20260117_141935.pkl --input data.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
from typing import Optional, Dict, List, Union

# Required features for generalizable model (no lat/lon/year for better cross-region performance)
REQUIRED_FEATURES_GENERALIZABLE = [
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

# Required features for full model (includes location features)
REQUIRED_FEATURES_FULL = [
    'latitude',
    'longitude',
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
    'year',
    'month',
    'day_of_year_sin',
    'day_of_year_cos',
]

# Default to generalizable model features
REQUIRED_FEATURES = REQUIRED_FEATURES_GENERALIZABLE

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    'latitude': 'Latitude in decimal degrees (WGS84)',
    'longitude': 'Longitude in decimal degrees (WGS84, negative for western hemisphere)',
    'elevation': 'Elevation in meters',
    'slope_degrees': 'Terrain slope in degrees (0-90)',
    'aspect_degrees': 'Terrain aspect in degrees (0-360, 0=North)',
    'canopy_cover_percent': 'Tree canopy cover percentage (0-100)',
    'land_cover_code': 'NLCD land cover code',
    'land_cover_type': 'NLCD land cover type name',
    'water_distance_miles': 'Distance to nearest water source in miles',
    'water_reliability': 'Water source reliability score (0.7-1.0)',
    'road_distance_miles': 'Distance to nearest road in miles',
    'trail_distance_miles': 'Distance to nearest trail in miles',
    'security_habitat_percent': 'Security habitat percentage (log-transformed)',
    'snow_depth_inches': 'Snow depth in inches',
    'snow_crust_detected': 'Whether snow crust is detected (True/False)',
    'temperature_f': 'Temperature in Fahrenheit',
    'precip_last_7_days_inches': 'Precipitation in last 7 days (inches)',
    'cloud_cover_percent': 'Cloud cover percentage (0-100)',
    'ndvi': 'Normalized Difference Vegetation Index (-1 to 1)',
    'ndvi_age_days': 'Age of NDVI observation in days',
    'irg': 'Instantaneous Rate of Green-up',
    'summer_integrated_ndvi': 'Summer integrated NDVI value',
    'effective_illumination': 'Effective lunar illumination',
    'year': 'Year of observation',
    'month': 'Month of observation (1-12)',
    'day_of_year_sin': 'Sine of day of year (cyclical encoding)',
    'day_of_year_cos': 'Cosine of day of year (cyclical encoding)',
}


def find_latest_model(models_dir: Path = Path('models'), prefer_generalizable: bool = True) -> Optional[Path]:
    """
    Find the most recent model file in the models directory.

    Args:
        models_dir: Directory containing model files
        prefer_generalizable: If True, prefer generalizable models over full models

    Returns:
        Path to the most recent model file, or None if not found
    """
    if prefer_generalizable:
        # First try to find generalizable models
        pkl_files = list(models_dir.glob('xgboost_generalizable_*.pkl'))
        # Exclude "with_coords" versions as they don't generalize as well
        pkl_files = [f for f in pkl_files if 'with_coords' not in f.name]

        if pkl_files:
            pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return pkl_files[0]

    # Fall back to any model
    pkl_files = list(models_dir.glob('xgboost_*.pkl'))
    if not pkl_files:
        return None

    # Sort by modification time, most recent first
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pkl_files[0]


def load_model(model_path: Path) -> tuple:
    """
    Load trained model and associated metadata.

    Returns:
        Tuple of (model, feature_cols, label_encoders)
    """
    print(f"Loading model from {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    feature_cols = data['feature_cols']
    label_encoders = data.get('label_encoders', {})

    # Detect model type
    is_generalizable = 'generalizable' in model_path.name
    has_coords = 'latitude' in feature_cols or 'longitude' in feature_cols

    print(f"  Model loaded successfully")
    print(f"  Type: {'Generalizable' if is_generalizable else 'Full'} model")
    if is_generalizable:
        print(f"  Note: Optimized for cross-region prediction (no lat/lon)")
    print(f"  Features: {len(feature_cols)}")

    # Show CV performance if available
    if 'cv_results' in data:
        cv = data['cv_results']
        print(f"  Spatial CV accuracy: {cv.get('mean_accuracy', 0):.1%}")

    return model, feature_cols, label_encoders


def encode_day_of_year(month: int, day: int = 15) -> tuple:
    """
    Convert month/day to cyclical day_of_year encoding.

    Args:
        month: Month (1-12)
        day: Day of month (default: 15, middle of month)

    Returns:
        Tuple of (day_of_year_sin, day_of_year_cos)
    """
    # Approximate day of year from month and day
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[:month-1]) + day

    # Cyclical encoding
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    return day_of_year_sin, day_of_year_cos


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list,
    label_encoders: dict
) -> pd.DataFrame:
    """
    Prepare features for prediction, applying necessary encodings.
    """
    df = df.copy()

    # Add cyclical day encoding if month is provided but sin/cos are not
    if 'month' in df.columns:
        if 'day_of_year_sin' not in df.columns or 'day_of_year_cos' not in df.columns:
            day = df.get('day', 15)
            if isinstance(day, pd.Series):
                day = day.iloc[0]

            for idx in df.index:
                month_val = df.loc[idx, 'month']
                sin_val, cos_val = encode_day_of_year(int(month_val))
                df.loc[idx, 'day_of_year_sin'] = sin_val
                df.loc[idx, 'day_of_year_cos'] = cos_val

    # Apply label encoders to categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen categories
            df[col] = df[col].fillna('unknown').astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Ensure all required columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    # Select and order features
    return df[feature_cols]


def predict(
    model,
    features_df: pd.DataFrame,
    feature_cols: list,
    label_encoders: dict,
    return_proba: bool = True
) -> pd.DataFrame:
    """
    Make predictions for given features.

    Args:
        model: Trained XGBoost model
        features_df: DataFrame with feature values
        feature_cols: List of feature column names
        label_encoders: Dictionary of label encoders for categorical features
        return_proba: Whether to return probability scores

    Returns:
        DataFrame with predictions and optional probabilities
    """
    # Prepare features
    X = prepare_features(features_df, feature_cols, label_encoders)

    # Make predictions
    predictions = model.predict(X)

    results = pd.DataFrame({
        'elk_present_predicted': predictions,
        'prediction_label': ['Present' if p == 1 else 'Absent' for p in predictions]
    })

    if return_proba:
        probabilities = model.predict_proba(X)
        results['probability_absent'] = probabilities[:, 0]
        results['probability_present'] = probabilities[:, 1]
        results['confidence'] = np.maximum(probabilities[:, 0], probabilities[:, 1])

    return results


def predict_single(
    model,
    features: Dict,
    feature_cols: list,
    label_encoders: dict
) -> Dict:
    """
    Make prediction for a single location.

    Args:
        model: Trained XGBoost model
        features: Dictionary of feature values
        feature_cols: List of feature column names
        label_encoders: Dictionary of label encoders

    Returns:
        Dictionary with prediction results
    """
    df = pd.DataFrame([features])
    results = predict(model, df, feature_cols, label_encoders)

    return {
        'prediction': int(results['elk_present_predicted'].iloc[0]),
        'label': results['prediction_label'].iloc[0],
        'probability_present': float(results['probability_present'].iloc[0]),
        'probability_absent': float(results['probability_absent'].iloc[0]),
        'confidence': float(results['confidence'].iloc[0])
    }


def predict_batch(
    model,
    input_file: Path,
    output_file: Path,
    feature_cols: list,
    label_encoders: dict
) -> pd.DataFrame:
    """
    Make predictions for batch of locations from CSV file.
    """
    print(f"\nLoading input data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows")

    # Check for required features
    missing = set(feature_cols) - set(df.columns)

    # Allow for day_of_year_sin/cos to be computed from month
    if 'day_of_year_sin' in missing and 'day_of_year_cos' in missing and 'month' in df.columns:
        missing.discard('day_of_year_sin')
        missing.discard('day_of_year_cos')

    if missing:
        print(f"\n  ERROR: Missing required features:")
        for col in sorted(missing):
            print(f"    - {col}: {FEATURE_DESCRIPTIONS.get(col, 'No description')}")
        return None

    print(f"\nMaking predictions...")
    results = predict(model, df, feature_cols, label_encoders)

    # Combine original data with predictions
    output_df = pd.concat([df, results], axis=1)

    # Save results
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Summary statistics
    n_present = (results['elk_present_predicted'] == 1).sum()
    n_absent = (results['elk_present_predicted'] == 0).sum()
    avg_confidence = results['confidence'].mean()

    print(f"\nPrediction Summary:")
    print(f"  Total locations: {len(results):,}")
    print(f"  Predicted Present: {n_present:,} ({n_present/len(results)*100:.1f}%)")
    print(f"  Predicted Absent: {n_absent:,} ({n_absent/len(results)*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.1%}")

    return output_df


def list_features(show_full: bool = False):
    """Print required features and their descriptions."""
    features = REQUIRED_FEATURES_FULL if show_full else REQUIRED_FEATURES_GENERALIZABLE
    model_type = "Full Model" if show_full else "Generalizable Model (Recommended)"

    print(f"\nRequired Features for Prediction - {model_type}")
    print("=" * 70)
    print(f"\nTotal features required: {len(features)}")

    if not show_full:
        print("\nNote: Generalizable model excludes lat/lon/year for better")
        print("      cross-region prediction (71.7% accuracy on new regions)")

    print("\nFeature details:")
    print("-" * 70)

    for feature in features:
        desc = FEATURE_DESCRIPTIONS.get(feature, 'No description available')
        print(f"\n  {feature}")
        print(f"    {desc}")

    print("\n" + "=" * 70)
    print("\nNotes:")
    print("  - day_of_year_sin and day_of_year_cos can be auto-computed from 'month'")
    print("  - land_cover_type should be one of: shrub, evergreen_forest, grassland,")
    print("    deciduous_forest, mixed_forest, emergent_wetland, hay_pasture,")
    print("    woody_wetland, cultivated_crops, barren_land")
    print("  - snow_crust_detected should be True or False")

    if not show_full:
        print("\nTo see full model features (with lat/lon/year), use --list-features-full")


def create_sample_input(include_coords: bool = False):
    """
    Create a sample input file for reference.

    Args:
        include_coords: If True, include lat/lon/year for full model
    """
    # Base features for generalizable model
    sample_data = {
        'elevation': [2200, 2500, 2100],
        'slope_degrees': [10, 15, 5],
        'aspect_degrees': [180, 90, 270],
        'canopy_cover_percent': [20, 40, 10],
        'land_cover_code': [52, 42, 71],
        'land_cover_type': ['shrub', 'evergreen_forest', 'grassland'],
        'water_distance_miles': [0.5, 1.2, 0.3],
        'water_reliability': [0.9, 0.8, 1.0],
        'road_distance_miles': [2.0, 5.0, 1.0],
        'trail_distance_miles': [1.0, 3.0, 0.5],
        'security_habitat_percent': [0.5, 1.2, 0.3],
        'snow_depth_inches': [12, 24, 6],
        'snow_crust_detected': [False, True, False],
        'temperature_f': [35, 28, 40],
        'precip_last_7_days_inches': [0.2, 0.5, 0.1],
        'cloud_cover_percent': [30, 50, 20],
        'ndvi': [0.4, 0.3, 0.5],
        'ndvi_age_days': [5, 7, 3],
        'irg': [0.002, 0.001, 0.003],
        'summer_integrated_ndvi': [40, 35, 45],
        'effective_illumination': [0.05, 0.02, 0.08],
        'month': [3, 1, 5],
        'day_of_year_sin': [0.5, -0.1, 0.9],
        'day_of_year_cos': [-0.9, 1.0, -0.4],
    }

    # Add location features if requested (for full model)
    if include_coords:
        sample_data['latitude'] = [43.5, 44.0, 43.8]
        sample_data['longitude'] = [-109.5, -110.0, -109.8]
        sample_data['year'] = [2024, 2024, 2024]

    df = pd.DataFrame(sample_data)
    output_path = Path('data/sample_prediction_input.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Sample input file created: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    if not include_coords:
        print(f"  Note: For generalizable model (no lat/lon/year)")
    else:
        print(f"  Note: Includes lat/lon/year for full model")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Predict elk presence for new locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict from CSV file
    python scripts/predict_elk_presence.py --input locations.csv --output predictions.csv

    # Single prediction with JSON features
    python scripts/predict_elk_presence.py --features '{"latitude": 43.5, "longitude": -109.5, ...}'

    # List required features
    python scripts/predict_elk_presence.py --list-features

    # Create sample input file
    python scripts/predict_elk_presence.py --create-sample
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=None,
        help='Path to trained model (.pkl file). Default: most recent model in models/'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=None,
        help='Input CSV file with feature values'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output CSV file for predictions (default: input_predictions.csv)'
    )
    parser.add_argument(
        '--features', '-f',
        type=str,
        default=None,
        help='JSON string with feature values for single prediction'
    )
    parser.add_argument(
        '--list-features',
        action='store_true',
        help='List required features for generalizable model'
    )
    parser.add_argument(
        '--list-features-full',
        action='store_true',
        help='List required features for full model (with lat/lon/year)'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample input CSV file for generalizable model'
    )
    parser.add_argument(
        '--use-full-model',
        action='store_true',
        help='Use full model with lat/lon instead of generalizable model'
    )

    args = parser.parse_args()

    # Handle special commands first
    if args.list_features:
        list_features(show_full=False)
        return 0

    if args.list_features_full:
        list_features(show_full=True)
        return 0

    if args.create_sample:
        create_sample_input(include_coords=args.use_full_model)
        return 0

    # Need either input file or features for prediction
    if not args.input and not args.features:
        parser.error("Either --input or --features is required for prediction")

    # Find or validate model
    if args.model:
        model_path = args.model
    else:
        prefer_generalizable = not args.use_full_model
        model_path = find_latest_model(prefer_generalizable=prefer_generalizable)
        if not model_path:
            print("Error: No model found in models/ directory")
            print("Train a model first with: python scripts/train_xgboost_model.py")
            return 1

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Load model
    model, feature_cols, label_encoders = load_model(model_path)

    # Make predictions
    if args.features:
        # Single prediction from JSON
        try:
            features = json.loads(args.features)
        except json.JSONDecodeError as e:
            print(f"Error parsing features JSON: {e}")
            return 1

        result = predict_single(model, features, feature_cols, label_encoders)

        print("\nPrediction Result:")
        print(f"  Prediction: {result['label']}")
        print(f"  Probability of presence: {result['probability_present']:.1%}")
        print(f"  Probability of absence: {result['probability_absent']:.1%}")
        print(f"  Confidence: {result['confidence']:.1%}")

    elif args.input:
        # Batch prediction from CSV
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}")
            return 1

        if args.output is None:
            args.output = args.input.parent / f"{args.input.stem}_predictions.csv"

        predict_batch(model, args.input, args.output, feature_cols, label_encoders)

    return 0


if __name__ == '__main__':
    sys.exit(main())
