#!/usr/bin/env python3
"""
Apply feature engineering recommendations based on notebook analysis.

This script applies the following changes to feature files:
1. Removes redundant features (high correlation pairs)
2. Removes weak discriminators (low Cohen's d)
3. Transforms skewed distributions
4. Optionally flags/removes geographic outliers

Usage:
    python scripts/apply_feature_recommendations.py
    python scripts/apply_feature_recommendations.py --input data/features/complete_context.csv
    python scripts/apply_feature_recommendations.py --all-datasets
    python scripts/apply_feature_recommendations.py --dry-run
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Features to remove based on analysis
FEATURES_TO_REMOVE = {
    # Redundant features (high correlation with another feature)
    'snow_water_equiv_inches',      # r=0.96 with snow_depth_inches
    'cloud_adjusted_illumination',  # r=0.94 with effective_illumination

    # Weak discriminators (Cohen's d ≈ 0, no predictive value)
    'moon_altitude_midnight',       # d=0.0001
    'moon_phase',                   # d=0.008
}

# Features to transform
FEATURES_TO_TRANSFORM = {
    'security_habitat_percent': 'log1p',  # Extreme skewness (10.1) and kurtosis (111.5)
}

# Wyoming bounding box for geographic validation
WYOMING_BOUNDS = {
    'lat_min': 40.99,
    'lat_max': 45.01,
    'lon_min': -111.06,
    'lon_max': -104.05,
}


def remove_features(df: pd.DataFrame, features: set) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove specified features from DataFrame.

    Returns:
        Tuple of (modified DataFrame, list of removed columns)
    """
    removed = []
    for col in features:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed.append(col)
    return df, removed


def transform_features(df: pd.DataFrame, transforms: dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply transformations to specified features.

    Supported transforms:
        - 'log1p': log(1 + x) - handles zeros, reduces right skew
        - 'cap_95': cap at 95th percentile
        - 'sqrt': square root transform

    Returns:
        Tuple of (modified DataFrame, list of transformed columns)
    """
    transformed = []

    for col, transform in transforms.items():
        if col not in df.columns:
            continue

        original_stats = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'skew': df[col].skew(),
        }

        if transform == 'log1p':
            # Handle negative values by shifting if necessary
            min_val = df[col].min()
            if min_val < 0:
                df[col] = np.log1p(df[col] - min_val)
            else:
                df[col] = np.log1p(df[col])
            transformed.append(f"{col} (log1p)")

        elif transform == 'cap_95':
            cap_value = df[col].quantile(0.95)
            df[col] = df[col].clip(upper=cap_value)
            transformed.append(f"{col} (capped at {cap_value:.2f})")

        elif transform == 'sqrt':
            min_val = df[col].min()
            if min_val < 0:
                df[col] = np.sqrt(df[col] - min_val)
            else:
                df[col] = np.sqrt(df[col])
            transformed.append(f"{col} (sqrt)")

        new_stats = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'skew': df[col].skew(),
        }

        print(f"    {col}: skew {original_stats['skew']:.2f} -> {new_stats['skew']:.2f}")

    return df, transformed


def flag_geographic_outliers(
    df: pd.DataFrame,
    bounds: dict = WYOMING_BOUNDS,
    remove: bool = False
) -> Tuple[pd.DataFrame, int]:
    """
    Flag or remove points outside Wyoming bounds.

    Returns:
        Tuple of (modified DataFrame, count of outliers)
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df, 0

    outside_bounds = (
        (df['latitude'] < bounds['lat_min']) |
        (df['latitude'] > bounds['lat_max']) |
        (df['longitude'] < bounds['lon_min']) |
        (df['longitude'] > bounds['lon_max'])
    )

    n_outliers = outside_bounds.sum()

    if remove and n_outliers > 0:
        df = df[~outside_bounds].copy()

    return df, n_outliers


def apply_recommendations(
    input_file: Path,
    output_file: Optional[Path] = None,
    remove_geo_outliers: bool = False,
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Apply all feature recommendations to a single file.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    # Load data
    df = pd.read_csv(input_file)
    original_shape = df.shape
    print(f"  Loaded: {original_shape[0]:,} rows, {original_shape[1]} columns")

    # 1. Remove features
    print(f"\n  Removing redundant/weak features:")
    df, removed = remove_features(df, FEATURES_TO_REMOVE)
    for col in removed:
        print(f"    - {col}")
    if not removed:
        print(f"    (none found)")

    # 2. Transform features
    print(f"\n  Transforming skewed features:")
    df, transformed = transform_features(df, FEATURES_TO_TRANSFORM)
    if not transformed:
        print(f"    (none found)")

    # 3. Handle geographic outliers
    df, n_geo_outliers = flag_geographic_outliers(df, remove=remove_geo_outliers)
    if n_geo_outliers > 0:
        action = "Removed" if remove_geo_outliers else "Found"
        print(f"\n  Geographic outliers: {action} {n_geo_outliers:,} points outside Wyoming bounds")

    # Summary
    print(f"\n  Summary:")
    print(f"    Original: {original_shape[0]:,} rows, {original_shape[1]} columns")
    print(f"    Final:    {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"    Removed features: {len(removed)}")
    print(f"    Transformed features: {len(transformed)}")

    # Save
    if not dry_run and output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n  Saved to: {output_file}")
    elif dry_run:
        print(f"\n  [DRY RUN] Would save to: {output_file}")

    return df


def process_all_datasets(
    features_dir: Path = Path('data/features'),
    output_dir: Optional[Path] = None,
    remove_geo_outliers: bool = False,
    dry_run: bool = False
):
    """
    Process all feature files and regenerate complete_context.csv.
    """
    if output_dir is None:
        output_dir = features_dir / 'optimized'

    # Find individual dataset feature files (exclude complete_context and test files)
    feature_files = [
        f for f in features_dir.glob('*_features.csv')
        if 'complete_context' not in f.name and '_test' not in f.name
    ]

    if not feature_files:
        print(f"No feature files found in {features_dir}")
        return

    print(f"Found {len(feature_files)} feature file(s) to process")
    print(f"Output directory: {output_dir}")

    processed_dfs = []

    for input_file in sorted(feature_files):
        output_file = output_dir / input_file.name
        df = apply_recommendations(
            input_file,
            output_file,
            remove_geo_outliers=remove_geo_outliers,
            dry_run=dry_run
        )
        processed_dfs.append(df)

    # Combine into complete_context_optimized.csv
    if processed_dfs:
        print(f"\n{'='*70}")
        print("Combining into complete_context_optimized.csv")
        print(f"{'='*70}")

        combined = pd.concat(processed_dfs, ignore_index=True)

        print(f"  Combined shape: {combined.shape[0]:,} rows, {combined.shape[1]} columns")

        # Target distribution
        if 'elk_present' in combined.columns:
            presence_rate = combined['elk_present'].mean()
            print(f"  Target distribution:")
            print(f"    Presence (1): {(combined['elk_present']==1).sum():,} ({presence_rate*100:.1f}%)")
            print(f"    Absence (0):  {(combined['elk_present']==0).sum():,} ({(1-presence_rate)*100:.1f}%)")

        output_file = output_dir / 'complete_context_optimized.csv'
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_file, index=False)
            print(f"\n  Saved to: {output_file}")
        else:
            print(f"\n  [DRY RUN] Would save to: {output_file}")

        # Print final column list
        print(f"\n  Final features ({combined.shape[1]} columns):")
        for col in sorted(combined.columns):
            print(f"    - {col}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply feature engineering recommendations to feature files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets and create optimized versions
    python scripts/apply_feature_recommendations.py --all-datasets

    # Process a single file
    python scripts/apply_feature_recommendations.py \\
        --input data/features/complete_context.csv \\
        --output data/features/optimized/complete_context_optimized.csv

    # Preview changes without saving
    python scripts/apply_feature_recommendations.py --all-datasets --dry-run

    # Also remove geographic outliers
    python scripts/apply_feature_recommendations.py --all-datasets --remove-geo-outliers

Changes applied:
    Removed features (4):
        - snow_water_equiv_inches (redundant with snow_depth_inches, r=0.96)
        - cloud_adjusted_illumination (redundant with effective_illumination, r=0.94)
        - moon_altitude_midnight (no predictive value, Cohen's d=0.0001)
        - moon_phase (minimal predictive value, Cohen's d=0.008)

    Transformed features (1):
        - security_habitat_percent: log1p transform (skewness 10.1 -> ~1.5)
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        help='Input feature file to process'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (default: data/features/optimized/<input_name>)'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Process all feature files in data/features/'
    )
    parser.add_argument(
        '--features-dir',
        type=Path,
        default=Path('data/features'),
        help='Directory containing feature files (default: data/features)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: data/features/optimized)'
    )
    parser.add_argument(
        '--remove-geo-outliers',
        action='store_true',
        help='Remove points outside Wyoming bounds (default: keep them)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving files'
    )

    args = parser.parse_args()

    if args.all_datasets:
        process_all_datasets(
            features_dir=args.features_dir,
            output_dir=args.output_dir,
            remove_geo_outliers=args.remove_geo_outliers,
            dry_run=args.dry_run
        )
    elif args.input:
        if args.output is None:
            args.output = args.input.parent / 'optimized' / args.input.name

        apply_recommendations(
            args.input,
            args.output,
            remove_geo_outliers=args.remove_geo_outliers,
            dry_run=args.dry_run
        )
    else:
        parser.error("Either --input or --all-datasets is required")

    return 0


if __name__ == '__main__':
    sys.exit(main())
