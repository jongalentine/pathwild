#!/usr/bin/env python3
"""
Prepare training-ready feature datasets by excluding metadata columns.

This script creates a clean feature set for model training by:
1. Excluding dataset-specific metadata columns (identifiers, source info)
2. Keeping only environmental and biological features
3. Preserving the target variable (elk_present)

Usage:
    python scripts/prepare_training_features.py [input_file] [output_file]
    python scripts/prepare_training_features.py --all-datasets
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
from typing import List, Set

# Metadata columns to exclude from training (dataset-specific identifiers and source info)
METADATA_COLUMNS = {
    # Identifiers
    'route_id', 'id', 'Elk_ID', 'elk_id',
    # Area-specific metadata
    'distance_to_area_048_km', 'inside_area_048',
    # Source temporal metadata
    'mig', 'firstdate', 'lastdate', 'season',
    # UTM coordinates (we have lat/lon)
    'UTM_X', 'UTM_Y', 'utm_easting', 'utm_northing', 'Zone',
    # Source data columns (duplicates of standardized columns)
    'Lat', 'Long', 'DT', 'TZ', 't',
    # Absence generation metadata (could cause data leakage)
    'absence_strategy',
    # Feedground info (dataset-specific)
    'feedground',
    # Day of year (redundant with month, could cause leakage)
    'day_of_year',
    # Date (redundant with year/month, could cause leakage)
    'date',
    # Dataset name (if present)
    'dataset_name', 'dataset',
    # SNOTEL station name (categorical with high cardinality, risk of overfitting)
    # Note: snow_data_source and snow_station_distance_km are kept as features
    'snow_station_name'
}

# Columns that might be useful but should be evaluated carefully
# (year, month could be useful for temporal patterns, but might encode dataset info)
OPTIONAL_TEMPORAL_COLUMNS = {
    'year', 'month'  # Keep by default, but can exclude if causing leakage
}

# Core features that should always be included
CORE_FEATURES = {
    'latitude', 'longitude', 'elk_present',
    'elevation', 'slope_degrees', 'aspect_degrees',
    'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
    'water_distance_miles', 'water_reliability',
    'road_distance_miles', 'trail_distance_miles',
    'security_habitat_percent',
    'wolves_per_1000_elk', 'wolf_data_quality',
    'bear_activity_distance_miles', 'bear_data_quality',
    'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
    'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
    'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi',
    'pregnancy_rate'
}


def get_feature_columns(df: pd.DataFrame, exclude_temporal: bool = False) -> List[str]:
    """
    Get list of feature columns to use for training.
    
    Args:
        df: Input DataFrame
        exclude_temporal: If True, exclude year/month columns
    
    Returns:
        List of column names to use as features
    """
    all_cols = set(df.columns)
    
    # Start with core features that exist in the dataframe
    feature_cols = [col for col in CORE_FEATURES if col in all_cols]
    
    # Add any other columns that aren't metadata
    for col in all_cols:
        if col not in METADATA_COLUMNS:
            if col not in feature_cols:
                if exclude_temporal and col in OPTIONAL_TEMPORAL_COLUMNS:
                    continue
                feature_cols.append(col)
    
    # Ensure target is included (but not as a feature)
    if 'elk_present' in feature_cols:
        feature_cols.remove('elk_present')
    
    # Sort for consistency
    feature_cols.sort()
    
    return feature_cols


def prepare_training_dataset(
    input_file: Path,
    output_file: Path,
    exclude_temporal: bool = False
) -> pd.DataFrame:
    """
    Prepare a training-ready dataset by excluding metadata columns.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        exclude_temporal: If True, exclude year/month columns
    
    Returns:
        Prepared DataFrame
    """
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Get feature columns
    feature_cols = get_feature_columns(df, exclude_temporal=exclude_temporal)
    
    # Always include target
    if 'elk_present' not in feature_cols:
        feature_cols = ['elk_present'] + feature_cols
    
    # Select only feature columns
    df_features = df[feature_cols].copy()
    
    # Report excluded columns
    excluded = set(df.columns) - set(feature_cols)
    if excluded:
        print(f"\n  Excluded {len(excluded)} metadata columns:")
        for col in sorted(excluded):
            print(f"    - {col}")
    
    print(f"\n  Selected {len(feature_cols)} feature columns:")
    print(f"    - Target: elk_present")
    print(f"    - Features: {len(feature_cols) - 1}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    print(f"  Shape: {df_features.shape}")
    
    return df_features


def prepare_all_datasets(
    processed_dir: Path = Path('data/processed'),
    features_dir: Path = Path('data/features'),
    exclude_temporal: bool = False
):
    """Prepare training features for all combined datasets."""
    processed_dir = Path(processed_dir)
    features_dir = Path(features_dir)
    
    # Find all combined datasets
    combined_files = list(processed_dir.glob('combined_*_presence_absence.csv'))
    
    if not combined_files:
        print(f"No combined datasets found in {processed_dir}")
        return
    
    print(f"Found {len(combined_files)} dataset(s) to process\n")
    
    for input_file in combined_files:
        # Extract dataset name
        dataset_name = input_file.stem.replace('combined_', '').replace('_presence_absence', '')
        output_file = features_dir / f"{dataset_name}_features.csv"
        
        print(f"{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            prepare_training_dataset(input_file, output_file, exclude_temporal=exclude_temporal)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed {len(combined_files)} dataset(s)")
    print(f"Output directory: {features_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare training-ready feature datasets by excluding metadata columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare a single dataset
  python scripts/prepare_training_features.py \\
      data/processed/combined_north_bighorn_presence_absence.csv \\
      data/features/north_bighorn_features.csv
  
  # Prepare all datasets
  python scripts/prepare_training_features.py --all-datasets
  
  # Exclude temporal columns (year, month) to prevent data leakage
  python scripts/prepare_training_features.py --all-datasets --exclude-temporal
        """
    )
    parser.add_argument(
        'input_file',
        type=Path,
        nargs='?',
        default=None,
        help='Input CSV file (required if not using --all-datasets)'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        nargs='?',
        default=None,
        help='Output CSV file (required if not using --all-datasets)'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Process all combined datasets in data/processed/'
    )
    parser.add_argument(
        '--exclude-temporal',
        action='store_true',
        help='Exclude year/month columns to prevent potential data leakage'
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory containing processed datasets (default: data/processed)'
    )
    parser.add_argument(
        '--features-dir',
        type=Path,
        default=Path('data/features'),
        help='Directory to save feature datasets (default: data/features)'
    )
    
    args = parser.parse_args()
    
    if args.all_datasets:
        prepare_all_datasets(
            processed_dir=args.processed_dir,
            features_dir=args.features_dir,
            exclude_temporal=args.exclude_temporal
        )
    else:
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required unless using --all-datasets")
        
        prepare_training_dataset(
            args.input_file,
            args.output_file,
            exclude_temporal=args.exclude_temporal
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

