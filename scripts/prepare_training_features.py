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
import numpy as np
from pathlib import Path
import sys
import importlib.util
from typing import List, Set, Optional

# Metadata columns to exclude from training (dataset-specific identifiers and source info)
METADATA_COLUMNS = {
    # Identifiers
    'route_id', 'id', 'Elk_ID', 'elk_id', 'point_index',  # point_index: route-specific metadata (only present for LineString routes, missing for Point geometries, CSV files, and absence data)
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
    # Date (redundant with year/month/day_of_year, could cause leakage)
    # Note: day_of_year is NOT excluded by default - see OPTIONAL_TEMPORAL_COLUMNS
    'date',
    # Dataset name (if present)
    'dataset_name', 'dataset',
    # SNOTEL station name (categorical with high cardinality, risk of overfitting)
    # Note: snow_data_source and snow_station_distance_km are kept as features
    'snow_station_name',
    # Predator/wildlife data quality and activity metrics (temporarily excluded until better modeling approach)
    'wolf_data_quality', 'bear_data_quality', 'bear_activity_distance_miles', 'wolves_per_1000_elk',
    # Pregnancy rate (temporarily excluded until better modeling approach)
    'pregnancy_rate'
}

# Columns that might be useful but should be evaluated carefully
# (year, month, day_of_year could be useful for temporal patterns, but might encode dataset info)
# Note: day_of_year should be encoded cyclically (sin/cos) if used to avoid treating day 365 and day 1 as very different
OPTIONAL_TEMPORAL_COLUMNS = {
    'year', 'month', 'day_of_year'  # Keep by default, but can exclude if causing leakage
}

# Core features that should always be included
CORE_FEATURES = {
    'latitude', 'longitude', 'elk_present',
    'elevation', 'slope_degrees', 'aspect_degrees',
    'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
    'water_distance_miles', 'water_reliability',
    'road_distance_miles', 'trail_distance_miles',
    'security_habitat_percent',
    'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
    'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
    'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
}


def get_feature_columns(df: pd.DataFrame, exclude_temporal: bool = False) -> List[str]:
    """
    Get list of feature columns to use for training.
    
    Args:
        df: Input DataFrame
        exclude_temporal: If True, exclude year/month/day_of_year columns
    
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
    exclude_temporal: bool = False,
    include_day_of_year: bool = True
) -> pd.DataFrame:
    """
    Prepare a training-ready dataset by excluding metadata columns.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        exclude_temporal: If True, exclude year/month/day_of_year columns
        include_day_of_year: If True, include day_of_year (encoded cyclically if present)
    
    Returns:
        Prepared DataFrame
    """
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Debug: Check if day_of_year exists
    if 'day_of_year' in df.columns:
        n_valid_doy = df['day_of_year'].notna().sum()
        print(f"  Found day_of_year column with {n_valid_doy:,} valid values (out of {len(df):,})")
    else:
        print(f"  ⚠ day_of_year column not found in input file")
    
    # Get feature columns
    feature_cols = get_feature_columns(df, exclude_temporal=exclude_temporal)
    
    # Handle day_of_year if present and requested
    # Note: day_of_year may already be in feature_cols from get_feature_columns()
    # We want to replace it with cyclical encoding
    # Note: We allow day_of_year even if exclude_temporal is True, since it's cyclically encoded
    if include_day_of_year and 'day_of_year' in df.columns:
        # Check if day_of_year has valid values
        valid_day_of_year = df['day_of_year'].notna()
        n_valid = valid_day_of_year.sum()
        
        if n_valid > 0:
            # Encode day_of_year cyclically (sin/cos) to handle circular nature
            # This treats day 365 and day 1 as similar (both near year end/start)
            # This is important because day 365 (Dec 31) and day 1 (Jan 1) are temporally close
            df['day_of_year_sin'] = np.nan
            df['day_of_year_cos'] = np.nan
            df.loc[valid_day_of_year, 'day_of_year_sin'] = np.sin(
                2 * np.pi * df.loc[valid_day_of_year, 'day_of_year'] / 365.25
            )
            df.loc[valid_day_of_year, 'day_of_year_cos'] = np.cos(
                2 * np.pi * df.loc[valid_day_of_year, 'day_of_year'] / 365.25
            )
            
            # Remove raw day_of_year from features (if present) and add cyclical encoding
            if 'day_of_year' in feature_cols:
                feature_cols.remove('day_of_year')
            
            # Add cyclical encoding (only if not already present)
            # Use a set to avoid duplicates, then convert back to list
            feature_cols_set = set(feature_cols)
            feature_cols_set.add('day_of_year_sin')
            feature_cols_set.add('day_of_year_cos')
            feature_cols = list(feature_cols_set)
            
            # Verify columns were created in dataframe
            if 'day_of_year_sin' in df.columns and 'day_of_year_cos' in df.columns:
                print(f"  ✓ Added cyclical encoding for day_of_year (sin/cos) - {n_valid:,} valid values")
            else:
                print(f"  ⚠ ERROR: Failed to create day_of_year sin/cos columns in dataframe")
        else:
            print(f"  ⚠ day_of_year column exists but has no valid values ({len(df)} rows), skipping cyclical encoding")
            # Remove day_of_year from features if it was added
            if 'day_of_year' in feature_cols:
                feature_cols.remove('day_of_year')
    elif not include_day_of_year:
        # Explicitly remove day_of_year if user requested exclusion
        if 'day_of_year' in feature_cols:
            feature_cols.remove('day_of_year')
        if 'day_of_year_sin' in feature_cols:
            feature_cols.remove('day_of_year_sin')
        if 'day_of_year_cos' in feature_cols:
            feature_cols.remove('day_of_year_cos')
        print(f"  ✓ Excluded day_of_year as requested")
    elif 'day_of_year' in df.columns and exclude_temporal:
        # day_of_year will be excluded by get_feature_columns, but make sure sin/cos aren't there
        if 'day_of_year_sin' in feature_cols:
            feature_cols.remove('day_of_year_sin')
        if 'day_of_year_cos' in feature_cols:
            feature_cols.remove('day_of_year_cos')
    
    # Always include target
    if 'elk_present' not in feature_cols:
        feature_cols = ['elk_present'] + feature_cols
    
    # Ensure all feature columns exist in dataframe (for debugging)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠ WARNING: Some feature columns are missing from dataframe: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Verify day_of_year sin/cos are included if they should be
    if include_day_of_year:
        if 'day_of_year_sin' in df.columns and 'day_of_year_sin' not in feature_cols:
            print(f"  ⚠ WARNING: day_of_year_sin exists in dataframe but not in feature_cols, adding it")
            feature_cols.append('day_of_year_sin')
        if 'day_of_year_cos' in df.columns and 'day_of_year_cos' not in feature_cols:
            print(f"  ⚠ WARNING: day_of_year_cos exists in dataframe but not in feature_cols, adding it")
            feature_cols.append('day_of_year_cos')
    
    # Select only feature columns
    df_features = df[feature_cols].copy()
    
    # Verify day_of_year sin/cos are in output
    if include_day_of_year:
        if 'day_of_year_sin' in df_features.columns and 'day_of_year_cos' in df_features.columns:
            n_valid = df_features['day_of_year_sin'].notna().sum()
            n_total = len(df_features)
            n_missing = n_total - n_valid
            print(f"  ✓ Verified: day_of_year_sin and day_of_year_cos are in output")
            print(f"    - Valid values: {n_valid:,} ({n_valid/n_total*100:.1f}%)")
            if n_missing > 0:
                print(f"    - Missing values: {n_missing:,} ({n_missing/n_total*100:.1f}%) - expected when source day_of_year is missing")
        else:
            print(f"  ⚠ WARNING: day_of_year sin/cos columns missing from output!")
            print(f"     Available columns: {[c for c in df_features.columns if 'day' in c.lower()]}")
    
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
    exclude_temporal: bool = False,
    include_day_of_year: bool = True,
    limit: Optional[int] = None
):
    """Prepare training features for all combined datasets."""
    processed_dir = Path(processed_dir)
    features_dir = Path(features_dir)
    
    # When limit is set, look for test files (created by integrate_environmental_features with --limit)
    if limit is not None:
        # Find test files: combined_*_presence_absence_test.csv
        combined_files = list(processed_dir.glob('combined_*_presence_absence_test.csv'))
        if not combined_files:
            print(f"⚠️  TEST MODE: No test files found in {processed_dir}")
            print(f"   Looking for: combined_*_presence_absence_test.csv")
            print(f"   Run integrate_environmental_features.py with --limit first")
            return
    else:
        # Find regular combined datasets
        combined_files = list(processed_dir.glob('combined_*_presence_absence.csv'))
        # Exclude test files when not in test mode
        combined_files = [f for f in combined_files if '_test' not in f.stem]
    
    if not combined_files:
        print(f"No combined datasets found in {processed_dir}")
        return
    
    print(f"Found {len(combined_files)} dataset(s) to process")
    if limit is not None:
        print(f"⚠️  TEST MODE: Processing test files (limit={limit})")
    print()
    
    for input_file in combined_files:
        # Extract dataset name
        dataset_name = input_file.stem.replace('combined_', '').replace('_presence_absence', '').replace('_test', '')
        
        # When limit is set, save to test file
        if limit is not None:
            output_file = features_dir / f"{dataset_name}_features_test.csv"
        else:
            output_file = features_dir / f"{dataset_name}_features.csv"
        
        print(f"{'='*70}")
        print(f"Processing: {dataset_name}")
        if limit is not None:
            print(f"  (TEST MODE)")
        print(f"{'='*70}")
        
        try:
            prepare_training_dataset(
                input_file, 
                output_file, 
                exclude_temporal=exclude_temporal,
                include_day_of_year=include_day_of_year
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed {len(combined_files)} dataset(s)")
    print(f"Output directory: {features_dir}")
    
    # Combine all feature files into complete_context.csv (or complete_context_test.csv in test mode)
    print(f"\n{'='*70}")
    if limit is None:
        print("Combining all feature files into complete_context.csv...")
    else:
        print("Combining all test feature files into complete_context_test.csv...")
    print(f"{'='*70}")
    
    try:
        # Import combine_feature_files module
        scripts_dir = Path(__file__).parent
        combine_script = scripts_dir / "combine_feature_files.py"
        
        if combine_script.exists():
            spec = importlib.util.spec_from_file_location(
                "combine_feature_files",
                combine_script
            )
            combine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(combine_module)
            
            # Determine output file based on test mode
            if limit is None:
                output_file = features_dir / "complete_context.csv"
                test_mode = False
            else:
                output_file = features_dir / "complete_context_test.csv"
                test_mode = True
            
            combine_module.combine_feature_files(
                features_dir=features_dir,
                output_file=output_file,
                exclude_test_files=True,
                test_mode=test_mode
            )
        else:
            print(f"⚠️  Warning: combine_feature_files.py not found, skipping combination step")
    except Exception as e:
        print(f"⚠️  Warning: Error combining feature files: {e}")
        if limit is None:
            print(f"   You can manually run: python scripts/combine_feature_files.py")
        else:
            print(f"   You can manually run: python scripts/combine_feature_files.py --test")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare training-ready feature datasets by excluding metadata columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare a single dataset
  python scripts/prepare_training_features.py \\
      data/processed/combined_northern_bighorn_presence_absence.csv \\
      data/features/northern_bighorn_features.csv
  
  # Prepare all datasets
  python scripts/prepare_training_features.py --all-datasets
  
  # Exclude temporal columns (year, month, day_of_year) to prevent data leakage
  python scripts/prepare_training_features.py --all-datasets --exclude-temporal
  
  # Include day_of_year with cyclical encoding (default, recommended)
  python scripts/prepare_training_features.py --all-datasets
  
  # Exclude only day_of_year (keep year/month)
  python scripts/prepare_training_features.py --all-datasets --exclude-day-of-year
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
        help='Exclude year/month/day_of_year columns to prevent potential data leakage'
    )
    parser.add_argument(
        '--exclude-day-of-year',
        action='store_true',
        help='Exclude day_of_year even if other temporal columns are kept (default: include day_of_year with cyclical encoding)'
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
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit mode: process test files (combined_*_presence_absence_test.csv) and save to test output files'
    )
    
    args = parser.parse_args()
    
    include_day_of_year = not args.exclude_day_of_year
    
    if args.all_datasets:
        prepare_all_datasets(
            processed_dir=args.processed_dir,
            features_dir=args.features_dir,
            exclude_temporal=args.exclude_temporal,
            include_day_of_year=include_day_of_year,
            limit=args.limit
        )
    else:
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required unless using --all-datasets")
        
        prepare_training_dataset(
            args.input_file,
            args.output_file,
            exclude_temporal=args.exclude_temporal,
            include_day_of_year=include_day_of_year
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

