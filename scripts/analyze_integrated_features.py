#!/usr/bin/env python3
"""
Analyze integrated environmental features in processed datasets.

This script validates feature integration and provides statistics on:
- Feature value ranges
- Placeholder value detection
- Missing value counts
- Feature distribution statistics
- Data quality warnings

Usage:
    python scripts/analyze_integrated_features.py [dataset_path]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Placeholder values that indicate data needs to be replaced
PLACEHOLDER_VALUES = {
    'elevation': 8500.0,
    'water_distance_miles': 0.5,
    'canopy_cover_percent': 30.0,
    'land_cover_code': 0
}

def analyze_integrated_features(dataset_path: Path):
    """
    Analyze integrated features in a processed dataset.
    
    Args:
        dataset_path: Path to the integrated dataset CSV file
    """
    print("=" * 70)
    print("INTEGRATED FEATURES ANALYSIS")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"✗ Error: File not found: {dataset_path}")
        return False
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    # Environmental feature columns
    env_columns = [
        'elevation', 'slope_degrees', 'aspect_degrees',
        'water_distance_miles', 'water_reliability',
        'canopy_cover_percent', 'land_cover_code',
        'distance_to_road_miles', 'distance_to_trail_miles',
        'security_habitat_percent'
    ]
    
    # Find which environmental columns exist
    existing_env_cols = [col for col in env_columns if col in df.columns]
    
    print(f"\n{'='*70}")
    print("FEATURE VALUE RANGES")
    print(f"{'='*70}")
    
    for col in existing_env_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Min: {values.min():.2f}")
                print(f"  Max: {values.max():.2f}")
                print(f"  Mean: {values.mean():.2f}")
                print(f"  Median: {values.median():.2f}")
                print(f"  Std Dev: {values.std():.2f}")
                print(f"  Unique values: {values.nunique():,}")
    
    print(f"\n{'='*70}")
    print("PLACEHOLDER VALUE DETECTION")
    print(f"{'='*70}")
    
    placeholder_found = False
    for col, placeholder_val in PLACEHOLDER_VALUES.items():
        if col in df.columns:
            placeholder_count = (df[col] == placeholder_val).sum()
            if placeholder_count > 0:
                placeholder_pct = (placeholder_count / len(df)) * 100
                print(f"\n⚠ {col}:")
                print(f"  Placeholder value ({placeholder_val}): {placeholder_count:,} rows ({placeholder_pct:.1f}%)")
                placeholder_found = True
    
    if not placeholder_found:
        print("\n✓ No placeholder values detected")
    
    print(f"\n{'='*70}")
    print("MISSING VALUE COUNTS")
    print(f"{'='*70}")
    
    missing_found = False
    for col in existing_env_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                print(f"\n⚠ {col}:")
                print(f"  Missing values: {missing_count:,} rows ({missing_pct:.1f}%)")
                missing_found = True
    
    if not missing_found:
        print("\n✓ No missing values in environmental features")
    
    print(f"\n{'='*70}")
    print("FEATURE DISTRIBUTION STATISTICS")
    print(f"{'='*70}")
    
    # Presence/absence distribution
    if 'elk_present' in df.columns:
        presence_count = df['elk_present'].sum()
        absence_count = len(df) - presence_count
        total = len(df)
        print(f"\nClass Distribution:")
        print(f"  Presence (elk=1): {presence_count:,} ({presence_count/total*100:.1f}%)")
        print(f"  Absence (elk=0):  {absence_count:,} ({absence_count/total*100:.1f}%)")
    
    # Geographic distribution
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print(f"\nGeographic Distribution:")
        print(f"  Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"  Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    
    print(f"\n{'='*70}")
    print("DATA QUALITY WARNINGS")
    print(f"{'='*70}")
    
    warnings = []
    
    # Check for placeholder values
    if placeholder_found:
        warnings.append("⚠ Placeholder values detected - re-run integration script to replace")
    
    # Check for missing values
    if missing_found:
        warnings.append("⚠ Missing values detected in environmental features")
    
    # Check for suspicious value ranges
    if 'elevation' in df.columns:
        elev_values = df['elevation'].dropna()
        if len(elev_values) > 0:
            if elev_values.min() < 1000 or elev_values.max() > 15000:
                warnings.append("⚠ Elevation values outside expected Wyoming range (1000-15000m)")
    
    if 'water_distance_miles' in df.columns:
        water_dist = df['water_distance_miles'].dropna()
        if len(water_dist) > 0 and water_dist.max() > 50:
            warnings.append("⚠ Some water distances exceed 50 miles (may indicate missing data)")
    
    if 'canopy_cover_percent' in df.columns:
        canopy = df['canopy_cover_percent'].dropna()
        if len(canopy) > 0:
            if canopy.min() < 0 or canopy.max() > 100:
                warnings.append("⚠ Canopy cover values outside valid range (0-100%)")
    
    if warnings:
        for warning in warnings:
            print(f"\n{warning}")
    else:
        print("\n✓ No data quality warnings")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze integrated environmental features in processed datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a specific dataset
    python scripts/analyze_integrated_features.py data/processed/combined_north_bighorn_presence_absence.csv
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=Path,
        help='Path to the integrated dataset CSV file'
    )
    
    args = parser.parse_args()
    
    success = analyze_integrated_features(args.dataset_path)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
