#!/usr/bin/env python3
"""
Assess the quality of processed environmental data for model training readiness.

This script analyzes various aspects of the integrated dataset to determine its suitability
for machine learning model training, including data volume, feature richness,
feature distribution, and data quality.

Usage:
    python scripts/assess_training_readiness.py [--dataset PATH] [--test-mode]
    
    --test-mode: Prefer test files over regular files when assessing all datasets.
                 Use this when --limit was used in the pipeline to assess test files.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional
from collections import Counter

def assess_training_readiness(dataset_path: Optional[Path] = None, test_mode: bool = False):
    """
    Perform a comprehensive assessment of the dataset's readiness for model training.
    If dataset_path is None, it assesses all known datasets.
    
    Args:
        dataset_path: Optional path to a specific integrated dataset CSV file.
                     If None, assesses all known datasets.
        test_mode: If True, prefer test files over regular files when assessing all datasets.
    
    Returns:
        Tuple of (readiness_score, readiness_percentage)
    """
    print("=" * 70)
    print("COMPREHENSIVE MODEL TRAINING READINESS ASSESSMENT")
    print("=" * 70)
    
    if dataset_path:
        # Handle test files (with _test suffix)
        is_test_file = '_test' in dataset_path.stem
        display_name = dataset_path.stem.replace('combined_', '').replace('_presence_absence', '').replace('_test', '')
        if is_test_file:
            display_name += ' (TEST)'
        
        datasets_to_assess = {display_name: dataset_path}
        print(f"\nAssessing single dataset: {dataset_path}")
        if is_test_file:
            print(f"  ⚠️  TEST MODE: Assessing test dataset (limited rows)")
    else:
        # When no specific dataset provided, look for both regular and test files
        # Prefer test files if they exist (indicates recent test run)
        base_datasets = {
            'North Bighorn': 'northern_bighorn',
            'South Bighorn': 'southern_bighorn',
            'National Refuge': 'national_refuge',
            'Southern GYE': 'southern_gye'
        }
        
        datasets_to_assess = {}
        for display_name, base_name in base_datasets.items():
            regular_path = Path(f'data/processed/combined_{base_name}_presence_absence.csv')
            test_path = Path(f'data/processed/combined_{base_name}_presence_absence_test.csv')
            
            if test_mode:
                # In test mode, prefer test files over regular files
                if test_path.exists():
                    datasets_to_assess[f"{display_name} (TEST)"] = test_path
                elif regular_path.exists():
                    # Fallback to regular file if test file doesn't exist
                    datasets_to_assess[display_name] = regular_path
            else:
                # Default: prefer regular files over test files (test files are for testing only)
                # Only use test files if regular file doesn't exist
                if regular_path.exists():
                    datasets_to_assess[display_name] = regular_path
                elif test_path.exists():
                    # Only use test file if regular file doesn't exist
                    datasets_to_assess[f"{display_name} (TEST)"] = test_path
        
        if datasets_to_assess:
            test_count = sum(1 for name in datasets_to_assess.keys() if '(TEST)' in name)
            if test_mode:
                print(f"\n⚠️  TEST MODE: Assessing test datasets (limited rows)")
                if test_count > 0:
                    print(f"   Found {test_count} test file(s)")
            elif test_count > 0:
                print(f"\nAssessing all known datasets ({test_count} test file(s) detected).")
            else:
                print("\nAssessing all known datasets.")
        else:
            print("\nNo datasets found to assess.")
    
    all_data = []
    for name, path in datasets_to_assess.items():
        if path.exists():
            df = pd.read_csv(path)
            df['dataset_name'] = name
            all_data.append(df)
        else:
            print(f"✗ Dataset not found: {path}")
    
    if not all_data:
        print('✗ No datasets found to assess.')
        return 0.0, 0.0
    
    combined_df = pd.concat(all_data, ignore_index=True)
    total_samples = len(combined_df)
    
    print(f"\n{'='*70}")
    print("DATA VOLUME ASSESSMENT")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples:,}")
    
    # Presence/absence balance
    if 'elk_present' in combined_df.columns:
        presence_count = combined_df['elk_present'].sum()
        absence_count = len(combined_df) - presence_count
        presence_pct = (presence_count / total_samples) * 100
        absence_pct = (absence_count / total_samples) * 100
        
        print(f"\nClass Distribution:")
        print(f"  Presence (elk=1): {presence_count:,} ({presence_pct:.1f}%)")
        print(f"  Absence (elk=0):  {absence_count:,} ({absence_pct:.1f}%)")
        
        if absence_count > 0:
            ratio = presence_count / absence_count
            print(f"  Ratio: {ratio:.2f}")
            
            if ratio < 0.5 or ratio > 2.0:
                print(f"  ⚠ Warning: Class imbalance detected (ratio outside [0.5, 2.0])")
            else:
                print(f"  ✓ Good class balance")
    
    # Volume recommendations
    print(f"\nVolume Recommendations:")
    if total_samples < 1000:
        print(f"  ⚠ Very small dataset (<1K samples). Consider collecting more data.")
        volume_score = 0.3
    elif total_samples < 10000:
        print(f"  ⚠ Small dataset (<10K samples). May work for simple models.")
        volume_score = 0.6
    elif total_samples < 100000:
        print(f"  ✓ Good dataset size (10K-100K samples). Suitable for most ML models.")
        volume_score = 0.9
    else:
        print(f"  ✓ Excellent dataset size (>100K samples). Suitable for complex models.")
        volume_score = 1.0
    
    print(f"\n{'='*70}")
    print("FEATURE RICHNESS ASSESSMENT")
    print(f"{'='*70}")
    
    # Environmental features
    env_features = [
        'elevation', 'slope_degrees', 'aspect_degrees',
        'canopy_cover_percent', 'land_cover_code', 'land_cover_type',
        'water_distance_miles', 'water_reliability',
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent'
    ]
    
    available_features = [f for f in env_features if f in combined_df.columns]
    missing_features = [f for f in env_features if f not in combined_df.columns]
    
    print(f"\nEnvironmental Features:")
    print(f"  Available: {len(available_features)}/{len(env_features)}")
    if available_features:
        print(f"    ✓ {', '.join(available_features)}")
    if missing_features:
        print(f"    ✗ Missing: {', '.join(missing_features)}")
    
    feature_richness_score = len(available_features) / len(env_features)
    
    print(f"\nFeature Richness Score: {feature_richness_score:.2%}")
    if feature_richness_score >= 0.8:
        print(f"  ✓ Excellent feature coverage")
    elif feature_richness_score >= 0.6:
        print(f"  ✓ Good feature coverage")
    else:
        print(f"  ⚠ Limited feature coverage - consider adding more environmental data")
    
    print(f"\n{'='*70}")
    print("FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    # Check for placeholder values
    placeholder_values = {
        'elevation': 8500.0,
        'slope_degrees': 15.0,
        'aspect_degrees': 180.0,
        'canopy_cover_percent': 30.0,
        'water_distance_miles': 0.5,
        'land_cover_code': 0
    }
    
    placeholder_counts = {}
    for feature, default_value in placeholder_values.items():
        if feature in combined_df.columns:
            placeholder_count = (combined_df[feature] == default_value).sum()
            if placeholder_count > 0:
                placeholder_pct = (placeholder_count / total_samples) * 100
                placeholder_counts[feature] = (placeholder_count, placeholder_pct)
    
    if placeholder_counts:
        print(f"\n⚠ Placeholder Values Detected:")
        for feature, (count, pct) in placeholder_counts.items():
            print(f"  {feature}: {count:,} ({pct:.1f}%) rows with placeholder value")
        print(f"  Recommendation: Re-run integration script to replace placeholders")
    else:
        print(f"\n✓ No placeholder values detected")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['elk_present', 'dataset_name']]
    
    for feature in numeric_features[:10]:  # Show first 10
        if feature in combined_df.columns:
            values = combined_df[feature].dropna()
            if len(values) > 0:
                print(f"  {feature}:")
                print(f"    Mean: {values.mean():.2f}, Std: {values.std():.2f}")
                print(f"    Min: {values.min():.2f}, Max: {values.max():.2f}")
                print(f"    Missing: {combined_df[feature].isna().sum():,} ({combined_df[feature].isna().sum()/total_samples*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("DATA QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    # Core environmental features that should be present in all rows
    core_features = [
        'latitude', 'longitude', 'elk_present',
        'elevation', 'slope_degrees', 'aspect_degrees',
        'canopy_cover_percent', 'land_cover_code',
        'water_distance_miles', 'water_reliability',
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent'
    ]
    
    # Check missing data only in core features (ignore dataset-specific metadata)
    core_missing_data = {}
    for feature in core_features:
        if feature in combined_df.columns:
            missing_count = combined_df[feature].isnull().sum()
            if missing_count > 0:
                core_missing_data[feature] = missing_count
    
    # Also check for all missing data (for reporting, but not scoring)
    all_missing_data = combined_df.isnull().sum()
    all_missing_data = all_missing_data[all_missing_data > 0]
    
    # Report core feature missing data
    if core_missing_data:
        print(f"\n⚠ Missing Data in Core Features:")
        for col, count in core_missing_data.items():
            pct = (count / total_samples) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\n✓ No missing data in core environmental features")
    
    # Report dataset-specific metadata columns (informational only)
    metadata_cols = [col for col in all_missing_data.index if col not in core_features]
    if metadata_cols:
        print(f"\n  Note: {len(metadata_cols)} dataset-specific metadata columns have missing data")
        print(f"        (expected - these columns vary by dataset source)")
        # Show a few examples
        examples = list(metadata_cols)[:5]
        for col in examples:
            count = all_missing_data[col]
            pct = (count / total_samples) * 100
            print(f"        - {col}: {count:,} ({pct:.1f}%)")
        if len(metadata_cols) > 5:
            print(f"        ... and {len(metadata_cols) - 5} more")
    
    # Data quality score based only on core features
    quality_score = 1.0
    if core_missing_data:
        max_missing_pct = max([(count / total_samples) * 100 for count in core_missing_data.values()])
        quality_score = max(0.0, 1.0 - (max_missing_pct / 100))
    
    if placeholder_counts:
        max_placeholder_pct = max([pct for _, pct in placeholder_counts.values()])
        # Reduce quality score based on placeholder percentage
        quality_score = min(quality_score, 1.0 - (max_placeholder_pct / 200))  # Less severe penalty
    
    print(f"\nData Quality Score: {quality_score:.2%}")
    print(f"  (Based on core environmental features only)")
    
    print(f"\n{'='*70}")
    print("MODEL TRAINING RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Overall readiness score
    readiness_score = (volume_score * 0.3 + feature_richness_score * 0.3 + quality_score * 0.4)
    readiness_pct = readiness_score * 100
    
    print(f"\nOverall Readiness Score: {readiness_pct:.1f}%")
    
    if readiness_pct >= 80:
        print(f"\n✓ Dataset is READY for model training")
        print(f"  Recommendations:")
        print(f"    - Proceed with model training")
        print(f"    - Consider cross-validation for robust evaluation")
        print(f"    - Monitor for overfitting with large feature set")
    elif readiness_pct >= 60:
        print(f"\n⚠ Dataset is MOSTLY READY with some concerns")
        print(f"  Recommendations:")
        if placeholder_counts:
            print(f"    - Re-run integration to replace placeholder values")
        if core_missing_data:
            print(f"    - Address missing data in core features (imputation or removal)")
        if feature_richness_score < 0.8:
            print(f"    - Consider adding more environmental features")
        print(f"    - Proceed with caution and monitor model performance")
    else:
        print(f"\n✗ Dataset is NOT READY for model training")
        print(f"  Critical Issues:")
        if volume_score < 0.5:
            print(f"    - Insufficient data volume")
        if feature_richness_score < 0.5:
            print(f"    - Missing critical environmental features")
        if quality_score < 0.5:
            print(f"    - Significant data quality issues")
        print(f"  Recommendations:")
        print(f"    - Collect more data or fix data quality issues before training")
        print(f"    - Re-run data integration pipeline")
    
    # Per-dataset breakdown if multiple datasets
    if len(datasets_to_assess) > 1:
        print(f"\n{'='*70}")
        print("PER-DATASET BREAKDOWN")
        print(f"{'='*70}")
        
        for name, path in datasets_to_assess.items():
            if path.exists():
                df = pd.read_csv(path)
                print(f"\n{name}:")
                print(f"  Samples: {len(df):,}")
                if 'elk_present' in df.columns:
                    pres = df['elk_present'].sum()
                    abs_count = len(df) - pres
                    print(f"  Presence: {pres:,}, Absence: {abs_count:,}")
                env_count = len([f for f in env_features if f in df.columns])
                print(f"  Environmental features: {env_count}/{len(env_features)}")
    
    return readiness_score, readiness_pct


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Assess the quality of processed environmental data for model training readiness."
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=None,
        help='Path to a specific integrated dataset CSV file. If not provided, all known datasets will be assessed.'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Prefer test files over regular files when assessing all datasets. Use this when --limit was used in the pipeline.'
    )
    args = parser.parse_args()
    
    score, score_pct = assess_training_readiness(args.dataset, test_mode=args.test_mode)
    
    return 0 if score > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
