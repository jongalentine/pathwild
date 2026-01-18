#!/usr/bin/env python3
"""
Analyze the complete_context.csv file to verify data quality.

Checks:
- Row/column counts
- Class distribution
- Missing values
- Feature ranges and distributions
- Dataset representation
- Data consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def analyze_complete_context(csv_path: Path):
    """Analyze complete_context.csv for data quality."""
    
    print("="*80)
    print("COMPLETE_CONTEXT.CSV DATA QUALITY ANALYSIS")
    print("="*80)
    print(f"\nReading: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print("1. BASIC INFORMATION")
    print(f"{'='*80}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n{'='*80}")
    print("2. TARGET VARIABLE (elk_present)")
    print(f"{'='*80}")
    if 'elk_present' in df.columns:
        target_counts = df['elk_present'].value_counts().sort_index()
        target_pct = df['elk_present'].value_counts(normalize=True).sort_index() * 100
        
        print(f"Class distribution:")
        for label in target_counts.index:
            print(f"  {label}: {target_counts[label]:,} ({target_pct[label]:.2f}%)")
        
        print(f"\nClass balance ratio: {target_counts[0] / target_counts[1]:.2f}:1 (0:1)")
        
        # Check for missing target
        missing_target = df['elk_present'].isna().sum()
        if missing_target > 0:
            print(f"\n⚠️  WARNING: {missing_target:,} rows have missing target values!")
    else:
        print("⚠️  ERROR: 'elk_present' column not found!")
    
    print(f"\n{'='*80}")
    print("3. MISSING VALUES")
    print(f"{'='*80}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    # Only show columns with missing values
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_cols) > 0:
        print("Columns with missing values:")
        for col, row in missing_cols.iterrows():
            print(f"  {col}: {row['Missing Count']:,} ({row['Missing %']:.2f}%)")
    else:
        print("✓ No missing values found!")
    
    print(f"\n{'='*80}")
    print("4. COLUMN SUMMARY")
    print(f"{'='*80}")
    print(f"All columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"  {i:2d}. {col} ({dtype})")
    
    print(f"\n{'='*80}")
    print("5. NUMERICAL FEATURES - BASIC STATISTICS")
    print(f"{'='*80}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'elk_present' in numeric_cols:
        numeric_cols.remove('elk_present')  # Don't show target in stats
    
    if len(numeric_cols) > 0:
        print(f"Numerical features: {len(numeric_cols)}")
        print("\nStatistics (sample - first 10):")
        stats = df[numeric_cols[:10]].describe().T
        stats['range'] = stats['max'] - stats['min']
        print(stats[['count', 'mean', 'std', 'min', 'max', 'range']].to_string())
        
        if len(numeric_cols) > 10:
            print(f"\n... and {len(numeric_cols) - 10} more numerical columns")
    
    print(f"\n{'='*80}")
    print("6. CATEGORICAL FEATURES")
    print(f"{'='*80}")
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"Categorical features: {len(categorical_cols)}")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"\n  {col}:")
            print(f"    Unique values: {unique_count}")
            if unique_count <= 20:
                value_counts = df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    print(f"      {val}: {count:,} ({pct:.2f}%)")
            else:
                print(f"    Top 5 values:")
                for val, count in df[col].value_counts().head(5).items():
                    pct = (count / len(df)) * 100
                    print(f"      {val}: {count:,} ({pct:.2f}%)")
    else:
        print("No categorical features found")
    
    print(f"\n{'='*80}")
    print("7. TEMPORAL FEATURES")
    print(f"{'='*80}")
    temporal_cols = ['year', 'month', 'day_of_year_sin', 'day_of_year_cos']
    found_temporal = [col for col in temporal_cols if col in df.columns]
    
    if found_temporal:
        print(f"Temporal features found: {found_temporal}")
        for col in found_temporal:
            if col in df.columns:
                if col == 'year':
                    print(f"\n  {col}:")
                    print(f"    Range: {df[col].min():.0f} - {df[col].max():.0f}")
                    year_counts = df[col].value_counts().sort_index()
                    print(f"    Distribution:")
                    for year, count in year_counts.items():
                        pct = (count / len(df)) * 100
                        print(f"      {year:.0f}: {count:,} ({pct:.2f}%)")
                elif col == 'month':
                    print(f"\n  {col}:")
                    print(f"    Range: {df[col].min():.0f} - {df[col].max():.0f}")
                    month_counts = df[col].value_counts().sort_index()
                    print(f"    Distribution:")
                    for month, count in month_counts.items():
                        pct = (count / len(df)) * 100
                        print(f"      Month {month:.0f}: {count:,} ({pct:.2f}%)")
                elif col in ['day_of_year_sin', 'day_of_year_cos']:
                    print(f"\n  {col}:")
                    print(f"    Range: {df[col].min():.4f} - {df[col].max():.4f}")
                    print(f"    Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}")
    else:
        print("⚠️  No temporal features found!")
    
    print(f"\n{'='*80}")
    print("8. LOCATION FEATURES")
    print(f"{'='*80}")
    location_cols = ['latitude', 'longitude']
    found_location = [col for col in location_cols if col in df.columns]
    
    if found_location:
        print("Location features:")
        if 'latitude' in df.columns:
            print(f"  Latitude range: {df['latitude'].min():.6f} - {df['latitude'].max():.6f}")
        if 'longitude' in df.columns:
            print(f"  Longitude range: {df['longitude'].min():.6f} - {df['longitude'].max():.6f}")
        
        # Check if coordinates are in Wyoming (approximate bounds)
        wyoming_lat_bounds = (40.9, 45.0)
        wyoming_lon_bounds = (-111.1, -104.0)
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            in_wyoming = (
                (df['latitude'] >= wyoming_lat_bounds[0]) & 
                (df['latitude'] <= wyoming_lat_bounds[1]) &
                (df['longitude'] >= wyoming_lon_bounds[0]) & 
                (df['longitude'] <= wyoming_lon_bounds[1])
            )
            out_of_bounds = (~in_wyoming).sum()
            if out_of_bounds > 0:
                print(f"\n  ⚠️  WARNING: {out_of_bounds:,} points ({out_of_bounds/len(df)*100:.2f}%) outside Wyoming bounds")
            else:
                print(f"\n  ✓ All coordinates are within Wyoming bounds")
    else:
        print("⚠️  Location features not found!")
    
    print(f"\n{'='*80}")
    print("9. DATA QUALITY CHECKS")
    print(f"{'='*80}")
    
    issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️  {duplicates:,} duplicate rows found")
    else:
        print("✓ No duplicate rows")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum().sum()
    if inf_counts > 0:
        issues.append(f"⚠️  {inf_counts:,} infinite values found in numerical columns")
    else:
        print("✓ No infinite values")
    
    # Check key feature ranges
    if 'elevation' in df.columns:
        if df['elevation'].min() < 0 or df['elevation'].max() > 14000:
            issues.append(f"⚠️  Elevation values may be out of range: {df['elevation'].min():.1f} - {df['elevation'].max():.1f}")
        else:
            print(f"✓ Elevation range looks reasonable: {df['elevation'].min():.1f} - {df['elevation'].max():.1f} feet")
    
    if 'ndvi' in df.columns:
        if df['ndvi'].min() < -1 or df['ndvi'].max() > 1:
            issues.append(f"⚠️  NDVI values may be out of range: {df['ndvi'].min():.3f} - {df['ndvi'].max():.3f}")
        else:
            print(f"✓ NDVI range looks reasonable: {df['ndvi'].min():.3f} - {df['ndvi'].max():.3f}")
    
    if 'temperature_f' in df.columns:
        temp_range = (df['temperature_f'].min(), df['temperature_f'].max())
        if temp_range[0] < -50 or temp_range[1] > 120:
            issues.append(f"⚠️  Temperature values may be out of range: {temp_range[0]:.1f} - {temp_range[1]:.1f} °F")
        else:
            print(f"✓ Temperature range looks reasonable: {temp_range[0]:.1f} - {temp_range[1]:.1f} °F")
    
    # Print any issues found
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious data quality issues found!")
    
    print(f"\n{'='*80}")
    print("10. SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Total samples: {len(df):,}")
    print(f"✓ Total features: {len(df.columns)}")
    
    if 'elk_present' in df.columns:
        positive_samples = (df['elk_present'] == 1).sum()
        negative_samples = (df['elk_present'] == 0).sum()
        print(f"✓ Positive samples (elk_present=1): {positive_samples:,}")
        print(f"✓ Negative samples (elk_present=0): {negative_samples:,}")
        print(f"✓ Class balance: {negative_samples/positive_samples:.2f}:1")
    
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        print(f"✓ No missing values")
    else:
        print(f"⚠️  Total missing values: {missing_total:,}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    csv_path = Path('data/features/complete_context.csv')
    
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    analyze_complete_context(csv_path)
