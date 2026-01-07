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
    'land_cover_code': 0,
    # Temporal features
    'temperature_f': 45.0,
    'precip_last_7_days_inches': 0.0,
    'ndvi': 0.5,
    'ndvi_age_days': 8,
    'irg': 0.0,
    'summer_integrated_ndvi': 0.0
}

# Expected value ranges for Wyoming (for validation)
EXPECTED_RANGES = {
    'latitude': (40.9950, 45.0060),  # Wyoming bounds
    'longitude': (-111.0550, -104.0530),  # Wyoming bounds
    'elevation': (1000, 14000),  # Feet (lowest point ~3000ft, highest ~13900ft)
    'slope_degrees': (0, 60),  # Degrees (0-60 typical, rare up to 80+)
    'aspect_degrees': (0, 360),  # Degrees (0-360)
    'canopy_cover_percent': (0, 100),  # Percent (0-100)
    'water_distance_miles': (0, 50),  # Miles (0-50 typical, some rare outliers)
    'water_reliability': (0, 1),  # 0-1 scale
    'road_distance_miles': (0, 100),  # Miles (0-100 within Wyoming)
    'trail_distance_miles': (0, 100),  # Miles (0-100 within Wyoming)
    'security_habitat_percent': (0, 100),  # Percent (0-100)
    'snow_depth_inches': (0, 300),  # Inches (0-300 typical for Wyoming)
    'snow_water_equiv_inches': (0, 200),  # Inches (typically < snow depth)
    'temperature_f': (-50, 110),  # Fahrenheit (Wyoming extremes)
    'precip_last_7_days_inches': (0, 20),  # Inches (0-20 typical)
    'cloud_cover_percent': (0, 100),  # Percent (0-100)
    'ndvi': (-1, 1),  # Normalized Difference Vegetation Index
    'ndvi_age_days': (0, 365),  # Days since acquisition
    'irg': (-1, 1),  # Instantaneous Rate of Green-up
    'summer_integrated_ndvi': (0, 10),  # Integrated NDVI over summer (sum of ~8 satellite images, max ~8)
}

# Valid NLCD land cover codes (2016 version)
VALID_NLCD_CODES = {
    11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95
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
        'road_distance_miles', 'trail_distance_miles',
        'security_habitat_percent',
        # Temporal features (SNOTEL, weather, satellite)
        'snow_depth_inches', 'snow_water_equiv_inches', 'snow_crust_detected',
        'snow_data_source', 'snow_station_name', 'snow_station_distance_km',
        'temperature_f', 'precip_last_7_days_inches', 'cloud_cover_percent',
        'ndvi', 'ndvi_age_days', 'irg', 'summer_integrated_ndvi'
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
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"  Min: {values.min():.2f}")
                    print(f"  Max: {values.max():.2f}")
                    print(f"  Mean: {values.mean():.2f}")
                    print(f"  Median: {values.median():.2f}")
                    print(f"  Std Dev: {values.std():.2f}")
                    print(f"  Unique values: {values.nunique():,}")
                else:
                    # For non-numeric columns, just show value counts
                    print(f"  Unique values: {values.nunique():,}")
                    value_counts = values.value_counts().head(10)
                    if len(value_counts) > 0:
                        print(f"  Top values:")
                        for val, count in value_counts.items():
                            pct = (count / len(values)) * 100
                            print(f"    {val}: {count:,} ({pct:.1f}%)")
    
    print(f"\n{'='*70}")
    print("PLACEHOLDER VALUE DETECTION")
    print(f"{'='*70}")
    
    placeholder_found = False
    for col, placeholder_val in PLACEHOLDER_VALUES.items():
        if col in df.columns:
            # For numeric columns, use tolerance for floating point comparison
            if pd.api.types.is_numeric_dtype(df[col]):
                if col in ['temperature_f']:
                    # Allow small tolerance for temperature (0.1°F)
                    placeholder_count = (abs(df[col] - placeholder_val) < 0.1).sum()
                elif col in ['ndvi', 'irg', 'precip_last_7_days_inches']:
                    # Allow small tolerance for small decimal values
                    placeholder_count = (abs(df[col] - placeholder_val) < 0.01).sum()
                else:
                    placeholder_count = (df[col] == placeholder_val).sum()
            else:
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
    
    # Columns that are expected to be missing when using estimates
    expected_missing_for_estimates = {'snow_station_name', 'snow_station_distance_km'}
    
    unexpected_missing_found = False
    for col in existing_env_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # Check if missing values are expected (for estimate rows)
                if col in expected_missing_for_estimates:
                    # Only warn if there are missing values for rows that should have real SNOTEL data
                    if 'snow_data_source' in df.columns:
                        # Count missing values that are NOT from estimate rows (unexpected missing)
                        unexpected_missing = df[(df[col].isna()) & (df['snow_data_source'] == 'snotel')].shape[0]
                        expected_missing = missing_count - unexpected_missing
                        
                        if unexpected_missing > 0:
                            # Report unexpected missing
                            missing_pct = (unexpected_missing / len(df)) * 100
                            print(f"\n⚠ {col}:")
                            print(f"  Missing values: {unexpected_missing:,} rows ({missing_pct:.1f}%) - unexpected (should have station data)")
                            if expected_missing > 0:
                                print(f"  Note: {expected_missing:,} rows ({expected_missing/len(df)*100:.1f}%) are missing because they use elevation estimates (expected)")
                            unexpected_missing_found = True
                        elif expected_missing > 0:
                            # Only expected missing - just note it, don't warn
                            print(f"\n✓ {col}:")
                            print(f"  Missing values: {expected_missing:,} rows ({expected_missing/len(df)*100:.1f}%) - expected (elevation estimates)")
                    else:
                        # Can't determine if missing is expected, so report all as unexpected
                        missing_pct = (missing_count / len(df)) * 100
                        print(f"\n⚠ {col}:")
                        print(f"  Missing values: {missing_count:,} rows ({missing_pct:.1f}%)")
                        unexpected_missing_found = True
                else:
                    # Not an expected-missing column, report all missing as unexpected
                    missing_pct = (missing_count / len(df)) * 100
                    print(f"\n⚠ {col}:")
                    print(f"  Missing values: {missing_count:,} rows ({missing_pct:.1f}%)")
                    unexpected_missing_found = True
    
    if not unexpected_missing_found:
        print("\n✓ No unexpected missing values in environmental features")
    
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
    print("INFRASTRUCTURE DATA QUALITY (ROADS & TRAILS)")
    print(f"{'='*70}")
    
    # Road distance analysis
    if 'road_distance_miles' in df.columns:
        road_dist = df['road_distance_miles'].dropna()
        if len(road_dist) > 0:
            print(f"\nRoad Distance Statistics:")
            print(f"  Total rows with road distance: {len(road_dist):,} ({len(road_dist)/len(df)*100:.1f}%)")
            print(f"  Min: {road_dist.min():.3f} miles")
            print(f"  Max: {road_dist.max():.3f} miles")
            print(f"  Mean: {road_dist.mean():.3f} miles")
            print(f"  Median: {road_dist.median():.3f} miles")
            print(f"  Std Dev: {road_dist.std():.3f} miles")
            
            # Distribution analysis
            print(f"\n  Distance Distribution:")
            bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
            labels = ['<1mi', '1-5mi', '5-10mi', '10-20mi', '20-50mi', '50-100mi', '100+mi']
            road_binned = pd.cut(road_dist, bins=bins, labels=labels, include_lowest=True)
            for label in labels:
                count = (road_binned == label).sum()
                if count > 0:
                    pct = (count / len(road_dist)) * 100
                    print(f"    {label:10s}: {count:6,} ({pct:5.1f}%)")
            
            # Check for default/placeholder values
            default_road_dist = 2.0  # Default when roads not loaded
            if (road_dist == default_road_dist).sum() > len(road_dist) * 0.1:  # >10% have default
                default_count = (road_dist == default_road_dist).sum()
                default_pct = (default_count / len(road_dist)) * 100
                print(f"\n  ⚠ Warning: {default_count:,} rows ({default_pct:.1f}%) have default road distance ({default_road_dist} mi)")
                print(f"    This may indicate roads dataset was not loaded properly")
            
            # Check for unreasonable values
            if road_dist.max() > 100:
                very_far = (road_dist > 100).sum()
                print(f"\n  ⚠ Warning: {very_far} rows have road distance >100 miles (outside Wyoming bounds?)")
            
            if road_dist.min() < 0:
                negative = (road_dist < 0).sum()
                print(f"\n  ⚠ Warning: {negative} rows have negative road distance")
        else:
            print(f"\n⚠ No road distance data available")
    else:
        print(f"\n⚠ 'road_distance_miles' column not found")
    
    # Trail distance analysis
    if 'trail_distance_miles' in df.columns:
        trail_dist = df['trail_distance_miles'].dropna()
        if len(trail_dist) > 0:
            print(f"\nTrail Distance Statistics:")
            print(f"  Total rows with trail distance: {len(trail_dist):,} ({len(trail_dist)/len(df)*100:.1f}%)")
            print(f"  Min: {trail_dist.min():.3f} miles")
            print(f"  Max: {trail_dist.max():.3f} miles")
            print(f"  Mean: {trail_dist.mean():.3f} miles")
            print(f"  Median: {trail_dist.median():.3f} miles")
            print(f"  Std Dev: {trail_dist.std():.3f} miles")
            
            # Distribution analysis
            print(f"\n  Distance Distribution:")
            bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
            labels = ['<1mi', '1-5mi', '5-10mi', '10-20mi', '20-50mi', '50-100mi', '100+mi']
            trail_binned = pd.cut(trail_dist, bins=bins, labels=labels, include_lowest=True)
            for label in labels:
                count = (trail_binned == label).sum()
                if count > 0:
                    pct = (count / len(trail_dist)) * 100
                    print(f"    {label:10s}: {count:6,} ({pct:5.1f}%)")
            
            # Check for default/placeholder values
            default_trail_dist = 1.5  # Default when trails not loaded
            if (trail_dist == default_trail_dist).sum() > len(trail_dist) * 0.1:  # >10% have default
                default_count = (trail_dist == default_trail_dist).sum()
                default_pct = (default_count / len(trail_dist)) * 100
                print(f"\n  ⚠ Warning: {default_count:,} rows ({default_pct:.1f}%) have default trail distance ({default_trail_dist} mi)")
                print(f"    This may indicate trails dataset was not loaded properly")
            
            # Check for unreasonable values
            if trail_dist.max() > 100:
                very_far = (trail_dist > 100).sum()
                print(f"\n  ⚠ Warning: {very_far} rows have trail distance >100 miles (outside Wyoming bounds?)")
            
            if trail_dist.min() < 0:
                negative = (trail_dist < 0).sum()
                print(f"\n  ⚠ Warning: {negative} rows have negative trail distance")
        else:
            print(f"\n⚠ No trail distance data available")
    else:
        print(f"\n⚠ 'trail_distance_miles' column not found")
    
    # Combined infrastructure analysis
    if 'road_distance_miles' in df.columns and 'trail_distance_miles' in df.columns:
        both_present = df[df['road_distance_miles'].notna() & df['trail_distance_miles'].notna()]
        if len(both_present) > 0:
            print(f"\nCombined Infrastructure Analysis:")
            print(f"  Rows with both road and trail distances: {len(both_present):,}")
            
            # Compare road vs trail distances
            road_vs_trail = both_present['road_distance_miles'] - both_present['trail_distance_miles']
            closer_to_road = (road_vs_trail < 0).sum()
            closer_to_trail = (road_vs_trail > 0).sum()
            equal_distance = (road_vs_trail == 0).sum()
            
            print(f"\n  Distance Comparison:")
            print(f"    Closer to road: {closer_to_road:,} rows ({closer_to_road/len(both_present)*100:.1f}%)")
            print(f"    Closer to trail: {closer_to_trail:,} rows ({closer_to_trail/len(both_present)*100:.1f}%)")
            if equal_distance > 0:
                print(f"    Equal distance: {equal_distance:,} rows ({equal_distance/len(both_present)*100:.1f}%)")
            
            # Correlation
            if len(both_present) > 10:
                correlation = both_present[['road_distance_miles', 'trail_distance_miles']].corr().iloc[0, 1]
                print(f"\n  Correlation (road vs trail distance): {correlation:.3f}")
                if correlation > 0.7:
                    print(f"    → Strong positive correlation (remote areas are remote from both)")
                elif correlation > 0.3:
                    print(f"    → Moderate positive correlation (some relationship)")
                else:
                    print(f"    → Weak correlation (roads and trails have different distributions)")
            
            # Check for locations very close to roads but far from trails (expected pattern)
            very_close_to_road = both_present[both_present['road_distance_miles'] < 1]
            if len(very_close_to_road) > 0:
                avg_trail_dist_near_roads = very_close_to_road['trail_distance_miles'].mean()
                print(f"\n  Locations <1 mile from roads:")
                print(f"    Count: {len(very_close_to_road):,}")
                print(f"    Average trail distance: {avg_trail_dist_near_roads:.2f} miles")
    
    print(f"\n{'='*70}")
    print("NDVI DATA QUALITY")
    print(f"{'='*70}")
    
    # NDVI analysis
    if 'ndvi' in df.columns:
        ndvi = df['ndvi'].dropna()
        if len(ndvi) > 0:
            print(f"\nNDVI Statistics:")
            print(f"  Total rows with NDVI: {len(ndvi):,} ({len(ndvi)/len(df)*100:.1f}%)")
            print(f"  Range: {ndvi.min():.3f} to {ndvi.max():.3f}")
            print(f"  Mean: {ndvi.mean():.3f}")
            print(f"  Median: {ndvi.median():.3f}")
            print(f"  Std Dev: {ndvi.std():.3f}")
            
            # Check for placeholder values
            placeholder_ndvi = 0.5  # Default placeholder
            placeholder_count = (ndvi == placeholder_ndvi).sum()
            if placeholder_count > 0:
                placeholder_pct = (placeholder_count / len(ndvi)) * 100
                print(f"\n  ⚠ Placeholder NDVI ({placeholder_ndvi}): {placeholder_count:,} rows ({placeholder_pct:.1f}%)")
                if placeholder_pct > 10:
                    print(f"    → Consider running integration script to fetch real NDVI data from AppEEARS")
            else:
                print(f"\n  ✓ No placeholder NDVI values detected")
            
            # Validate NDVI range (-1 to 1)
            invalid_ndvi = ndvi[(ndvi < -1) | (ndvi > 1)]
            if len(invalid_ndvi) > 0:
                print(f"\n  ⚠ Warning: {len(invalid_ndvi):,} rows have NDVI outside valid range (-1 to 1)")
            
            # Seasonal variation check (if timestamp/date column exists)
            if 'timestamp' in df.columns or 'date' in df.columns:
                date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df_with_date = df[df[date_col].notna() & df['ndvi'].notna()].copy()
                if len(df_with_date) > 0:
                    try:
                        df_with_date['month'] = pd.to_datetime(df_with_date[date_col]).dt.month
                        
                        print(f"\n  Seasonal NDVI Variation:")
                        seasonal_stats = df_with_date.groupby('month')['ndvi'].agg(['mean', 'min', 'max'])
                        for month in sorted(seasonal_stats.index):
                            month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
                            stats = seasonal_stats.loc[month]
                            print(f"    {month_name:9s}: Mean={stats['mean']:.3f}, Range=[{stats['min']:.3f}, {stats['max']:.3f}]")
                        
                        # Check if seasonal variation exists (should see higher NDVI in summer)
                        summer_ndvi = df_with_date[df_with_date['month'].isin([6, 7, 8])]['ndvi'].mean()
                        winter_ndvi = df_with_date[df_with_date['month'].isin([12, 1, 2])]['ndvi'].mean()
                        if summer_ndvi < winter_ndvi + 0.1:  # Summer should be significantly higher
                            print(f"\n    ⚠ Warning: Summer NDVI ({summer_ndvi:.3f}) not significantly higher than winter ({winter_ndvi:.3f})")
                            print(f"      Expected seasonal variation may be missing (could indicate placeholder data)")
                        else:
                            print(f"\n    ✓ Seasonal variation detected: Summer NDVI ({summer_ndvi:.3f}) > Winter NDVI ({winter_ndvi:.3f})")
                    except Exception as e:
                        print(f"\n    Note: Could not analyze seasonal variation: {e}")
        
        # NDVI age analysis
        if 'ndvi_age_days' in df.columns:
            ndvi_age = df['ndvi_age_days'].dropna()
            if len(ndvi_age) > 0:
                print(f"\nNDVI Age Statistics:")
                print(f"  Range: {ndvi_age.min():.0f} to {ndvi_age.max():.0f} days")
                print(f"  Mean: {ndvi_age.mean():.1f} days")
                print(f"  Median: {ndvi_age.median():.0f} days")
                
                # Check for placeholder age
                placeholder_age = 8
                placeholder_age_count = (ndvi_age == placeholder_age).sum()
                if placeholder_age_count > len(ndvi_age) * 0.5:  # >50% have placeholder
                    placeholder_age_pct = (placeholder_age_count / len(ndvi_age)) * 100
                    print(f"\n  ⚠ Warning: {placeholder_age_count:,} rows ({placeholder_age_pct:.1f}%) have placeholder age ({placeholder_age} days)")
                    print(f"    → This may indicate placeholder NDVI data")
                
                # Very old NDVI (>90 days) may be less reliable
                old_ndvi = (ndvi_age > 90).sum()
                if old_ndvi > 0:
                    old_pct = (old_ndvi / len(ndvi_age)) * 100
                    print(f"\n  ⚠ Warning: {old_ndvi:,} rows ({old_pct:.1f}%) have NDVI age >90 days (may be less reliable)")
        
        # IRG (Instantaneous Rate of Green-up) analysis
        if 'irg' in df.columns:
            irg = df['irg'].dropna()
            if len(irg) > 0:
                print(f"\nIRG (Instantaneous Rate of Green-up) Statistics:")
                print(f"  Range: {irg.min():.3f} to {irg.max():.3f}")
                print(f"  Mean: {irg.mean():.3f}")
                
                # Check for placeholder IRG
                placeholder_irg = 0.0
                placeholder_irg_count = (irg == placeholder_irg).sum()
                if placeholder_irg_count > len(irg) * 0.8:  # >80% have zero
                    placeholder_irg_pct = (placeholder_irg_count / len(irg)) * 100
                    print(f"\n  ⚠ Warning: {placeholder_irg_count:,} rows ({placeholder_irg_pct:.1f}%) have IRG = 0")
                    print(f"    → This may indicate placeholder data (IRG should vary, especially in spring/fall)")
                else:
                    # IRG should be positive in spring, negative in fall
                    if 'timestamp' in df.columns or 'date' in df.columns:
                        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                        df_with_date = df[df[date_col].notna() & df['irg'].notna()].copy()
                        if len(df_with_date) > 0:
                            try:
                                df_with_date['month'] = pd.to_datetime(df_with_date[date_col]).dt.month
                                spring_irg = df_with_date[df_with_date['month'].isin([4, 5])]['irg'].mean()
                                fall_irg = df_with_date[df_with_date['month'].isin([9, 10])]['irg'].mean()
                                if spring_irg <= 0:
                                    print(f"\n    ⚠ Warning: Spring IRG ({spring_irg:.3f}) should be positive (vegetation growing)")
                                if fall_irg >= 0:
                                    print(f"\n    ⚠ Warning: Fall IRG ({fall_irg:.3f}) should typically be negative (vegetation declining)")
                            except Exception:
                                pass
        
        # Summer integrated NDVI analysis
        if 'summer_integrated_ndvi' in df.columns:
            summer_ndvi = df['summer_integrated_ndvi'].dropna()
            if len(summer_ndvi) > 0:
                print(f"\nSummer Integrated NDVI Statistics:")
                print(f"  Total rows: {len(summer_ndvi):,} ({len(summer_ndvi)/len(df)*100:.1f}%)")
                print(f"  Range: {summer_ndvi.min():.1f} to {summer_ndvi.max():.1f}")
                print(f"  Mean: {summer_ndvi.mean():.1f}")
                
                # Check for placeholder (0.0)
                placeholder_summer = 0.0
                placeholder_summer_count = (summer_ndvi == placeholder_summer).sum()
                if placeholder_summer_count > 0:
                    placeholder_summer_pct = (placeholder_summer_count / len(summer_ndvi)) * 100
                    print(f"\n  ⚠ Placeholder summer NDVI ({placeholder_summer}): {placeholder_summer_count:,} rows ({placeholder_summer_pct:.1f}%)")
                
                # Check for reasonable values (sum of ~8 satellite images over summer)
                if 'ndvi' in df.columns:
                    both_present = df[df['summer_integrated_ndvi'].notna() & df['ndvi'].notna()]
                    if len(both_present) > 0:
                        # Summer integrated NDVI is sum of ~8 satellite images (Landsat 16-day revisit)
                        # With NDVI range -1 to 1, max sum of 8 images = ~8
                        # Typical values: 1-5 (3-4 images found on average)
                        if summer_ndvi.max() < 0.5:
                            print(f"\n  ⚠ Warning: Maximum summer integrated NDVI ({summer_ndvi.max():.1f}) is unusually low")
                            print(f"    → Expected range: 0.5-8.0 (sum of ~8 satellite images over summer)")
                            print(f"    → Typical values: 1-5 (3-4 images found on average)")
                        elif summer_ndvi.max() > 10:
                            print(f"\n  ⚠ Warning: Maximum summer integrated NDVI ({summer_ndvi.max():.1f}) is unusually high")
                            print(f"    → Expected range: 0.5-8.0 (sum of ~8 satellite images over summer)")
                            print(f"    → Values >10 may indicate incorrect calculation or too many images")
    else:
        print(f"\n⚠ 'ndvi' column not found")
    
    print(f"\n{'='*70}")
    print("WEATHER & TEMPERATURE DATA QUALITY")
    print(f"{'='*70}")
    
    # Temperature analysis
    if 'temperature_f' in df.columns:
        temp = df['temperature_f'].dropna()
        if len(temp) > 0:
            print(f"\nTemperature Statistics:")
            print(f"  Total rows with temperature: {len(temp):,} ({len(temp)/len(df)*100:.1f}%)")
            print(f"  Range: {temp.min():.1f} to {temp.max():.1f} °F")
            print(f"  Mean: {temp.mean():.1f} °F")
            print(f"  Median: {temp.median():.1f} °F")
            print(f"  Std Dev: {temp.std():.1f} °F")
            
            # Check for placeholder temperature
            placeholder_temp = 45.0  # Default placeholder
            placeholder_temp_count = (abs(temp - placeholder_temp) < 0.1).sum()  # Allow small tolerance
            if placeholder_temp_count > 0:
                placeholder_temp_pct = (placeholder_temp_count / len(temp)) * 100
                print(f"\n  ⚠ Placeholder temperature (~{placeholder_temp}°F): {placeholder_temp_count:,} rows ({placeholder_temp_pct:.1f}%)")
                if placeholder_temp_pct > 10:
                    print(f"    → Consider running integration script to fetch real weather data from PRISM/Open-Meteo")
            
            # Seasonal variation check
            if 'timestamp' in df.columns or 'date' in df.columns:
                date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df_with_date = df[df[date_col].notna() & df['temperature_f'].notna()].copy()
                if len(df_with_date) > 0:
                    try:
                        df_with_date['month'] = pd.to_datetime(df_with_date[date_col]).dt.month
                        
                        print(f"\n  Seasonal Temperature Variation:")
                        seasonal_stats = df_with_date.groupby('month')['temperature_f'].agg(['mean', 'min', 'max'])
                        for month in sorted(seasonal_stats.index):
                            month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
                            stats = seasonal_stats.loc[month]
                            print(f"    {month_name:9s}: Mean={stats['mean']:.1f}°F, Range=[{stats['min']:.1f}, {stats['max']:.1f}]°F")
                        
                        # Check if seasonal variation exists (should see warmer summer)
                        summer_temp = df_with_date[df_with_date['month'].isin([6, 7, 8])]['temperature_f'].mean()
                        winter_temp = df_with_date[df_with_date['month'].isin([12, 1, 2])]['temperature_f'].mean()
                        temp_range = summer_temp - winter_temp
                        if temp_range < 20:  # Summer should be significantly warmer
                            print(f"\n    ⚠ Warning: Temperature variation ({temp_range:.1f}°F) is smaller than expected")
                            print(f"      Expected ~30-50°F difference between summer and winter in Wyoming")
                            print(f"      This may indicate placeholder data or insufficient date range")
                        else:
                            print(f"\n    ✓ Seasonal variation detected: Summer ({summer_temp:.1f}°F) > Winter ({winter_temp:.1f}°F)")
                    except Exception as e:
                        print(f"\n    Note: Could not analyze seasonal variation: {e}")
            
            # Elevation correlation check
            if 'elevation' in df.columns:
                both_present = df[df['temperature_f'].notna() & df['elevation'].notna()]
                if len(both_present) > 50:
                    correlation = both_present[['elevation', 'temperature_f']].corr().iloc[0, 1]
                    # Expected negative correlation (higher elevation = colder)
                    if correlation > -0.2:  # Weak or positive correlation
                        print(f"\n  ⚠ Warning: Weak correlation ({correlation:.2f}) between elevation and temperature")
                        print(f"    → Expected negative correlation (higher elevation = colder)")
                        print(f"    → This may indicate placeholder data or insufficient geographic variation")
                    else:
                        print(f"\n  ✓ Elevation-temperature correlation: {correlation:.2f} (as expected: higher elevation = colder)")
    else:
        print(f"\n⚠ 'temperature_f' column not found")
    
    # Precipitation analysis
    if 'precip_last_7_days_inches' in df.columns:
        precip = df['precip_last_7_days_inches'].dropna()
        if len(precip) > 0:
            print(f"\nPrecipitation (Last 7 Days) Statistics:")
            print(f"  Total rows with precipitation: {len(precip):,} ({len(precip)/len(df)*100:.1f}%)")
            print(f"  Range: {precip.min():.3f} to {precip.max():.3f} inches")
            print(f"  Mean: {precip.mean():.3f} inches")
            print(f"  Median: {precip.median():.3f} inches")
            
            # Check for placeholder (0.0)
            placeholder_precip = 0.0
            zero_precip_count = (precip == placeholder_precip).sum()
            if zero_precip_count > len(precip) * 0.7:  # >70% have zero
                zero_precip_pct = (zero_precip_count / len(precip)) * 100
                print(f"\n  ⚠ Warning: {zero_precip_count:,} rows ({zero_precip_pct:.1f}%) have zero precipitation")
                print(f"    → This may indicate placeholder data (some precipitation is expected in Wyoming)")
            
            # Unusually high precipitation
            high_precip = (precip > 5.0).sum()
            if high_precip > 0:
                high_precip_pct = (high_precip / len(precip)) * 100
                print(f"\n  ⚠ Warning: {high_precip:,} rows ({high_precip_pct:.1f}%) have >5 inches precipitation in 7 days")
                print(f"    → This is unusually high for Wyoming (verify data source)")
    else:
        print(f"\n⚠ 'precip_last_7_days_inches' column not found")
    
    # Cloud cover analysis
    if 'cloud_cover_percent' in df.columns:
        cloud = df['cloud_cover_percent'].dropna()
        if len(cloud) > 0:
            print(f"\nCloud Cover Statistics:")
            print(f"  Total rows with cloud cover: {len(cloud):,} ({len(cloud)/len(df)*100:.1f}%)")
            print(f"  Range: {cloud.min():.0f} to {cloud.max():.0f}%")
            print(f"  Mean: {cloud.mean():.1f}%")
            
            # Check for placeholder values (typically 20, 30, or 40)
            placeholder_cloud = [20, 30, 40]  # Common placeholder values
            placeholder_cloud_count = cloud[cloud.isin(placeholder_cloud)].sum()
            if placeholder_cloud_count > 0:
                # Count how many have placeholder values
                placeholder_count = cloud.isin(placeholder_cloud).sum()
                placeholder_pct = (placeholder_count / len(cloud)) * 100
                if placeholder_pct > 50:  # >50% have placeholder
                    print(f"\n  ⚠ Warning: {placeholder_count:,} rows ({placeholder_pct:.1f}%) have placeholder cloud cover ({placeholder_cloud})")
                    print(f"    → Cloud cover data may not be available from PRISM/Open-Meteo (not in free tier)")
            
            # Check range
            invalid_cloud = cloud[(cloud < 0) | (cloud > 100)]
            if len(invalid_cloud) > 0:
                print(f"\n  ⚠ Warning: {len(invalid_cloud):,} rows have cloud cover outside valid range (0-100%)")
    else:
        print(f"\n⚠ 'cloud_cover_percent' column not found")
    
    print(f"\n{'='*70}")
    print("SNOTEL DATA QUALITY")
    print(f"{'='*70}")
    
    # Check SNOTEL data source
    if 'snow_data_source' in df.columns:
        snotel_source_counts = df['snow_data_source'].value_counts()
        total = len(df)
        
        print(f"\nSNOTEL Data Source Distribution:")
        for source, count in snotel_source_counts.items():
            pct = (count / total) * 100
            status = "✓" if source == "snotel" else "⚠"
            print(f"  {status} {source}: {count:,} rows ({pct:.1f}%)")
        
        # Real SNOTEL data statistics
        real_snotel = df[df['snow_data_source'] == 'snotel']
        if len(real_snotel) > 0:
            print(f"\nReal SNOTEL Data Details:")
            print(f"  Total rows with real SNOTEL: {len(real_snotel):,}")
            
            # Station usage
            if 'snow_station_name' in df.columns:
                unique_stations = real_snotel['snow_station_name'].nunique()
                print(f"  Unique SNOTEL stations used: {unique_stations}")
                print(f"\n  Top 10 most used stations:")
                station_counts = real_snotel['snow_station_name'].value_counts().head(10)
                for station, count in station_counts.items():
                    pct = (count / len(real_snotel)) * 100
                    print(f"    {station}: {count:,} rows ({pct:.1f}%)")
            
            # Distance statistics
            if 'snow_station_distance_km' in df.columns:
                distances = real_snotel['snow_station_distance_km'].dropna()
                if len(distances) > 0:
                    print(f"\n  Station Distance Statistics:")
                    print(f"    Min: {distances.min():.1f} km")
                    print(f"    Max: {distances.max():.1f} km")
                    print(f"    Mean: {distances.mean():.1f} km")
                    print(f"    Median: {distances.median():.1f} km")
                    
                    # Distance warnings
                    far_stations = real_snotel[real_snotel['snow_station_distance_km'] > 50]
                    if len(far_stations) > 0:
                        pct = (len(far_stations) / len(real_snotel)) * 100
                        print(f"    ⚠ {len(far_stations):,} rows ({pct:.1f}%) use stations >50 km away")
        else:
            print("\n  ⚠ No real SNOTEL data found (all data is estimated)")
        
        # Estimated data statistics
        estimated = df[df['snow_data_source'] == 'estimate']
        if len(estimated) > 0:
            print(f"\nEstimated Data (Elevation-based):")
            print(f"  Total rows with estimates: {len(estimated):,}")
            pct = (len(estimated) / total) * 100
            if pct > 50:
                print(f"  ⚠ Warning: {pct:.1f}% of data uses elevation estimates (consider expanding SNOTEL coverage)")
            
            # Validate elevation-based estimates (DEM formula approach)
            if 'elevation' in estimated.columns and 'snow_depth_inches' in estimated.columns:
                print(f"\n  Elevation-Based Estimate Validation (DEM Formula):")
                
                # Check if low elevations have minimal snow (as expected with DEM formula)
                low_elev_estimates = estimated[(estimated['elevation'] < 3000) & (estimated['snow_depth_inches'].notna())]
                if len(low_elev_estimates) > 0:
                    high_snow_low_elev = low_elev_estimates[low_elev_estimates['snow_depth_inches'] > 5]
                    if len(high_snow_low_elev) > 0:
                        print(f"    ⚠ WARNING: {len(high_snow_low_elev):,} low-elevation rows (<3000 ft) have >5 inches snow")
                        print(f"      This indicates HARDCODED ELEVATION BUG - should use actual DEM elevation")
                        print(f"      Expected: 0 inches for elevations < 6000 ft (winter threshold)")
                        # Show examples
                        example = high_snow_low_elev.head(3)
                        for idx, row in example.iterrows():
                            print(f"      Example: elev={row['elevation']:.1f}ft, snow={row['snow_depth_inches']:.1f}in (should be ~0)")
                    else:
                        print(f"    ✓ Low elevations correctly have minimal snow (DEM formula working)")
                
                # Check for suspicious patterns (all same value indicates hardcoded bug)
                unique_depths = estimated['snow_depth_inches'].nunique()
                if unique_depths == 1 and len(estimated) > 5:
                    constant_depth = estimated['snow_depth_inches'].iloc[0]
                    if constant_depth > 0:
                        print(f"    ⚠ WARNING: All {len(estimated):,} estimates have same snow depth ({constant_depth:.1f} inches)")
                        print(f"      This indicates HARDCODED ELEVATION BUG - estimates should vary by location elevation")
                        print(f"      DEM formula should produce different values based on actual elevation")
                    else:
                        # All zeros is okay if all locations are low elevation
                        elev_range = f"{estimated['elevation'].min():.0f}-{estimated['elevation'].max():.0f}"
                        print(f"    Note: All estimates are 0 inches (locations at {elev_range} ft - may be correct if all < 6000 ft)")
                
                # Check that estimates vary by elevation (when multiple elevations present)
                if unique_depths > 1:
                    elev_range = estimated['elevation'].max() - estimated['elevation'].min()
                    if elev_range > 2000:  # Significant elevation range
                        # Check correlation between elevation and snow depth
                        correlation = estimated[['elevation', 'snow_depth_inches']].corr().iloc[0, 1]
                        if correlation < 0.5 and len(estimated) > 10:
                            print(f"    ⚠ Warning: Weak correlation ({correlation:.2f}) between elevation and snow depth")
                            print(f"      DEM formula should show positive correlation (higher elevation = more snow)")
                        else:
                            print(f"    ✓ Estimates vary appropriately with elevation (correlation: {correlation:.2f})")
                
                # Check if estimates are elevation-appropriate for high elevations
                high_elev_estimates = estimated[(estimated['elevation'] > 8000) & (estimated['snow_depth_inches'].notna())]
                if len(high_elev_estimates) > 0:
                    avg_snow = high_elev_estimates['snow_depth_inches'].mean()
                    # For 8000+ ft in winter: (8000 - 6000) / 100 = 20+ inches expected
                    if avg_snow < 10:
                        print(f"    ⚠ Warning: High-elevation estimates average only {avg_snow:.1f} inches")
                        print(f"      Expected higher values for elevations > 8000 ft (DEM formula)")
                
                # Verify formula behavior: elevations < 6000 ft should have 0 snow (winter)
                winter_estimates = estimated[estimated['snow_depth_inches'].notna()].copy()
                if len(winter_estimates) > 0:
                    # Check if we have date info to filter by season (if available)
                    below_6000 = winter_estimates[winter_estimates['elevation'] < 6000]
                    if len(below_6000) > 0:
                        non_zero = below_6000[below_6000['snow_depth_inches'] > 0.1]
                        if len(non_zero) > 0:
                            print(f"    ⚠ Warning: {len(non_zero):,} rows with elevation < 6000 ft have snow > 0.1 inches")
                            print(f"      DEM formula should give 0 inches for elevations < 6000 ft in winter")
                            print(f"      This may indicate hardcoded elevation bug or incorrect formula")
    else:
        print("\n⚠ 'snow_data_source' column not found - SNOTEL data quality cannot be assessed")
    
    # Validate SNOTEL values
    if 'snow_depth_inches' in df.columns:
        snow_depth = df['snow_depth_inches'].dropna()
        if len(snow_depth) > 0:
            print(f"\nSnow Depth Validation:")
            print(f"  Range: {snow_depth.min():.1f} to {snow_depth.max():.1f} inches")
            
            # Check for suspicious values
            if snow_depth.min() < 0:
                negative = (df['snow_depth_inches'] < 0).sum()
                print(f"  ⚠ Warning: {negative} rows have negative snow depth")
            
            if snow_depth.max() > 300:
                very_high = (df['snow_depth_inches'] > 300).sum()
                print(f"  ⚠ Warning: {very_high} rows have snow depth >300 inches (unusually high)")
    
    if 'snow_water_equiv_inches' in df.columns:
        swe = df['snow_water_equiv_inches'].dropna()
        if len(swe) > 0:
            print(f"\nSnow Water Equivalent (SWE) Validation:")
            print(f"  Range: {swe.min():.1f} to {swe.max():.1f} inches")
            
            # Check for suspicious values
            if swe.min() < 0:
                negative = (df['snow_water_equiv_inches'] < 0).sum()
                print(f"  ⚠ Warning: {negative} rows have negative SWE")
            
            # SWE should generally be less than snow depth (density < 1.0)
            if 'snow_depth_inches' in df.columns:
                both_present = df[df['snow_depth_inches'].notna() & df['snow_water_equiv_inches'].notna()]
                if len(both_present) > 0:
                    invalid_density = both_present[both_present['snow_water_equiv_inches'] > both_present['snow_depth_inches'] * 1.2]
                    if len(invalid_density) > 0:
                        print(f"  ⚠ Warning: {len(invalid_density)} rows have SWE >120% of snow depth (unrealistic density)")
    
    print(f"\n{'='*70}")
    print("GEOGRAPHIC VALIDATION")
    print(f"{'='*70}")
    
    # Check geographic bounds
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].dropna()
        lon = df['longitude'].dropna()
        both_present = df[df['latitude'].notna() & df['longitude'].notna()]
        
        if len(both_present) > 0:
            print(f"\nGeographic Bounds:")
            print(f"  Latitude range: {lat.min():.4f} to {lat.max():.4f}")
            print(f"  Longitude range: {lon.min():.4f} to {lon.max():.4f}")
            
            # Check if within Wyoming bounds
            wyoming_lat_bounds = EXPECTED_RANGES['latitude']
            wyoming_lon_bounds = EXPECTED_RANGES['longitude']
            
            within_lat = ((both_present['latitude'] >= wyoming_lat_bounds[0]) & 
                         (both_present['latitude'] <= wyoming_lat_bounds[1])).sum()
            within_lon = ((both_present['longitude'] >= wyoming_lon_bounds[0]) & 
                         (both_present['longitude'] <= wyoming_lon_bounds[1])).sum()
            within_bounds = ((both_present['latitude'] >= wyoming_lat_bounds[0]) &
                            (both_present['latitude'] <= wyoming_lat_bounds[1]) &
                            (both_present['longitude'] >= wyoming_lon_bounds[0]) &
                            (both_present['longitude'] <= wyoming_lon_bounds[1])).sum()
            
            total = len(both_present)
            print(f"\n  Within Wyoming bounds:")
            print(f"    Latitude: {within_lat:,}/{total:,} ({within_lat/total*100:.1f}%)")
            print(f"    Longitude: {within_lon:,}/{total:,} ({within_lon/total*100:.1f}%)")
            print(f"    Both (within Wyoming): {within_bounds:,}/{total:,} ({within_bounds/total*100:.1f}%)")
            
            if within_bounds < total:
                outside = total - within_bounds
                print(f"\n  ⚠ Warning: {outside:,} rows ({outside/total*100:.1f}%) are outside Wyoming bounds")
        else:
            print(f"\n⚠ No geographic data available")
    else:
        print(f"\n⚠ Latitude/longitude columns not found - cannot validate geographic bounds")
    
    print(f"\n{'='*70}")
    print("EXPECTED VALUE RANGES VALIDATION")
    print(f"{'='*70}")
    
    range_violations = {}
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                below_min = (values < min_val).sum()
                above_max = (values > max_val).sum()
                total = len(values)
                
                if below_min > 0 or above_max > 0:
                    range_violations[col] = {
                        'below_min': below_min,
                        'above_max': above_max,
                        'min_val': min_val,
                        'max_val': max_val,
                        'total': total,
                        'actual_min': values.min(),
                        'actual_max': values.max()
                    }
    
    if range_violations:
        print(f"\n⚠ Value Range Violations:")
        for col, violation in range_violations.items():
            print(f"\n  {col}:")
            print(f"    Expected range: {violation['min_val']} to {violation['max_val']}")
            print(f"    Actual range: {violation['actual_min']:.2f} to {violation['actual_max']:.2f}")
            if violation['below_min'] > 0:
                pct = (violation['below_min'] / violation['total']) * 100
                print(f"    ⚠ {violation['below_min']:,} values ({pct:.1f}%) below minimum")
            if violation['above_max'] > 0:
                pct = (violation['above_max'] / violation['total']) * 100
                print(f"    ⚠ {violation['above_max']:,} values ({pct:.1f}%) above maximum")
    else:
        print(f"\n✓ All numeric features within expected ranges")
    
    print(f"\n{'='*70}")
    print("LAND COVER VALIDATION")
    print(f"{'='*70}")
    
    # Validate NLCD land cover codes
    if 'land_cover_code' in df.columns:
        lc_codes = df['land_cover_code'].dropna()
        if len(lc_codes) > 0:
            invalid_codes = lc_codes[~lc_codes.isin(VALID_NLCD_CODES)]
            invalid_count = len(invalid_codes)
            total = len(lc_codes)
            
            if invalid_count > 0:
                print(f"\n⚠ Invalid NLCD Land Cover Codes:")
                print(f"  Invalid codes: {invalid_count:,}/{total:,} ({invalid_count/total*100:.1f}%)")
                invalid_unique = invalid_codes.unique()
                print(f"  Unique invalid codes: {sorted(invalid_unique)}")
                print(f"  Valid NLCD codes: {sorted(VALID_NLCD_CODES)}")
            else:
                print(f"\n✓ All land cover codes are valid NLCD codes")
            
            # Show distribution of land cover codes
            if total > 0:
                print(f"\n  Land Cover Code Distribution:")
                code_counts = lc_codes.value_counts().head(10)
                for code, count in code_counts.items():
                    pct = (count / total) * 100
                    code_status = "✓" if code in VALID_NLCD_CODES else "✗"
                    print(f"    {code_status} {code}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\n⚠ 'land_cover_code' column not found")
    
    print(f"\n{'='*70}")
    print("STATISTICAL OUTLIER DETECTION")
    print(f"{'='*70}")
    
    # Detect outliers using IQR method for key numeric features
    outlier_features = ['elevation', 'slope_degrees', 'water_distance_miles', 
                       'road_distance_miles', 'trail_distance_miles', 
                       'canopy_cover_percent', 'snow_depth_inches']
    
    outlier_summary = {}
    for col in outlier_features:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 10:  # Need enough data for outlier detection
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_pct = (outlier_count / len(values)) * 100
                    outlier_summary[col] = {
                        'count': outlier_count,
                        'pct': outlier_pct,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'min_outlier': outliers.min() if len(outliers) > 0 else None,
                        'max_outlier': outliers.max() if len(outliers) > 0 else None
                    }
    
    if outlier_summary:
        print(f"\n⚠ Statistical Outliers Detected (IQR method):")
        for col, summary in outlier_summary.items():
            print(f"\n  {col}:")
            print(f"    Outliers: {summary['count']:,} ({summary['pct']:.1f}%)")
            print(f"    Normal range: {summary['lower_bound']:.2f} to {summary['upper_bound']:.2f}")
            if summary['min_outlier'] is not None:
                print(f"    Outlier range: {summary['min_outlier']:.2f} to {summary['max_outlier']:.2f}")
    else:
        print(f"\n✓ No significant statistical outliers detected")
    
    print(f"\n{'='*70}")
    print("FEATURE RELATIONSHIP VALIDATION")
    print(f"{'='*70}")
    
    # Check feature relationships
    relationship_issues = []
    
    # 1. Land cover vs Canopy cover consistency
    if 'land_cover_code' in df.columns and 'canopy_cover_percent' in df.columns:
        both_present = df[df['land_cover_code'].notna() & df['canopy_cover_percent'].notna()]
        if len(both_present) > 0:
            # Forest codes (41, 42, 43) should generally have canopy
            forest_codes = [41, 42, 43]  # Deciduous, Evergreen, Mixed Forest
            forest_rows = both_present[both_present['land_cover_code'].isin(forest_codes)]
            if len(forest_rows) > 0:
                no_canopy_forest = forest_rows[forest_rows['canopy_cover_percent'] < 10]
                if len(no_canopy_forest) > len(forest_rows) * 0.1:  # >10% have no canopy
                    pct = (len(no_canopy_forest) / len(forest_rows)) * 100
                    relationship_issues.append(
                        f"⚠ {len(no_canopy_forest):,} forest land cover rows ({pct:.1f}%) have <10% canopy cover"
                    )
            
            # Non-forest codes shouldn't have high canopy
            non_forest = both_present[~both_present['land_cover_code'].isin(forest_codes + [52])]  # Exclude shrubland
            if len(non_forest) > 0:
                high_canopy_non_forest = non_forest[non_forest['canopy_cover_percent'] > 50]
                if len(high_canopy_non_forest) > len(non_forest) * 0.1:  # >10% have high canopy
                    pct = (len(high_canopy_non_forest) / len(non_forest)) * 100
                    relationship_issues.append(
                        f"⚠ {len(high_canopy_non_forest):,} non-forest rows ({pct:.1f}%) have >50% canopy cover"
                    )
    
    # 2. Elevation vs Temperature (basic sanity check - higher elevation should be colder)
    if 'elevation' in df.columns and 'temperature_f' in df.columns:
        both_present = df[df['elevation'].notna() & df['temperature_f'].notna()]
        if len(both_present) > 50:  # Need enough data
            correlation = both_present[['elevation', 'temperature_f']].corr().iloc[0, 1]
            # Expected negative correlation (higher elevation = colder)
            if correlation > 0.3:  # Strong positive correlation would be suspicious
                relationship_issues.append(
                    f"⚠ Unexpected positive correlation ({correlation:.2f}) between elevation and temperature"
                )
    
    # 3. NDVI vs Season (should vary seasonally)
    if 'ndvi' in df.columns and ('timestamp' in df.columns or 'date' in df.columns):
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        both_present = df[df['ndvi'].notna() & df[date_col].notna()].copy()
        if len(both_present) > 100:
            try:
                both_present['month'] = pd.to_datetime(both_present[date_col]).dt.month
                summer_ndvi = both_present[both_present['month'].isin([6, 7, 8])]['ndvi'].mean()
                winter_ndvi = both_present[both_present['month'].isin([12, 1, 2])]['ndvi'].mean()
                if summer_ndvi < winter_ndvi + 0.05:  # Summer should be noticeably higher
                    relationship_issues.append(
                        f"⚠ NDVI seasonal variation missing: Summer ({summer_ndvi:.3f}) not significantly higher than winter ({winter_ndvi:.3f})"
                    )
            except Exception:
                pass
    
    # 4. Temperature vs Season (should vary seasonally)
    if 'temperature_f' in df.columns and ('timestamp' in df.columns or 'date' in df.columns):
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        both_present = df[df['temperature_f'].notna() & df[date_col].notna()].copy()
        if len(both_present) > 100:
            try:
                both_present['month'] = pd.to_datetime(both_present[date_col]).dt.month
                summer_temp = both_present[both_present['month'].isin([6, 7, 8])]['temperature_f'].mean()
                winter_temp = both_present[both_present['month'].isin([12, 1, 2])]['temperature_f'].mean()
                temp_diff = summer_temp - winter_temp
                if temp_diff < 15:  # Should have significant seasonal variation
                    relationship_issues.append(
                        f"⚠ Temperature seasonal variation weak: Summer-winter difference ({temp_diff:.1f}°F) is smaller than expected"
                    )
            except Exception:
                pass
    
    # 5. NDVI vs Land Cover (forest should have higher NDVI)
    if 'ndvi' in df.columns and 'land_cover_code' in df.columns:
        both_present = df[df['ndvi'].notna() & df['land_cover_code'].notna()]
        if len(both_present) > 50:
            forest_codes = [41, 42, 43]  # Deciduous, Evergreen, Mixed Forest
            forest_ndvi = both_present[both_present['land_cover_code'].isin(forest_codes)]['ndvi'].mean()
            non_forest_ndvi = both_present[~both_present['land_cover_code'].isin(forest_codes)]['ndvi'].mean()
            if forest_ndvi < non_forest_ndvi:  # Forest should generally have higher NDVI
                relationship_issues.append(
                    f"⚠ Forest land cover has lower NDVI ({forest_ndvi:.3f}) than non-forest ({non_forest_ndvi:.3f}) - unexpected"
                )
    
    # 6. Aspect validation (should be 0-360)
    if 'aspect_degrees' in df.columns:
        aspect = df['aspect_degrees'].dropna()
        if len(aspect) > 0:
            invalid_aspect = aspect[(aspect < 0) | (aspect >= 360)]
            if len(invalid_aspect) > 0:
                relationship_issues.append(
                    f"⚠ {len(invalid_aspect):,} rows have aspect outside valid range (0-360 degrees)"
                )
    
    # 7. Water reliability should be 0-1
    if 'water_reliability' in df.columns:
        reliability = df['water_reliability'].dropna()
        if len(reliability) > 0:
            invalid_reliability = reliability[(reliability < 0) | (reliability > 1)]
            if len(invalid_reliability) > 0:
                relationship_issues.append(
                    f"⚠ {len(invalid_reliability):,} rows have water_reliability outside valid range (0-1)"
                )
    
    # 8. Security habitat should be 0-100
    if 'security_habitat_percent' in df.columns:
        security = df['security_habitat_percent'].dropna()
        if len(security) > 0:
            invalid_security = security[(security < 0) | (security > 100)]
            if len(invalid_security) > 0:
                relationship_issues.append(
                    f"⚠ {len(invalid_security):,} rows have security_habitat_percent outside valid range (0-100)"
                )
    
    if relationship_issues:
        print(f"\n⚠ Feature Relationship Issues:")
        for issue in relationship_issues:
            print(f"  {issue}")
    else:
        print(f"\n✓ No feature relationship issues detected")
    
    print(f"\n{'='*70}")
    print("FEATURE COMPLETENESS")
    print(f"{'='*70}")
    
    # Check if all expected features are present
    missing_features = [col for col in env_columns if col not in df.columns]
    if missing_features:
        print(f"\n⚠ Missing Expected Features ({len(missing_features)}):")
        for col in sorted(missing_features):
            print(f"  - {col}")
    else:
        print(f"\n✓ All expected environmental features are present")
    
    # Check coverage of each feature
    print(f"\n  Feature Coverage:")
    for col in sorted(existing_env_cols):
        if col in df.columns:
            non_null = df[col].notna().sum()
            total = len(df)
            pct = (non_null / total) * 100
            status = "✓" if pct >= 90 else "⚠" if pct >= 50 else "✗"
            print(f"    {status} {col:30s}: {non_null:6,}/{total:6,} ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("DATA QUALITY WARNINGS")
    print(f"{'='*70}")
    
    warnings = []
    
    # Check for placeholder values
    if placeholder_found:
        warnings.append("⚠ Placeholder values detected - re-run integration script to replace")
        
        # Add specific warnings for temporal features
        if 'ndvi' in df.columns:
            placeholder_ndvi = (abs(df['ndvi'] - 0.5) < 0.01).sum()
            if placeholder_ndvi > len(df) * 0.1:
                warnings.append(f"⚠ {placeholder_ndvi:,} rows ({placeholder_ndvi/len(df)*100:.1f}%) have placeholder NDVI (0.5)")
                warnings.append("  → Set APPEEARS_USERNAME and APPEEARS_PASSWORD environment variables to fetch real NDVI data")
        
        if 'temperature_f' in df.columns:
            placeholder_temp = (abs(df['temperature_f'] - 45.0) < 0.1).sum()
            if placeholder_temp > len(df) * 0.1:
                warnings.append(f"⚠ {placeholder_temp:,} rows ({placeholder_temp/len(df)*100:.1f}%) have placeholder temperature (45°F)")
                warnings.append("  → Weather data should be automatically fetched from PRISM (historical) or Open-Meteo (forecasts)")
        
        if 'precip_last_7_days_inches' in df.columns:
            placeholder_precip = (df['precip_last_7_days_inches'] == 0.0).sum()
            if placeholder_precip > len(df) * 0.7:
                warnings.append(f"⚠ {placeholder_precip:,} rows ({placeholder_precip/len(df)*100:.1f}%) have zero precipitation (may indicate placeholder data)")
    
    # Check for missing values (only unexpected ones)
    if unexpected_missing_found:
        warnings.append("⚠ Unexpected missing values detected in environmental features")
    
    # SNOTEL-specific warnings
    if 'snow_data_source' in df.columns:
        estimated_pct = (df['snow_data_source'] == 'estimate').sum() / len(df) * 100
        if estimated_pct > 50:
            warnings.append(f"⚠ {estimated_pct:.1f}% of rows use estimated SNOTEL data (consider expanding station coverage)")
        elif estimated_pct > 25:
            warnings.append(f"⚠ {estimated_pct:.1f}% of rows use estimated SNOTEL data")
        
        # Check for elevation-based estimate issues (DEM formula validation)
        estimated = df[df['snow_data_source'] == 'estimate']
        if len(estimated) > 0 and 'elevation' in estimated.columns and 'snow_depth_inches' in estimated.columns:
            # Check for hardcoded elevation bug (low elevations with high snow)
            low_elev_high_snow = estimated[(estimated['elevation'] < 3000) & (estimated['snow_depth_inches'] > 5)]
            if len(low_elev_high_snow) > 0:
                warnings.append(f"⚠ CRITICAL: {len(low_elev_high_snow):,} low-elevation rows (<3000 ft) have >5 inches snow")
                warnings.append("  → HARDCODED ELEVATION BUG DETECTED - DEM formula should use actual elevation, not hardcoded 8500 ft")
            
            # Check for constant values (indicates hardcoded bug)
            if estimated['snow_depth_inches'].nunique() == 1 and len(estimated) > 5:
                constant_val = estimated['snow_depth_inches'].iloc[0]
                if constant_val > 0:
                    warnings.append(f"⚠ CRITICAL: All {len(estimated):,} elevation-based estimates have same value ({constant_val:.1f} inches)")
                    warnings.append("  → HARDCODED ELEVATION BUG DETECTED - estimates should vary by actual location elevation")
            
            # Check for elevation correlation (DEM formula should show positive correlation)
            if len(estimated) > 10:
                elev_range = estimated['elevation'].max() - estimated['elevation'].min()
                if elev_range > 2000:  # Significant elevation range
                    correlation = estimated[['elevation', 'snow_depth_inches']].corr().iloc[0, 1]
                    if correlation < 0.3:
                        warnings.append(f"⚠ Weak correlation ({correlation:.2f}) between elevation and snow depth")
                        warnings.append("  → DEM formula should show positive correlation - verify elevation is being used correctly")
    
    # Infrastructure-specific warnings
    if 'road_distance_miles' in df.columns:
        road_dist = df['road_distance_miles'].dropna()
        if len(road_dist) > 0:
            default_count = (road_dist == 2.0).sum()
            if default_count > len(road_dist) * 0.1:  # >10% have default
                default_pct = (default_count / len(road_dist)) * 100
                warnings.append(f"⚠ {default_pct:.1f}% of rows have default road distance (2.0 mi) - roads dataset may not be loaded")
            
            if road_dist.max() > 100:
                very_far = (road_dist > 100).sum()
                warnings.append(f"⚠ {very_far} rows have road distance >100 miles (outside Wyoming bounds?)")
    
    if 'trail_distance_miles' in df.columns:
        trail_dist = df['trail_distance_miles'].dropna()
        if len(trail_dist) > 0:
            default_count = (trail_dist == 1.5).sum()
            if default_count > len(trail_dist) * 0.1:  # >10% have default
                default_pct = (default_count / len(trail_dist)) * 100
                warnings.append(f"⚠ {default_pct:.1f}% of rows have default trail distance (1.5 mi) - trails dataset may not be loaded")
            
            if trail_dist.max() > 100:
                very_far = (trail_dist > 100).sum()
                warnings.append(f"⚠ {very_far} rows have trail distance >100 miles (outside Wyoming bounds?)")
    
    # Add range violations to warnings
    if range_violations:
        for col, violation in range_violations.items():
            if violation['below_min'] > 0:
                pct = (violation['below_min'] / violation['total']) * 100
                warnings.append(f"⚠ {col}: {violation['below_min']:,} values ({pct:.1f}%) below expected minimum ({violation['min_val']})")
            if violation['above_max'] > 0:
                pct = (violation['above_max'] / violation['total']) * 100
                warnings.append(f"⚠ {col}: {violation['above_max']:,} values ({pct:.1f}%) above expected maximum ({violation['max_val']})")
    
    # Geographic warnings
    if 'latitude' in df.columns and 'longitude' in df.columns:
        both_present = df[df['latitude'].notna() & df['longitude'].notna()]
        if len(both_present) > 0:
            wyoming_lat_bounds = EXPECTED_RANGES['latitude']
            wyoming_lon_bounds = EXPECTED_RANGES['longitude']
            within_bounds = ((both_present['latitude'] >= wyoming_lat_bounds[0]) &
                            (both_present['latitude'] <= wyoming_lat_bounds[1]) &
                            (both_present['longitude'] >= wyoming_lon_bounds[0]) &
                            (both_present['longitude'] <= wyoming_lon_bounds[1])).sum()
            outside = len(both_present) - within_bounds
            if outside > 0:
                warnings.append(f"⚠ {outside:,} rows ({outside/len(both_present)*100:.1f}%) are outside Wyoming geographic bounds")
    
    # Land cover validation warnings
    if 'land_cover_code' in df.columns:
        lc_codes = df['land_cover_code'].dropna()
        if len(lc_codes) > 0:
            invalid_codes = lc_codes[~lc_codes.isin(VALID_NLCD_CODES)]
            if len(invalid_codes) > 0:
                warnings.append(f"⚠ {len(invalid_codes):,} rows have invalid NLCD land cover codes")
    
    # Outlier warnings (only flag if >5% of data are outliers)
    for col, summary in outlier_summary.items():
        if summary['pct'] > 5:
            warnings.append(f"⚠ {col}: {summary['pct']:.1f}% of values are statistical outliers")
    
    # Feature relationship warnings
    for issue in relationship_issues:
        warnings.append(issue)
    
    # Feature completeness warnings
    if missing_features:
        warnings.append(f"⚠ {len(missing_features)} expected environmental features are missing")
    
    if 'water_distance_miles' in df.columns:
        water_dist = df['water_distance_miles'].dropna()
        if len(water_dist) > 0 and water_dist.max() > 50:
            warnings.append("⚠ Some water distances exceed 50 miles (may indicate missing data)")
    
    if 'canopy_cover_percent' in df.columns:
        canopy = df['canopy_cover_percent'].dropna()
        if len(canopy) > 0:
            if canopy.min() < 0 or canopy.max() > 100:
                warnings.append("⚠ Canopy cover values outside valid range (0-100%)")
    
    if 'snow_depth_inches' in df.columns:
        snow_depth = df['snow_depth_inches'].dropna()
        if len(snow_depth) > 0:
            if snow_depth.min() < 0:
                warnings.append("⚠ Negative snow depth values detected")
            if snow_depth.max() > 300:
                warnings.append("⚠ Unusually high snow depth values (>300 inches) detected")
    
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
