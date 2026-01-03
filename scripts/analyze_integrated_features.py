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
    print("DATA QUALITY WARNINGS")
    print(f"{'='*70}")
    
    warnings = []
    
    # Check for placeholder values
    if placeholder_found:
        warnings.append("⚠ Placeholder values detected - re-run integration script to replace")
    
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
