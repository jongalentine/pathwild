#!/usr/bin/env python3
"""
Investigate elevation data points greater than 13,800 feet.

This script helps identify and analyze high elevation points that may be:
- Real high peaks
- Data errors
- Placeholder values that weren't replaced

Usage:
    python scripts/investigate_high_elevations.py [dataset_path] [--threshold-feet FEET]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Convert 13,800 feet to meters
FEET_TO_METERS = 0.3048
THRESHOLD_FEET = 13800
THRESHOLD_METERS = THRESHOLD_FEET * FEET_TO_METERS  # ~4,206.6 meters


def investigate_high_elevations(dataset_path: Path, threshold_meters: float = THRESHOLD_METERS):
    """Investigate elevation points above threshold."""
    
    print("=" * 60)
    print("HIGH ELEVATION INVESTIGATION")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Threshold: {threshold_meters:.1f} meters ({threshold_meters * 3.28084:.0f} feet)")
    
    if not dataset_path.exists():
        print(f"\n✗ ERROR: Dataset not found: {dataset_path}")
        return False
    
    # Load dataset
    print(f"\nLoading dataset...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"  ✓ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"  ✗ ERROR loading dataset: {e}")
        return False
    
    if 'elevation' not in df.columns:
        print(f"  ✗ ERROR: 'elevation' column not found")
        return False
    
    # Filter high elevation points
    high_elev = df[df['elevation'] > threshold_meters].copy()
    
    print(f"\n--- HIGH ELEVATION POINTS ---")
    print(f"Points above {threshold_meters:.1f}m ({threshold_meters * 3.28084:.0f}ft): {len(high_elev):,}")
    print(f"Percentage of total: {len(high_elev)/len(df)*100:.2f}%")
    
    if len(high_elev) == 0:
        print(f"\n✓ No points found above threshold!")
        return True
    
    # Analyze elevation distribution
    print(f"\n--- ELEVATION STATISTICS ---")
    print(f"Minimum: {high_elev['elevation'].min():.1f}m ({high_elev['elevation'].min() * 3.28084:.0f}ft)")
    print(f"Maximum: {high_elev['elevation'].max():.1f}m ({high_elev['elevation'].max() * 3.28084:.0f}ft)")
    print(f"Mean: {high_elev['elevation'].mean():.1f}m ({high_elev['elevation'].mean() * 3.28084:.0f}ft)")
    print(f"Median: {high_elev['elevation'].median():.1f}m ({high_elev['elevation'].median() * 3.28084:.0f}ft)")
    
    # Check for unique values (might indicate placeholder)
    unique_vals = high_elev['elevation'].nunique()
    print(f"\nUnique elevation values: {unique_vals}")
    
    if unique_vals == 1:
        print(f"  ⚠ WARNING: Only one unique value - likely placeholder!")
        print(f"  Value: {high_elev['elevation'].iloc[0]:.1f}m")
    elif unique_vals < 10:
        print(f"  ⚠ WARNING: Very few unique values ({unique_vals}) - may be placeholders")
        print(f"  Values: {sorted(high_elev['elevation'].unique())}")
    else:
        print(f"  ✓ Multiple unique values - likely real data")
    
    # Geographic distribution
    print(f"\n--- GEOGRAPHIC DISTRIBUTION ---")
    if 'latitude' in high_elev.columns and 'longitude' in high_elev.columns:
        print(f"Latitude range: {high_elev['latitude'].min():.4f}° to {high_elev['latitude'].max():.4f}°")
        print(f"Longitude range: {high_elev['longitude'].min():.4f}° to {high_elev['longitude'].max():.4f}°")
        
        # Check if they're in Wyoming
        wyoming_lat = (41.0, 45.0)
        wyoming_lon = (-111.0, -104.0)
        
        in_wyoming = (
            (high_elev['latitude'] >= wyoming_lat[0]) & 
            (high_elev['latitude'] <= wyoming_lat[1]) &
            (high_elev['longitude'] >= wyoming_lon[0]) & 
            (high_elev['longitude'] <= wyoming_lon[1])
        )
        
        print(f"Points in Wyoming bounds: {in_wyoming.sum():,} ({in_wyoming.sum()/len(high_elev)*100:.1f}%)")
        
        if in_wyoming.sum() < len(high_elev):
            print(f"  ⚠ WARNING: Some points are outside Wyoming bounds!")
            outside = high_elev[~in_wyoming]
            print(f"  Outside Wyoming: {len(outside):,} points")
            if len(outside) <= 10:
                print(f"  Locations:")
                for idx, row in outside.iterrows():
                    print(f"    {row['latitude']:.4f}°, {row['longitude']:.4f}° - {row['elevation']:.1f}m")
    
    # Presence vs Absence
    print(f"\n--- PRESENCE VS ABSENCE ---")
    if 'elk_present' in high_elev.columns:
        presence = high_elev[high_elev['elk_present'] == 1]
        absence = high_elev[high_elev['elk_present'] == 0]
        print(f"Presence points: {len(presence):,}")
        print(f"Absence points: {len(absence):,}")
        
        if len(presence) > 0:
            print(f"  Presence elevation range: {presence['elevation'].min():.1f}m to {presence['elevation'].max():.1f}m")
        if len(absence) > 0:
            print(f"  Absence elevation range: {absence['elevation'].min():.1f}m to {absence['elevation'].max():.1f}m")
    else:
        print(f"  ⚠ 'elk_present' column not found")
    
    # Check for specific suspicious values
    print(f"\n--- SUSPICIOUS VALUES ---")
    
    # Check for exactly 8500 (common placeholder)
    exactly_8500 = high_elev[high_elev['elevation'] == 8500.0]
    if len(exactly_8500) > 0:
        print(f"  ⚠ Found {len(exactly_8500):,} points with elevation exactly 8500.0m")
        print(f"    This is likely a placeholder value that wasn't replaced!")
        print(f"    Percentage of high points: {len(exactly_8500)/len(high_elev)*100:.1f}%")
    
    # Check for round numbers (might be placeholders)
    round_numbers = high_elev[high_elev['elevation'] % 100 == 0]
    if len(round_numbers) > len(exactly_8500):
        print(f"  ⚠ Found {len(round_numbers):,} points with round-number elevations (divisible by 100)")
        print(f"    This might indicate placeholder values")
    
    # Check for very high values (>5000m is extremely high for Wyoming)
    very_high = high_elev[high_elev['elevation'] > 5000]
    if len(very_high) > 0:
        print(f"  ⚠ Found {len(very_high):,} points above 5,000m (16,400ft)")
        print(f"    Wyoming's highest peak is ~4,207m (13,804ft) - these are likely errors!")
        if len(very_high) <= 20:
            print(f"    Values: {sorted(very_high['elevation'].unique())}")
    
    # Sample high elevation points
    print(f"\n--- SAMPLE HIGH ELEVATION POINTS ---")
    sample_size = min(10, len(high_elev))
    sample = high_elev.head(sample_size)
    
    for idx, row in sample.iterrows():
        elev_m = row['elevation']
        elev_ft = elev_m * 3.28084
        lat = row.get('latitude', 'N/A')
        lon = row.get('longitude', 'N/A')
        present = row.get('elk_present', 'N/A')
        
        print(f"  {elev_m:.1f}m ({elev_ft:.0f}ft) at {lat:.4f}°, {lon:.4f}° - Present: {present}")
    
    # Save detailed report
    output_file = dataset_path.parent / f"{dataset_path.stem}_high_elevations.csv"
    high_elev.to_csv(output_file, index=False)
    print(f"\n--- DETAILED REPORT ---")
    print(f"✓ Saved detailed report to: {output_file}")
    print(f"  Contains {len(high_elev):,} high elevation points with all columns")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    
    if len(exactly_8500) > 0:
        print(f"1. ⚠ Remove or replace {len(exactly_8500):,} points with elevation 8500.0m (placeholder)")
    
    if len(very_high) > 0:
        print(f"2. ⚠ Investigate {len(very_high):,} points above 5,000m - likely data errors")
        print(f"   Wyoming's highest point is Gannett Peak at 4,207m (13,804ft)")
    
    if unique_vals == 1:
        print(f"3. ⚠ All high points have the same value - definitely placeholders")
        print(f"   Consider removing these points or replacing with actual DEM values")
    
    if len(high_elev) > 0 and len(high_elev) < len(df) * 0.01:
        print(f"4. ✓ High points are rare ({len(high_elev)/len(df)*100:.2f}%) - likely real peaks")
        print(f"   Verify against known Wyoming peaks (Gannett Peak, Grand Teton, etc.)")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Investigate high elevation data points"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        default='data/processed/combined_north_bighorn_presence_absence.csv',
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--threshold-feet',
        type=float,
        default=THRESHOLD_FEET,
        help=f'Elevation threshold in feet (default: {THRESHOLD_FEET})'
    )
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    threshold_meters = args.threshold_feet * FEET_TO_METERS
    
    success = investigate_high_elevations(dataset_path, threshold_meters)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

