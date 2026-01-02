#!/usr/bin/env python3
"""
Remove placeholder elevation points and flag them for manual review.

This script:
1. Identifies points with placeholder elevation (8500.0m)
2. Saves them to a separate file for manual review
3. Removes them from the main dataset
4. Saves the cleaned dataset

Usage:
    python scripts/remove_placeholder_elevations.py [dataset_path] [--placeholder VALUE]
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def remove_placeholder_elevations(
    dataset_path: Path,
    placeholder_value: float = None,
    threshold_meters: float = None,
    threshold_feet: float = None,
    output_suffix: str = "_cleaned"
):
    """Remove placeholder elevation points and flag for review."""
    
    print("=" * 60)
    print("REMOVING PLACEHOLDER ELEVATIONS")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    
    # Determine removal criteria
    FEET_TO_METERS = 0.3048
    
    if threshold_feet is not None:
        threshold_meters = threshold_feet * FEET_TO_METERS
        print(f"Removing points above: {threshold_meters:.1f}m ({threshold_feet:.0f}ft)")
        use_threshold = True
    elif threshold_meters is not None:
        print(f"Removing points above: {threshold_meters:.1f}m ({threshold_meters / FEET_TO_METERS:.0f}ft)")
        use_threshold = True
    elif placeholder_value is not None:
        print(f"Removing points with exact value: {placeholder_value:.1f}m")
        use_threshold = False
    else:
        # Default: remove 8500.0m placeholders
        placeholder_value = 8500.0
        print(f"Removing points with exact value: {placeholder_value:.1f}m (default)")
        use_threshold = False
    
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
    
    # Identify points to remove
    if use_threshold:
        removal_mask = df['elevation'] > threshold_meters
        removal_reason = f'Elevation above {threshold_meters:.1f}m ({threshold_meters / FEET_TO_METERS:.0f}ft)'
    else:
        removal_mask = df['elevation'] == placeholder_value
        removal_reason = f'Placeholder elevation ({placeholder_value:.1f}m)'
    
    removal_count = removal_mask.sum()
    
    print(f"\n--- POINTS TO REMOVE ---")
    if use_threshold:
        print(f"Points above {threshold_meters:.1f}m ({threshold_meters / FEET_TO_METERS:.0f}ft): {removal_count:,}")
    else:
        print(f"Points with placeholder elevation: {removal_count:,}")
    print(f"Percentage of total: {removal_count/len(df)*100:.2f}%")
    
    if removal_count == 0:
        print(f"\n✓ No points found to remove!")
        return True
    
    # Extract points for review
    removal_df = df[removal_mask].copy()
    
    # Add metadata about why they were removed
    removal_df['removal_reason'] = removal_reason
    removal_df['removal_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    removal_df['original_dataset'] = dataset_path.name
    
    # Save removed points for manual review
    review_file = dataset_path.parent / f"{dataset_path.stem}_removed_for_review.csv"
    removal_df.to_csv(review_file, index=False)
    
    print(f"\n--- FLAGGED FOR REVIEW ---")
    print(f"✓ Saved {len(removal_df):,} points to: {review_file}")
    print(f"  This file contains all removed points for manual review")
    
    # Show summary of removed points
    print(f"\n--- REMOVED POINTS SUMMARY ---")
    if 'latitude' in removal_df.columns and 'longitude' in removal_df.columns:
        print(f"Latitude range: {removal_df['latitude'].min():.4f}° to {removal_df['latitude'].max():.4f}°")
        print(f"Longitude range: {removal_df['longitude'].min():.4f}° to {removal_df['longitude'].max():.4f}°")
    
    if 'elevation' in removal_df.columns:
        print(f"Elevation range: {removal_df['elevation'].min():.1f}m to {removal_df['elevation'].max():.1f}m")
        print(f"  ({removal_df['elevation'].min() * 3.28084:.0f}ft to {removal_df['elevation'].max() * 3.28084:.0f}ft)")
    
    if 'elk_present' in removal_df.columns:
        presence = removal_df[removal_df['elk_present'] == 1]
        absence = removal_df[removal_df['elk_present'] == 0]
        print(f"Presence points removed: {len(presence):,}")
        print(f"Absence points removed: {len(absence):,}")
    
    # Remove points from main dataset
    df_cleaned = df[~removal_mask].copy()
    
    print(f"\n--- CLEANED DATASET ---")
    print(f"Original rows: {len(df):,}")
    print(f"Removed rows: {removal_count:,}")
    print(f"Remaining rows: {len(df_cleaned):,}")
    print(f"Reduction: {removal_count/len(df)*100:.2f}%")
    
    # Verify no high elevations remain
    if use_threshold:
        remaining_high = (df_cleaned['elevation'] > threshold_meters).sum()
        if remaining_high > 0:
            print(f"\n⚠ WARNING: {remaining_high} points above threshold still remain!")
        else:
            print(f"\n✓ All points above threshold removed")
    else:
        remaining_placeholders = (df_cleaned['elevation'] == placeholder_value).sum()
        if remaining_placeholders > 0:
            print(f"\n⚠ WARNING: {remaining_placeholders} placeholder values still remain!")
        else:
            print(f"\n✓ All placeholder values removed")
    
    # Save cleaned dataset
    output_path = dataset_path.parent / f"{dataset_path.stem}{output_suffix}.csv"
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"\n--- OUTPUT FILES ---")
    print(f"✓ Cleaned dataset: {output_path}")
    print(f"  Rows: {len(df_cleaned):,}")
    print(f"✓ Removed points (for review): {review_file}")
    print(f"  Rows: {len(removal_df):,}")
    
    # Show elevation statistics of cleaned dataset
    print(f"\n--- CLEANED DATASET ELEVATION STATS ---")
    elev = df_cleaned['elevation'].dropna()
    if len(elev) > 0:
        print(f"Range: {elev.min():.1f}m to {elev.max():.1f}m")
        print(f"  ({elev.min() * 3.28084:.0f}ft to {elev.max() * 3.28084:.0f}ft)")
        print(f"Mean: {elev.mean():.1f}m ({elev.mean() * 3.28084:.0f}ft)")
        print(f"Median: {elev.median():.1f}m ({elev.median() * 3.28084:.0f}ft)")
        
        # Check if any are still above 13,800ft
        high_threshold = 13800 * 0.3048  # 13,800 feet in meters
        still_high = (elev > high_threshold).sum()
        if still_high > 0:
            print(f"\n  ⚠ {still_high} points are still above 13,800ft")
            print(f"    These may be real high peaks - verify manually")
        else:
            if use_threshold and threshold_meters >= high_threshold:
                print(f"\n  ✓ No points above 13,800ft (all removed)")
            else:
                print(f"\n  ✓ No points above 13,800ft")
    
    # Show class balance after removal
    if 'elk_present' in df_cleaned.columns:
        print(f"\n--- CLASS BALANCE (CLEANED) ---")
        presence = df_cleaned[df_cleaned['elk_present'] == 1]
        absence = df_cleaned[df_cleaned['elk_present'] == 0]
        print(f"Presence: {len(presence):,} ({len(presence)/len(df_cleaned)*100:.1f}%)")
        print(f"Absence: {len(absence):,} ({len(absence)/len(df_cleaned)*100:.1f}%)")
        
        if abs(len(presence) - len(absence)) > 10:
            print(f"  ⚠ Class imbalance: difference of {abs(len(presence) - len(absence))} points")
        else:
            print(f"  ✓ Classes are balanced")
    
    print(f"\n✓ Complete! Placeholder elevations removed and flagged for review.")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove placeholder elevation points and flag for review"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        default='data/processed/combined_north_bighorn_presence_absence.csv',
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--placeholder',
        type=float,
        default=None,
        help='Placeholder elevation value to remove (exact match)'
    )
    parser.add_argument(
        '--threshold-meters',
        type=float,
        default=None,
        help='Remove points above this elevation in meters'
    )
    parser.add_argument(
        '--threshold-feet',
        type=float,
        default=None,
        help='Remove points above this elevation in feet (e.g., 13800)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_cleaned',
        help='Suffix for output cleaned file (default: _cleaned)'
    )
    
    args = parser.parse_args()
    
    success = remove_placeholder_elevations(
        Path(args.dataset),
        placeholder_value=args.placeholder,
        threshold_meters=args.threshold_meters,
        threshold_feet=args.threshold_feet,
        output_suffix=args.output_suffix
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

