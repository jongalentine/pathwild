#!/usr/bin/env python3
"""
Fix placeholder elevation values (8500.0m) by sampling from DEM.

This script:
1. Identifies points with placeholder elevation (8500.0m)
2. Samples actual elevation from DEM for those points
3. Updates the dataset

Usage:
    python scripts/fix_placeholder_elevations.py [dataset_path] [--dem-file PATH]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("ERROR: rasterio not available")
    print("Install with: pip install rasterio or conda install rasterio")
    sys.exit(1)


def sample_elevation_from_dem(lat: float, lon: float, dem) -> float:
    """Sample elevation from DEM at given coordinates."""
    try:
        row, col = dem.index(lon, lat)
        window = rasterio.windows.Window(col, row, 1, 1)
        data = dem.read(1, window=window)
        value = float(data[0, 0])
        
        # Check for nodata
        if value == dem.nodata or np.isnan(value):
            return None
        
        return value
    except Exception:
        return None


def fix_placeholder_elevations(dataset_path: Path, dem_path: Path, placeholder_value: float = 8500.0):
    """Fix placeholder elevation values by sampling from DEM."""
    
    print("=" * 60)
    print("FIXING PLACEHOLDER ELEVATIONS")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"DEM: {dem_path}")
    print(f"Placeholder value: {placeholder_value:.1f}m")
    
    if not dataset_path.exists():
        print(f"\n✗ ERROR: Dataset not found: {dataset_path}")
        return False
    
    if not dem_path.exists():
        print(f"\n✗ ERROR: DEM file not found: {dem_path}")
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
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print(f"  ✗ ERROR: 'latitude' or 'longitude' columns not found")
        return False
    
    # Find placeholder points
    placeholder_mask = df['elevation'] == placeholder_value
    placeholder_count = placeholder_mask.sum()
    
    print(f"\n--- PLACEHOLDER POINTS ---")
    print(f"Points with placeholder elevation: {placeholder_count:,}")
    print(f"Percentage of total: {placeholder_count/len(df)*100:.2f}%")
    
    if placeholder_count == 0:
        print(f"\n✓ No placeholder values found!")
        return True
    
    # Open DEM
    print(f"\nOpening DEM...")
    try:
        dem = rasterio.open(dem_path)
        print(f"  ✓ DEM opened: {dem.width}x{dem.height}")
        print(f"  CRS: {dem.crs}")
        print(f"  Bounds: {dem.bounds}")
    except Exception as e:
        print(f"  ✗ ERROR opening DEM: {e}")
        return False
    
    # Sample elevations for placeholder points
    print(f"\nSampling elevations from DEM...")
    placeholder_df = df[placeholder_mask].copy()
    
    fixed_count = 0
    failed_count = 0
    
    for idx, row in placeholder_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        elev = sample_elevation_from_dem(lat, lon, dem)
        
        if elev is not None:
            df.at[idx, 'elevation'] = elev
            fixed_count += 1
        else:
            failed_count += 1
        
        if (fixed_count + failed_count) % 50 == 0:
            print(f"  Processed {fixed_count + failed_count}/{placeholder_count}...")
    
    dem.close()
    
    print(f"\n--- RESULTS ---")
    print(f"Successfully fixed: {fixed_count:,} points")
    print(f"Failed to sample: {failed_count:,} points")
    
    if failed_count > 0:
        print(f"\n  ⚠ {failed_count} points couldn't be sampled from DEM")
        print(f"    Possible reasons:")
        print(f"    - Points outside DEM bounds")
        print(f"    - DEM has NoData at those locations")
        print(f"    - Coordinate system mismatch")
    
    # Show statistics of fixed values
    if fixed_count > 0:
        fixed_elevations = df.loc[placeholder_mask & (df['elevation'] != placeholder_value), 'elevation']
        print(f"\n--- FIXED ELEVATION STATISTICS ---")
        print(f"Range: {fixed_elevations.min():.1f}m to {fixed_elevations.max():.1f}m")
        print(f"Mean: {fixed_elevations.mean():.1f}m")
        print(f"Median: {fixed_elevations.median():.1f}m")
        
        # Check if any are still above 13,800ft
        high_threshold = 13800 * 0.3048  # 13,800 feet in meters
        still_high = (fixed_elevations > high_threshold).sum()
        if still_high > 0:
            print(f"\n  ⚠ {still_high} fixed points are still above 13,800ft")
            print(f"    These may be real high peaks or still need investigation")
    
    # Save updated dataset
    output_path = dataset_path.parent / f"{dataset_path.stem}_fixed.csv"
    df.to_csv(output_path, index=False)
    print(f"\n--- OUTPUT ---")
    print(f"✓ Saved updated dataset to: {output_path}")
    print(f"  Original: {dataset_path}")
    print(f"  Updated: {output_path}")
    
    # Verify no more placeholders
    remaining_placeholders = (df['elevation'] == placeholder_value).sum()
    if remaining_placeholders == 0:
        print(f"\n✓ All placeholder values have been replaced!")
    else:
        print(f"\n⚠ {remaining_placeholders} placeholder values remain")
        print(f"   These points couldn't be sampled from DEM")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix placeholder elevation values by sampling from DEM"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        default='data/processed/combined_north_bighorn_presence_absence.csv',
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--dem-file',
        type=Path,
        default=Path('data/dem/wyoming_dem.tif'),
        help='Path to DEM file (default: data/dem/wyoming_dem.tif)'
    )
    parser.add_argument(
        '--placeholder',
        type=float,
        default=8500.0,
        help='Placeholder elevation value to replace (default: 8500.0)'
    )
    
    args = parser.parse_args()
    
    success = fix_placeholder_elevations(
        Path(args.dataset),
        args.dem_file,
        args.placeholder
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

