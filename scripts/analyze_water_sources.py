#!/usr/bin/env python3
"""
Analyze water sources GeoJSON file.

This script provides statistics and verification for water sources data:
- Feature counts and types
- Geographic coverage
- Water type distribution
- Reliability distribution
- CRS verification

Usage:
    python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson
"""

import argparse
import sys
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("ERROR: geopandas required")
    print("Install with: conda install geopandas")
    sys.exit(1)

# Wyoming bounding box (WGS84)
WYOMING_BOUNDS = box(-111.0, 41.0, -104.0, 45.0)


def analyze_water_sources(water_path: Path):
    """Analyze water sources GeoJSON file."""
    
    print(f"File: {water_path}")
    
    if not water_path.exists():
        print(f"ERROR: File not found: {water_path}")
        return False
    
    # Get file size
    file_size_mb = water_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB\n")
    
    # Load GeoDataFrame
    print("Loading water sources...")
    try:
        gdf = gpd.read_file(water_path)
    except Exception as e:
        print(f"ERROR: Failed to load file: {e}")
        return False
    
    print(f"Features: {len(gdf):,}\n")
    
    # --- METADATA ---
    print("--- METADATA ---")
    print(f"CRS: {gdf.crs}")
    
    # Get bounds
    bounds = gdf.total_bounds
    print(f"Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    print(f"  (West, South, East, North)")
    
    # Check if covers Wyoming
    wyoming_covered = (
        bounds[0] <= -111.0 and bounds[2] >= -104.0 and
        bounds[1] <= 41.0 and bounds[3] >= 45.0
    )
    if wyoming_covered:
        print("✓ File covers Wyoming")
    else:
        print("⚠ File may not fully cover Wyoming")
        print(f"  Wyoming bounds: [-111.0, 41.0, -104.0, 45.0]")
        print(f"  File bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    
    # Geometry types
    geom_types = gdf.geometry.type.value_counts()
    print(f"\nGeometry types:")
    for geom_type, count in geom_types.items():
        pct = (count / len(gdf)) * 100
        print(f"  {geom_type}: {count:,} ({pct:.1f}%)")
    
    # --- FEATURE TYPES ---
    if 'water_type' in gdf.columns:
        print("\n--- FEATURE TYPES ---")
        type_counts = gdf['water_type'].value_counts()
        for water_type, count in type_counts.items():
            pct = (count / len(gdf)) * 100
            print(f"{water_type}: {count:,} ({pct:.1f}%)")
    else:
        print("\n--- FEATURE TYPES ---")
        print("⚠ No 'water_type' column found")
    
    # --- RELIABILITY ---
    if 'reliability' in gdf.columns:
        print("\n--- RELIABILITY ---")
        reliability = gdf['reliability']
        print(f"Range: {reliability.min():.2f} to {reliability.max():.2f}")
        print(f"Mean: {reliability.mean():.2f}")
        print(f"Median: {reliability.median():.2f}")
        
        # Reliability categories
        high = (reliability >= 0.9).sum()
        medium = ((reliability >= 0.7) & (reliability < 0.9)).sum()
        low = (reliability < 0.7).sum()
        
        print(f"\nReliability categories:")
        print(f"  High (0.9-1.0): {high:,} ({high/len(gdf)*100:.1f}%)")
        print(f"  Medium (0.7-0.9): {medium:,} ({medium/len(gdf)*100:.1f}%)")
        print(f"  Low (0.4-0.7): {low:,} ({low/len(gdf)*100:.1f}%)")
    else:
        print("\n--- RELIABILITY ---")
        print("⚠ No 'reliability' column found")
    
    # --- NAMES ---
    if 'name' in gdf.columns:
        named_features = gdf['name'].notna().sum()
        print(f"\n--- NAMES ---")
        print(f"Features with names: {named_features:,} ({named_features/len(gdf)*100:.1f}%)")
        
        if named_features > 0:
            # Show some example names
            example_names = gdf[gdf['name'].notna()]['name'].head(10).tolist()
            print(f"\nExample names:")
            for name in example_names[:5]:
                print(f"  - {name}")
    else:
        print("\n--- NAMES ---")
        print("⚠ No 'name' column found")
    
    # --- VALIDATION ---
    print("\n--- VALIDATION ---")
    
    issues = []
    
    # Check CRS
    if gdf.crs != 'EPSG:4326':
        issues.append(f"CRS is {gdf.crs}, expected EPSG:4326 (WGS84)")
    else:
        print("✓ CRS is WGS84 (EPSG:4326)")
    
    # Check for required columns
    required_cols = ['water_type', 'reliability']
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print("✓ Required columns present")
    
    # Check feature count
    if len(gdf) < 1000:
        issues.append(f"Very few features ({len(gdf):,}), may be incomplete")
    else:
        print(f"✓ Feature count looks reasonable ({len(gdf):,})")
    
    # Check geometry validity
    invalid_geoms = ~gdf.geometry.is_valid
    if invalid_geoms.any():
        issues.append(f"Invalid geometries: {invalid_geoms.sum()}")
    else:
        print("✓ All geometries are valid")
    
    # Check for empty geometries
    empty_geoms = gdf.geometry.is_empty
    if empty_geoms.any():
        issues.append(f"Empty geometries: {empty_geoms.sum()}")
    else:
        print("✓ No empty geometries")
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All validations passed!")
    
    # --- SUMMARY ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if wyoming_covered and len(gdf) >= 1000 and gdf.crs == 'EPSG:4326':
        print("✓ File looks good for use in PathWild")
    else:
        print("⚠ File may have issues - review warnings above")
    
    print(f"\nTotal features: {len(gdf):,}")
    if 'water_type' in gdf.columns:
        print(f"Water types: {gdf['water_type'].nunique()}")
    if 'reliability' in gdf.columns:
        print(f"Reliability range: {gdf['reliability'].min():.2f} - {gdf['reliability'].max():.2f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Analyze water sources GeoJSON file'
    )
    parser.add_argument(
        'water_sources',
        type=Path,
        help='Path to water sources GeoJSON file'
    )
    
    args = parser.parse_args()
    
    success = analyze_water_sources(args.water_sources)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

