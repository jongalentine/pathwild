#!/usr/bin/env python3
"""
Verify that water sources GeoJSON covers all of Wyoming.

This script analyzes the geographic extent of all features to ensure
complete coverage of Wyoming (41.0°N to 45.0°N, 111.0°W to 104.0°W).

Usage:
    python scripts/verify_wyoming_coverage.py data/hydrology/water_sources.geojson
"""

import argparse
import sys
import json
from pathlib import Path

# Wyoming bounds
WYOMING_BOUNDS = {
    'west': -111.0,
    'east': -104.0,
    'south': 41.0,
    'north': 45.0
}


def extract_coordinates(feature):
    """Extract all coordinates from a feature geometry."""
    coords = []
    geom = feature.get('geometry', {})
    geom_type = geom.get('type', '')
    coords_data = geom.get('coordinates', [])
    
    if geom_type == 'Point':
        if len(coords_data) >= 2:
            coords.append((coords_data[0], coords_data[1]))
    elif geom_type == 'LineString':
        for coord in coords_data:
            if len(coord) >= 2:
                coords.append((coord[0], coord[1]))
    elif geom_type == 'Polygon':
        for ring in coords_data:
            for coord in ring:
                if len(coord) >= 2:
                    coords.append((coord[0], coord[1]))
    elif geom_type == 'MultiPolygon':
        for polygon in coords_data:
            for ring in polygon:
                for coord in ring:
                    if len(coord) >= 2:
                        coords.append((coord[0], coord[1]))
    
    return coords


def verify_coverage(geojson_path: Path):
    """Verify geographic coverage of Wyoming."""
    
    print("="*60)
    print("VERIFYING WYOMING COVERAGE")
    print("="*60)
    print()
    print(f"File: {geojson_path}")
    print()
    
    if not geojson_path.exists():
        print(f"ERROR: File not found: {geojson_path}")
        return False
    
    print("Loading GeoJSON...")
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load file - {e}")
        return False
    
    features = data.get('features', [])
    if not features:
        print("ERROR: No features found")
        return False
    
    print(f"Total features: {len(features):,}")
    print()
    print("Extracting coordinates from all features...")
    print("(This may take a minute for large files)")
    
    # Extract all coordinates
    all_coords = []
    for i, feature in enumerate(features):
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1:,} features...")
        coords = extract_coordinates(feature)
        all_coords.extend(coords)
    
    if not all_coords:
        print("ERROR: No coordinates found")
        return False
    
    print(f"  ✓ Extracted {len(all_coords):,} coordinates")
    print()
    
    # Calculate bounds
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    
    data_bounds = {
        'west': min(lons),
        'east': max(lons),
        'south': min(lats),
        'north': max(lats)
    }
    
    print("--- GEOGRAPHIC BOUNDS ---")
    print(f"Data bounds:")
    print(f"  West:  {data_bounds['west']:.4f}° (Wyoming: {WYOMING_BOUNDS['west']:.4f}°)")
    print(f"  East:  {data_bounds['east']:.4f}° (Wyoming: {WYOMING_BOUNDS['east']:.4f}°)")
    print(f"  South: {data_bounds['south']:.4f}° (Wyoming: {WYOMING_BOUNDS['south']:.4f}°)")
    print(f"  North: {data_bounds['north']:.4f}° (Wyoming: {WYOMING_BOUNDS['north']:.4f}°)")
    print()
    
    # Check coverage
    print("--- COVERAGE ANALYSIS ---")
    
    coverage_issues = []
    
    # Check each boundary
    if data_bounds['west'] > WYOMING_BOUNDS['west']:
        gap = data_bounds['west'] - WYOMING_BOUNDS['west']
        coverage_issues.append(f"Missing western coverage: {gap:.4f}° ({gap * 111:.1f} km)")
    elif data_bounds['west'] < WYOMING_BOUNDS['west'] - 0.1:
        print(f"  ✓ West: Covers beyond Wyoming border ({data_bounds['west']:.4f}° < {WYOMING_BOUNDS['west']:.4f}°)")
    else:
        print(f"  ✓ West: Covers Wyoming border")
    
    if data_bounds['east'] < WYOMING_BOUNDS['east']:
        gap = WYOMING_BOUNDS['east'] - data_bounds['east']
        coverage_issues.append(f"Missing eastern coverage: {gap:.4f}° ({gap * 111:.1f} km)")
    elif data_bounds['east'] > WYOMING_BOUNDS['east'] + 0.1:
        print(f"  ✓ East: Covers beyond Wyoming border ({data_bounds['east']:.4f}° > {WYOMING_BOUNDS['east']:.4f}°)")
    else:
        print(f"  ✓ East: Covers Wyoming border")
    
    if data_bounds['south'] > WYOMING_BOUNDS['south']:
        gap = data_bounds['south'] - WYOMING_BOUNDS['south']
        coverage_issues.append(f"Missing southern coverage: {gap:.4f}° ({gap * 111:.1f} km)")
        print(f"  ✗ South: Missing {gap:.4f}° ({gap * 111:.1f} km) - data starts at {data_bounds['south']:.4f}°, Wyoming starts at {WYOMING_BOUNDS['south']:.4f}°")
    elif data_bounds['south'] < WYOMING_BOUNDS['south'] - 0.1:
        print(f"  ✓ South: Covers beyond Wyoming border ({data_bounds['south']:.4f}° < {WYOMING_BOUNDS['south']:.4f}°)")
    else:
        print(f"  ✓ South: Covers Wyoming border")
    
    if data_bounds['north'] < WYOMING_BOUNDS['north']:
        gap = WYOMING_BOUNDS['north'] - data_bounds['north']
        coverage_issues.append(f"Missing northern coverage: {gap:.4f}° ({gap * 111:.1f} km)")
    elif data_bounds['north'] > WYOMING_BOUNDS['north'] + 0.1:
        print(f"  ✓ North: Covers beyond Wyoming border ({data_bounds['north']:.4f}° > {WYOMING_BOUNDS['north']:.4f}°)")
    else:
        print(f"  ✓ North: Covers Wyoming border")
    
    print()
    
    # Calculate coverage percentage
    data_width = data_bounds['east'] - data_bounds['west']
    data_height = data_bounds['north'] - data_bounds['south']
    wyoming_width = WYOMING_BOUNDS['east'] - WYOMING_BOUNDS['west']
    wyoming_height = WYOMING_BOUNDS['north'] - WYOMING_BOUNDS['south']
    
    # Calculate overlap
    overlap_width = min(data_bounds['east'], WYOMING_BOUNDS['east']) - max(data_bounds['west'], WYOMING_BOUNDS['west'])
    overlap_height = min(data_bounds['north'], WYOMING_BOUNDS['north']) - max(data_bounds['south'], WYOMING_BOUNDS['south'])
    
    # Calculate areas (define before conditional to avoid NameError)
    wyoming_area = wyoming_width * wyoming_height
    data_area = data_width * data_height
    
    if overlap_width > 0 and overlap_height > 0:
        overlap_area = overlap_width * overlap_height
        coverage_pct = (overlap_area / wyoming_area) * 100
    else:
        overlap_area = 0.0
        coverage_pct = 0.0
    
    print("--- COVERAGE SUMMARY ---")
    print(f"Wyoming area: {wyoming_width:.4f}° × {wyoming_height:.4f}° = {wyoming_area:.4f} square degrees")
    print(f"Data area: {data_width:.4f}° × {data_height:.4f}° = {data_area:.4f} square degrees")
    print(f"Overlap area: {overlap_width:.4f}° × {overlap_height:.4f}° = {overlap_area:.4f} square degrees")
    print(f"Coverage: {coverage_pct:.1f}% of Wyoming")
    print()
    
    # Check for features in different regions
    print("--- REGIONAL DISTRIBUTION ---")
    
    # Divide Wyoming into quadrants
    mid_lon = (WYOMING_BOUNDS['west'] + WYOMING_BOUNDS['east']) / 2
    mid_lat = (WYOMING_BOUNDS['south'] + WYOMING_BOUNDS['north']) / 2
    
    regions = {
        'NW': (WYOMING_BOUNDS['west'], mid_lat, mid_lon, WYOMING_BOUNDS['north']),
        'NE': (mid_lon, mid_lat, WYOMING_BOUNDS['east'], WYOMING_BOUNDS['north']),
        'SW': (WYOMING_BOUNDS['west'], WYOMING_BOUNDS['south'], mid_lon, mid_lat),
        'SE': (mid_lon, WYOMING_BOUNDS['south'], WYOMING_BOUNDS['east'], mid_lat),
    }
    
    region_counts = {region: 0 for region in regions}
    
    for lon, lat in all_coords:
        for region, (w, s, e, n) in regions.items():
            if w <= lon <= e and s <= lat <= n:
                region_counts[region] += 1
                break
    
    print("Features by quadrant:")
    for region, count in region_counts.items():
        pct = (count / len(all_coords)) * 100 if all_coords else 0
        print(f"  {region}: {count:,} features ({pct:.1f}%)")
    
    print()
    
    # Final assessment
    print("="*60)
    print("ASSESSMENT")
    print("="*60)
    
    if coverage_issues:
        print("⚠ COVERAGE ISSUES FOUND:")
        for issue in coverage_issues:
            print(f"  - {issue}")
        print()
        print("Possible causes:")
        print("  1. Not all HU8 regions were downloaded")
        print("  2. Some HU8 regions don't have water features")
        print("  3. Data was clipped too aggressively")
        print()
        print("Recommendations:")
        if data_bounds['south'] > WYOMING_BOUNDS['south']:
            print("  - Check if southern Wyoming HU8 regions were downloaded")
            print("  - Verify all 102 zip files were processed")
        if data_bounds['east'] < WYOMING_BOUNDS['east']:
            print("  - Check if eastern Wyoming HU8 regions were downloaded")
        print("  - Re-run extraction if needed")
        print("  - Consider processing without clipping to see full extent")
    else:
        print("✓ COMPLETE COVERAGE!")
        print("  All of Wyoming is covered by the water sources data.")
    
    print()
    print(f"Coverage: {coverage_pct:.1f}% of Wyoming")
    
    return len(coverage_issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify water sources GeoJSON covers all of Wyoming'
    )
    parser.add_argument(
        'geojson_file',
        type=Path,
        help='Path to water sources GeoJSON file'
    )
    
    args = parser.parse_args()
    
    success = verify_coverage(args.geojson_file)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

