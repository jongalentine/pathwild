#!/usr/bin/env python3
"""
Analyze NLCD land cover file to verify it's correct.

Usage:
    python scripts/analyze_nlcd.py [nlcd_file_path]
"""

import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("ERROR: rasterio not available")
    print("Install with: pip install rasterio or conda install rasterio")
    sys.exit(1)


# NLCD Land Cover Class Codes
NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}


def analyze_nlcd(nlcd_path: Path):
    """Analyze NLCD land cover file."""
    
    print("=" * 60)
    print("NLCD LAND COVER ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {nlcd_path}")
    
    if not nlcd_path.exists():
        print(f"\n✗ ERROR: File not found: {nlcd_path}")
        return False
    
    file_size_mb = nlcd_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        with rasterio.open(nlcd_path) as src:
            print(f"\n--- METADATA ---")
            print(f"Dimensions: {src.width:,} x {src.height:,} pixels")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"No data value: {src.nodata}")
            
            # Check if covers Wyoming
            bounds = src.bounds
            wyoming_lat = (41.0, 45.0)
            wyoming_lon = (-111.0, -104.0)
            
            print(f"\n--- GEOGRAPHIC COVERAGE ---")
            
            # Convert bounds to lat/lon if needed
            if src.crs and not src.crs.is_geographic:
                # Projected CRS - convert bounds to lat/lon
                from pyproj import Transformer
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                left_lon, bottom_lat = transformer.transform(bounds.left, bounds.bottom)
                right_lon, top_lat = transformer.transform(bounds.right, bounds.top)
                print(f"File bounds (converted to lat/lon):")
                print(f"  Latitude: {bottom_lat:.4f}° to {top_lat:.4f}°")
                print(f"  Longitude: {left_lon:.4f}° to {right_lon:.4f}°")
                print(f"\n  (Original projected bounds: {bounds})")
            else:
                # Already in lat/lon
                left_lon, right_lon = bounds.left, bounds.right
                bottom_lat, top_lat = bounds.bottom, bounds.top
                print(f"File bounds:")
                print(f"  Latitude: {bottom_lat:.4f}° to {top_lat:.4f}°")
                print(f"  Longitude: {left_lon:.4f}° to {right_lon:.4f}°")
            
            print(f"\nWyoming bounds:")
            print(f"  Latitude: {wyoming_lat[0]:.1f}° to {wyoming_lat[1]:.1f}°")
            print(f"  Longitude: {wyoming_lon[0]:.1f}° to {wyoming_lon[1]:.1f}°")
            
            covers_wyoming = (
                bottom_lat <= wyoming_lat[0] and top_lat >= wyoming_lat[1] and
                left_lon <= wyoming_lon[0] and right_lon >= wyoming_lon[1]
            )
            
            if covers_wyoming:
                print(f"\n  ✓ File covers all of Wyoming")
            elif (bottom_lat <= wyoming_lat[1] and top_lat >= wyoming_lat[0] and
                  left_lon <= wyoming_lon[1] and right_lon >= wyoming_lon[0]):
                print(f"\n  ⚠ File partially covers Wyoming (may need clipping)")
            else:
                print(f"\n  ✗ WARNING: File may not cover Wyoming!")
            
            # Check if this is the wrong product type
            file_name = nlcd_path.name.lower()
            is_wrong_product = any(keyword in file_name for keyword in ['fctimp', 'impervious', 'canopy', 'developed'])
            
            if is_wrong_product:
                print(f"\n⚠️  WARNING: This appears to be the WRONG NLCD product!")
                print(f"   File name suggests: {'Impervious' if 'fctimp' in file_name or 'impervious' in file_name else 'Canopy/Developed'}")
                print(f"   You need: NLCD Land Cover product")
                print(f"   See: docs/nlcd_wrong_product_fix.md for how to download the correct product")
            
            # Read and analyze land cover classes
            print(f"\n--- LAND COVER CLASSES ---")
            data = src.read(1)
            
            # Handle nodata
            if src.nodata is not None:
                valid_mask = data != src.nodata
                valid_data = data[valid_mask]
            else:
                valid_data = data.flatten()
            
            unique_classes = np.unique(valid_data)
            print(f"Unique land cover classes: {len(unique_classes)}")
            print(f"Classes present: {sorted(unique_classes)}")
            
            # Check if values look like percentages (0-100) instead of land cover codes
            max_value = np.max(valid_data)
            if max_value <= 100 and len(unique_classes) > 50:
                print(f"\n⚠️  WARNING: Values appear to be percentages (0-100), not land cover codes!")
                print(f"   This is likely the Fractional Impervious or Percent Canopy product, not Land Cover")
                print(f"   Land cover codes should be: 11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, etc.")
                print(f"   See: docs/nlcd_wrong_product_fix.md")
            
            # Count pixels per class
            print(f"\n--- CLASS DISTRIBUTION ---")
            class_counts = {}
            for class_code in unique_classes:
                count = np.sum(valid_data == class_code)
                percentage = (count / len(valid_data)) * 100
                class_name = NLCD_CLASSES.get(int(class_code), f"Unknown ({class_code})")
                class_counts[class_code] = {
                    'count': count,
                    'percentage': percentage,
                    'name': class_name
                }
            
            # Sort by count (descending)
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1]['count'], reverse=True)
            
            print(f"\nTop 10 land cover classes:")
            for i, (code, info) in enumerate(sorted_classes[:10], 1):
                print(f"  {i:2d}. {info['name']:30s} (Code {code:2d}): "
                      f"{info['count']:>12,} pixels ({info['percentage']:5.2f}%)")
            
            # Check for common Wyoming classes
            print(f"\n--- WYOMING-SPECIFIC CLASSES ---")
            wyoming_common = [52, 71, 41, 42, 43, 81, 82]  # Shrub, Grassland, Forest, Pasture, Crops
            found_common = [code for code in wyoming_common if code in unique_classes]
            
            if found_common:
                print(f"✓ Found common Wyoming classes:")
                for code in found_common:
                    info = class_counts[code]
                    print(f"  - {info['name']} (Code {code}): {info['percentage']:.1f}%")
            else:
                print(f"⚠ No common Wyoming classes found - may be wrong area")
            
            # Resolution check
            print(f"\n--- RESOLUTION ---")
            transform = src.transform
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            
            # Convert to meters (approximate)
            if src.crs and src.crs.is_geographic:
                # Geographic coordinates - convert to meters
                center_lat = (bounds.top + bounds.bottom) / 2
                meters_per_degree_lat = 111320.0
                meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
                pixel_size_m_x = pixel_size_x * meters_per_degree_lon
                pixel_size_m_y = pixel_size_y * meters_per_degree_lat
            else:
                # Already in meters (projected)
                pixel_size_m_x = pixel_size_x
                pixel_size_m_y = pixel_size_y
            
            print(f"Pixel size: {pixel_size_x:.6f} x {pixel_size_y:.6f} (native units)")
            print(f"Pixel size: ~{pixel_size_m_x:.0f}m x ~{pixel_size_m_y:.0f}m")
            
            if 25 < pixel_size_m_x < 35:
                print(f"  ✓ Resolution looks correct for NLCD (30m)")
            else:
                print(f"  ⚠ Resolution may not be standard NLCD (expected ~30m)")
            
            print(f"\n--- SUMMARY ---")
            print(f"✓ File is readable")
            print(f"✓ Contains {len(unique_classes)} land cover classes")
            print(f"✓ Total valid pixels: {len(valid_data):,}")
            
            if covers_wyoming or (bounds.bottom <= wyoming_lat[1] and bounds.top >= wyoming_lat[0]):
                print(f"✓ Geographic coverage looks good")
            else:
                print(f"⚠ Geographic coverage may be insufficient")
            
            return True
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze NLCD land cover file"
    )
    parser.add_argument(
        'nlcd_file',
        nargs='?',
        default='data/landcover/nlcd.tif',
        help='Path to NLCD file (default: data/landcover/nlcd.tif)'
    )
    
    args = parser.parse_args()
    nlcd_path = Path(args.nlcd_file)
    
    success = analyze_nlcd(nlcd_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

