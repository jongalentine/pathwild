#!/usr/bin/env python3
"""
Analyze NLCD Tree Canopy Cover file to verify it's correct.

Usage:
    python scripts/analyze_canopy.py [canopy_file_path]
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


def analyze_canopy(canopy_path: Path):
    """Analyze NLCD Tree Canopy Cover file."""
    
    print("=" * 60)
    print("NLCD TREE CANOPY COVER ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {canopy_path}")
    
    if not canopy_path.exists():
        print(f"\n✗ ERROR: File not found: {canopy_path}")
        return False
    
    file_size_mb = canopy_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        with rasterio.open(canopy_path) as src:
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
            file_name = canopy_path.name.lower()
            is_wrong_product = any(keyword in file_name for keyword in ['landcover', 'land_cover', 'fctimp', 'impervious'])
            
            if is_wrong_product:
                print(f"\n⚠️  WARNING: This appears to be the WRONG NLCD product!")
                print(f"   File name suggests: Land Cover or Impervious (not Tree Canopy)")
                print(f"   You need: NLCD Tree Canopy Cover product")
                print(f"   Tree Canopy files should contain 'TreeCanopy' or 'Canopy' in the name")
            
            # Read and analyze canopy cover values
            print(f"\n--- VALUE STATISTICS ---")
            data = src.read(1)
            
            # Handle nodata
            if src.nodata is not None:
                valid_mask = data != src.nodata
                valid_data = data[valid_mask]
            else:
                valid_data = data.flatten()
            
            if len(valid_data) == 0:
                print(f"  ✗ ERROR: No valid data found!")
                return False
            
            min_val = float(valid_data.min())
            max_val = float(valid_data.max())
            mean_val = float(valid_data.mean())
            median_val = float(np.median(valid_data))
            
            print(f"Value range: {min_val} to {max_val}")
            print(f"Mean: {mean_val:.2f}%")
            print(f"Median: {median_val:.2f}%")
            
            # Check if values are in expected range (0-100)
            if min_val >= 0 and max_val <= 100:
                print(f"  ✓ Values are in expected range (0-100%)")
            else:
                # Count pixels above 100
                above_100 = valid_data[valid_data > 100]
                pct_above_100 = (len(above_100) / len(valid_data)) * 100
                
                print(f"\n⚠️  WARNING: Values outside expected range (0-100%)!")
                print(f"   Tree Canopy Cover should be percentages (0-100)")
                print(f"   Pixels with values > 100: {len(above_100):,} ({pct_above_100:.4f}%)")
                
                if len(above_100) > 0:
                    print(f"   Range of values > 100: {int(above_100.min())} to {int(above_100.max())}")
                    print(f"   Unique values > 100: {sorted(set(above_100.astype(int)))}")
                    print(f"   Note: These may be data errors. Consider filtering to 0-100 range when using the data.")
                
                if max_val > 100 and min_val < 200 and len(above_100) / len(valid_data) > 0.01:
                    print(f"   This might be Land Cover data (categorical codes) instead")
                    print(f"   Land Cover codes: 11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, etc.")
            
            # Percentiles
            percentiles = [25, 50, 75, 90, 95, 99]
            print(f"\nPercentiles:")
            for p in percentiles:
                val = float(np.percentile(valid_data, p))
                print(f"  {p:2d}th: {val:.1f}%")
            
            # Distribution analysis
            print(f"\n--- DISTRIBUTION ANALYSIS ---")
            
            # Count pixels in different canopy cover ranges
            ranges = [
                (0, 0, "No canopy (0%)"),
                (1, 10, "Sparse (1-10%)"),
                (11, 20, "Low (11-20%)"),
                (21, 40, "Moderate (21-40%)"),
                (41, 60, "High (41-60%)"),
                (61, 80, "Very High (61-80%)"),
                (81, 100, "Dense (81-100%)")
            ]
            
            total_pixels = len(valid_data)
            print(f"\nCanopy cover distribution:")
            for min_r, max_r, label in ranges:
                count = np.sum((valid_data >= min_r) & (valid_data <= max_r))
                percentage = (count / total_pixels) * 100
                print(f"  {label:20s}: {count:>12,} pixels ({percentage:5.2f}%)")
            
            # Check for realistic Wyoming distribution
            # Wyoming has relatively sparse tree cover - most areas should be 0-20%
            sparse_pct = (np.sum(valid_data <= 20) / total_pixels) * 100
            if sparse_pct > 70:
                print(f"\n  ✓ Distribution looks realistic for Wyoming (most areas sparse)")
            elif sparse_pct > 50:
                print(f"\n  ✓ Distribution reasonable for Wyoming")
            else:
                print(f"\n  ⚠ Distribution may not match Wyoming (expect mostly sparse areas)")
            
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
            print(f"✓ Total valid pixels: {len(valid_data):,}")
            
            if min_val >= 0 and max_val <= 100:
                print(f"✓ Values are percentages (0-100) as expected")
            else:
                print(f"✗ Values are NOT in expected range (0-100)")
            
            if covers_wyoming or (bounds.bottom <= wyoming_lat[1] and bounds.top >= wyoming_lat[0]):
                print(f"✓ Geographic coverage looks good")
            else:
                print(f"⚠ Geographic coverage may be insufficient")
            
            if 25 < pixel_size_m_x < 35:
                print(f"✓ Resolution looks correct")
            
            return True
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze NLCD Tree Canopy Cover file"
    )
    parser.add_argument(
        'canopy_file',
        nargs='?',
        default='data/canopy/canopy_cover.tif',
        help='Path to canopy cover file (default: data/canopy/canopy_cover.tif)'
    )
    
    args = parser.parse_args()
    canopy_path = Path(args.canopy_file)
    
    success = analyze_canopy(canopy_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

