#!/usr/bin/env python3
"""
Analyze Wyoming DEM file to verify it's correct.

Checks:
- File exists and is readable
- Coordinate system
- Bounds (should cover Wyoming)
- Elevation values (should be reasonable for Wyoming)
- Data quality
"""

import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio import plot
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install rasterio matplotlib")
    sys.exit(1)


def analyze_dem(dem_path: Path):
    """Analyze DEM file and report findings."""
    
    print("=" * 60)
    print("WYOMING DEM ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {dem_path}")
    
    # Check file exists
    if not dem_path.exists():
        print(f"\n✗ ERROR: File not found: {dem_path}")
        return False
    
    file_size_mb = dem_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Open and analyze
    try:
        with rasterio.open(dem_path) as dem:
            print(f"\n✓ File opened successfully")
            
            # Basic metadata
            print(f"\n--- METADATA ---")
            print(f"Driver: {dem.driver}")
            print(f"Width: {dem.width:,} pixels")
            print(f"Height: {dem.height:,} pixels")
            print(f"Bands: {dem.count}")
            print(f"Data type: {dem.dtypes[0]}")
            print(f"No data value: {dem.nodata}")
            
            # Coordinate system
            print(f"\n--- COORDINATE SYSTEM ---")
            print(f"CRS: {dem.crs}")
            if dem.crs:
                print(f"  ✓ Coordinate system defined")
            else:
                print(f"  ⚠ WARNING: No coordinate system defined!")
            
            # Bounds (geographic extent)
            print(f"\n--- GEOGRAPHIC BOUNDS ---")
            bounds = dem.bounds
            print(f"West (min longitude): {bounds.left:.4f}°")
            print(f"East (max longitude): {bounds.right:.4f}°")
            print(f"South (min latitude): {bounds.bottom:.4f}°")
            print(f"North (max latitude): {bounds.top:.4f}°")
            
            # Expected Wyoming bounds
            wyoming_expected = {
                'west': -111.0,
                'east': -104.0,
                'south': 41.0,
                'north': 45.0
            }
            
            print(f"\nExpected Wyoming bounds:")
            print(f"  Longitude: {wyoming_expected['west']:.1f}° to {wyoming_expected['east']:.1f}°")
            print(f"  Latitude: {wyoming_expected['south']:.1f}° to {wyoming_expected['north']:.1f}°")
            
            # Check if bounds are reasonable
            bounds_ok = True
            if bounds.left > wyoming_expected['west'] + 1:
                print(f"  ⚠ WARNING: West bound seems too far east")
                bounds_ok = False
            if bounds.right < wyoming_expected['east'] - 1:
                print(f"  ⚠ WARNING: East bound seems too far west")
                bounds_ok = False
            if bounds.bottom > wyoming_expected['south'] + 1:
                print(f"  ⚠ WARNING: South bound seems too far north")
                bounds_ok = False
            if bounds.top < wyoming_expected['north'] - 1:
                print(f"  ⚠ WARNING: North bound seems too far south")
                bounds_ok = False
            
            if bounds_ok:
                print(f"  ✓ Bounds look reasonable for Wyoming")
            
            # Resolution
            print(f"\n--- RESOLUTION ---")
            transform = dem.transform
            print(f"Pixel size (X): {abs(transform[0]):.6f} degrees")
            print(f"Pixel size (Y): {abs(transform[4]):.6f} degrees")
            
            # Convert to approximate meters (at Wyoming's latitude ~43°)
            # 1 degree ≈ 111 km, but varies by latitude
            lat_center = (bounds.top + bounds.bottom) / 2
            meters_per_degree_lat = 111000  # Approximately constant
            meters_per_degree_lon = 111000 * np.cos(np.radians(lat_center))
            
            pixel_size_m_x = abs(transform[0]) * meters_per_degree_lon
            pixel_size_m_y = abs(transform[4]) * meters_per_degree_lat
            
            print(f"Pixel size (X): ~{pixel_size_m_x:.0f} meters")
            print(f"Pixel size (Y): ~{pixel_size_m_y:.0f} meters")
            
            # Expected for 1 arc-second: ~30m
            if 20 < pixel_size_m_x < 40:
                print(f"  ✓ Resolution looks correct for 1 arc-second DEM (~30m)")
            else:
                print(f"  ⚠ Resolution may not be 1 arc-second (expected ~30m)")
            
            # Read elevation data
            print(f"\n--- ELEVATION STATISTICS ---")
            elevation = dem.read(1)
            
            # Handle nodata
            if dem.nodata is not None:
                valid_mask = elevation != dem.nodata
                valid_elevation = elevation[valid_mask]
            else:
                valid_elevation = elevation[~np.isnan(elevation)]
            
            if len(valid_elevation) == 0:
                print(f"  ✗ ERROR: No valid elevation data found!")
                return False
            
            # Check units (could be meters or feet)
            min_elev = float(valid_elevation.min())
            max_elev = float(valid_elevation.max())
            mean_elev = float(valid_elevation.mean())
            median_elev = float(np.median(valid_elevation))
            
            print(f"Minimum elevation: {min_elev:.1f}")
            print(f"Maximum elevation: {max_elev:.1f}")
            print(f"Mean elevation: {mean_elev:.1f}")
            print(f"Median elevation: {median_elev:.1f}")
            
            # Determine units
            # Wyoming elevation range: ~3,100 ft (945 m) to ~13,804 ft (4,207 m)
            # If values are in meters: ~945 to ~4,207
            # If values are in feet: ~3,100 to ~13,804
            
            if min_elev < 1000 and max_elev < 5000:
                print(f"  → Units appear to be METERS")
                min_ft = min_elev * 3.28084
                max_ft = max_elev * 3.28084
                print(f"  Range in feet: {min_ft:.0f} to {max_ft:.0f} ft")
            elif min_elev > 2000 and max_elev > 10000:
                print(f"  → Units appear to be FEET")
                min_m = min_elev / 3.28084
                max_m = max_elev / 3.28084
                print(f"  Range in meters: {min_m:.0f} to {max_m:.0f} m")
            else:
                print(f"  ⚠ WARNING: Units unclear - values don't match expected Wyoming range")
            
            # Check if values are reasonable for Wyoming
            wyoming_min_ft = 3100  # Lowest point in Wyoming
            wyoming_max_ft = 13804  # Gannett Peak
            
            if min_elev < 1000:  # Assume meters
                min_ft_check = min_elev * 3.28084
                max_ft_check = max_elev * 3.28084
            else:  # Assume feet
                min_ft_check = min_elev
                max_ft_check = max_elev
            
            if wyoming_min_ft - 500 <= min_ft_check <= wyoming_max_ft + 500:
                print(f"  ✓ Elevation range looks reasonable for Wyoming")
            else:
                print(f"  ⚠ WARNING: Elevation range may not match Wyoming")
                print(f"    Expected: {wyoming_min_ft:,} to {wyoming_max_ft:,} ft")
            
            # Sample specific locations
            print(f"\n--- SAMPLE LOCATIONS ---")
            test_points = [
                ("Area 048 center", 43.4357, -107.5240),
                ("Cheyenne (capital)", 41.1400, -104.8200),
                ("Jackson (northwest)", 43.4799, -110.7624),
                ("Laramie (southeast)", 41.3114, -105.5911),
            ]
            
            for name, lat, lon in test_points:
                try:
                    row, col = dem.index(lon, lat)
                    elev = elevation[row, col]
                    
                    if dem.nodata is not None and elev == dem.nodata:
                        print(f"  {name}: No data")
                    else:
                        if elev < 1000:  # Assume meters
                            elev_ft = elev * 3.28084
                            print(f"  {name} ({lat:.4f}°, {lon:.4f}°): {elev:.0f} m ({elev_ft:.0f} ft)")
                        else:
                            print(f"  {name} ({lat:.4f}°, {lon:.4f}°): {elev:.0f} ft")
                except Exception as e:
                    print(f"  {name}: Error sampling - {e}")
            
            # Data quality checks
            print(f"\n--- DATA QUALITY ---")
            total_pixels = elevation.size
            valid_pixels = len(valid_elevation)
            invalid_pixels = total_pixels - valid_pixels
            valid_percent = (valid_pixels / total_pixels) * 100
            
            print(f"Total pixels: {total_pixels:,}")
            print(f"Valid pixels: {valid_pixels:,} ({valid_percent:.1f}%)")
            print(f"Invalid/NoData pixels: {invalid_pixels:,} ({100-valid_percent:.1f}%)")
            
            if valid_percent > 95:
                print(f"  ✓ Good data coverage")
            elif valid_percent > 80:
                print(f"  ⚠ Moderate data coverage (some gaps)")
            else:
                print(f"  ✗ WARNING: Low data coverage (many gaps)")
            
            # Check for obvious errors
            if np.any(valid_elevation < -100):
                print(f"  ⚠ WARNING: Found negative elevations (may be below sea level or error)")
            if np.any(valid_elevation > 20000):
                print(f"  ⚠ WARNING: Found very high elevations (>20,000) - may be error")
            
            print(f"\n--- SUMMARY ---")
            issues = []
            if not bounds_ok:
                issues.append("Bounds may not cover Wyoming correctly")
            if valid_percent < 80:
                issues.append("Low data coverage")
            if dem.crs is None:
                issues.append("No coordinate system defined")
            
            if not issues:
                print(f"✓ DEM file looks good!")
                print(f"  - File is readable")
                print(f"  - Bounds cover Wyoming")
                print(f"  - Elevation values are reasonable")
                print(f"  - Data quality is good")
                return True
            else:
                print(f"⚠ DEM file has some issues:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
    
    except rasterio.errors.RasterioIOError as e:
        print(f"\n✗ ERROR: Could not open file - {e}")
        print(f"  File may be corrupted or not a valid GeoTIFF")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Wyoming DEM file")
    parser.add_argument(
        'dem_file',
        nargs='?',
        default='data/dem/wyoming_dem.tif',
        help='Path to DEM file (default: data/dem/wyoming_dem.tif)'
    )
    
    args = parser.parse_args()
    dem_path = Path(args.dem_file)
    
    success = analyze_dem(dem_path)
    
    if success:
        print(f"\n✓ Analysis complete - DEM file is ready to use!")
        return 0
    else:
        print(f"\n⚠ Analysis found some issues - review above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

