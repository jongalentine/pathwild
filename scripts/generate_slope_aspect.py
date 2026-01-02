#!/usr/bin/env python3
"""
Generate slope and aspect rasters from DEM.

This script calculates slope and aspect from a DEM file using Python/rasterio,
which is more reliable than requiring GDAL command-line tools.

Usage:
    python scripts/generate_slope_aspect.py [--dem-file PATH] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.transform import xy
    from scipy import ndimage
except ImportError as e:
    print(f"ERROR: Required packages not installed: {e}")
    print("Install with: pip install rasterio scipy")
    print("Or: conda install rasterio scipy")
    sys.exit(1)


def calculate_slope(elevation, pixel_size_x, pixel_size_y):
    """
    Calculate slope in degrees from elevation array.
    
    Uses the Horn method (similar to GDAL's slope calculation).
    
    Args:
        elevation: 2D numpy array of elevation values
        pixel_size_x: Pixel width in meters
        pixel_size_y: Pixel height in meters
        
    Returns:
        2D numpy array of slope in degrees
    """
    # Calculate gradients using numpy
    # np.gradient returns gradients in units per pixel
    # For elevation in meters, this gives meters per pixel
    dy, dx = np.gradient(elevation)
    
    # Convert gradients from "meters per pixel" to "meters per meter"
    # by dividing by pixel size (not multiplying!)
    # This gives us the rise/run ratio
    dx_normalized = dx / pixel_size_x  # meters per meter (unitless)
    dy_normalized = dy / pixel_size_y  # meters per meter (unitless)
    
    # Calculate slope using Horn's method
    # Slope = atan(sqrt((dz/dx)^2 + (dz/dy)^2))
    # This gives slope in radians
    slope_rad = np.arctan(np.sqrt(dx_normalized**2 + dy_normalized**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg


def calculate_aspect(elevation, pixel_size_x, pixel_size_y):
    """
    Calculate aspect in degrees from elevation array.
    
    Aspect: direction the slope faces (0-360°, where 0°=North, 90°=East, etc.)
    
    Args:
        elevation: 2D numpy array of elevation values
        pixel_size_x: Pixel width in meters
        pixel_size_y: Pixel height in meters
        
    Returns:
        2D numpy array of aspect in degrees (0-360)
    """
    # Calculate gradients
    dy, dx = np.gradient(elevation)
    
    # Convert gradients from "meters per pixel" to normalized units
    # by dividing by pixel size (consistent with slope calculation)
    dx_normalized = dx / pixel_size_x
    dy_normalized = dy / pixel_size_y
    
    # Calculate aspect
    # Aspect = atan2(-dz/dx, dz/dy) * 180/π
    # Negative dx because we want direction slope faces (not direction of steepest descent)
    aspect_rad = np.arctan2(-dx_normalized, dy_normalized)
    aspect_deg = np.degrees(aspect_rad)
    
    # Convert from -180 to 180 range to 0 to 360 range
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
    
    # Flat areas (slope = 0) should have aspect = 0
    slope = np.sqrt(dx_normalized**2 + dy_normalized**2)
    aspect_deg = np.where(slope < 0.01, 0, aspect_deg)
    
    return aspect_deg


def generate_slope_aspect(dem_path: Path, output_dir: Path):
    """
    Generate slope and aspect rasters from DEM.
    
    Args:
        dem_path: Path to input DEM file
        output_dir: Directory to save slope and aspect files
    """
    print("=" * 60)
    print("GENERATING SLOPE AND ASPECT FROM DEM")
    print("=" * 60)
    print(f"\nInput DEM: {dem_path}")
    print(f"Output directory: {output_dir}")
    
    if not dem_path.exists():
        print(f"\n✗ ERROR: DEM file not found: {dem_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    slope_path = output_dir / "slope.tif"
    aspect_path = output_dir / "aspect.tif"
    
    print(f"\nOpening DEM...")
    with rasterio.open(dem_path) as dem:
        print(f"  Dimensions: {dem.width} x {dem.height}")
        print(f"  CRS: {dem.crs}")
        print(f"  Bounds: {dem.bounds}")
        
        # Read elevation data
        print(f"\nReading elevation data...")
        elevation = dem.read(1).astype(np.float32)
        
        # Handle NoData values
        if dem.nodata is not None:
            nodata_mask = elevation == dem.nodata
            elevation = np.where(nodata_mask, np.nan, elevation)
            print(f"  NoData pixels: {np.sum(nodata_mask):,}")
        
        # Calculate pixel size in meters
        # Get transform to calculate pixel dimensions
        transform = dem.transform
        
        # Pixel size in degrees
        pixel_size_x_deg = abs(transform[0])
        pixel_size_y_deg = abs(transform[4])
        
        # Convert to meters (approximate, using center latitude)
        center_lat = (dem.bounds.top + dem.bounds.bottom) / 2
        meters_per_degree_lat = 111320.0  # Approximately constant
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
        
        pixel_size_x_m = pixel_size_x_deg * meters_per_degree_lon
        pixel_size_y_m = pixel_size_y_deg * meters_per_degree_lat
        
        print(f"  Pixel size: {pixel_size_x_deg:.6f}° x {pixel_size_y_deg:.6f}°")
        print(f"  Pixel size: ~{pixel_size_x_m:.0f}m x ~{pixel_size_y_m:.0f}m")
        
        # Calculate slope
        print(f"\nCalculating slope...")
        slope = calculate_slope(elevation, pixel_size_x_m, pixel_size_y_m)
        
        # Handle NaN values (NoData areas)
        if dem.nodata is not None:
            slope = np.where(nodata_mask, dem.nodata, slope)
        
        print(f"  Slope range: {np.nanmin(slope):.1f}° to {np.nanmax(slope):.1f}°")
        print(f"  Mean slope: {np.nanmean(slope):.1f}°")
        
        # Calculate aspect
        print(f"\nCalculating aspect...")
        aspect = calculate_aspect(elevation, pixel_size_x_m, pixel_size_y_m)
        
        # Handle NaN values
        if dem.nodata is not None:
            aspect = np.where(nodata_mask, dem.nodata, aspect)
        
        print(f"  Aspect range: {np.nanmin(aspect):.0f}° to {np.nanmax(aspect):.0f}°")
        
        # Save slope
        print(f"\nSaving slope raster...")
        slope_meta = dem.meta.copy()
        slope_meta.update({
            'dtype': 'float32',
            'nodata': dem.nodata if dem.nodata is not None else -9999.0,
            'compress': 'lzw'  # Compress to save space
        })
        
        with rasterio.open(slope_path, 'w', **slope_meta) as dst:
            dst.write(slope.astype(np.float32), 1)
        
        slope_size_mb = slope_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {slope_path} ({slope_size_mb:.1f} MB)")
        
        # Save aspect
        print(f"\nSaving aspect raster...")
        aspect_meta = dem.meta.copy()
        aspect_meta.update({
            'dtype': 'float32',
            'nodata': dem.nodata if dem.nodata is not None else -9999.0,
            'compress': 'lzw'
        })
        
        with rasterio.open(aspect_path, 'w', **aspect_meta) as dst:
            dst.write(aspect.astype(np.float32), 1)
        
        aspect_size_mb = aspect_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {aspect_path} ({aspect_size_mb:.1f} MB)")
        
        print(f"\n✓ Success! Slope and aspect rasters generated.")
        print(f"\nFiles created:")
        print(f"  - {slope_path}")
        print(f"  - {aspect_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate slope and aspect rasters from DEM"
    )
    parser.add_argument(
        '--dem-file',
        type=Path,
        default=Path('data/dem/wyoming_dem.tif'),
        help='Path to input DEM file (default: data/dem/wyoming_dem.tif)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/terrain'),
        help='Directory to save slope and aspect files (default: data/terrain)'
    )
    
    args = parser.parse_args()
    
    success = generate_slope_aspect(args.dem_file, args.output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

