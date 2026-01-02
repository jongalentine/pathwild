#!/usr/bin/env python3
"""
Extract and clip CONUS NLCD dataset to Wyoming.

This script:
1. Extracts the .tif file from the CONUS NLCD zip
2. Clips it to Wyoming bounds
3. Saves to data/landcover/nlcd.tif

Usage:
    python scripts/clip_conus_nlcd_to_wyoming.py [--zip-file PATH] [--output-file PATH]
"""

import argparse
import sys
from pathlib import Path
import zipfile
import tempfile
import shutil

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import box
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("ERROR: rasterio and geopandas required")
    print("Install with: conda install rasterio geopandas")
    sys.exit(1)

# Wyoming bounding box (WGS84)
WYOMING_BOUNDS = {
    'west': -111.0,
    'east': -104.0,
    'south': 41.0,
    'north': 45.0
}


def extract_tif_from_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract .tif file from zip archive."""
    print(f"Extracting .tif from zip: {zip_path}")
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find .tif file in zip
        tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
        
        if not tif_files:
            raise ValueError(f"No .tif file found in zip: {zip_path}")
        
        if len(tif_files) > 1:
            print(f"  ⚠ Multiple .tif files found, using first: {tif_files[0]}")
        
        tif_file = tif_files[0]
        print(f"  Extracting: {tif_file}")
        
        # Extract to temp directory
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_ref.extract(tif_file, extract_dir)
        
        extracted_path = extract_dir / tif_file
        if not extracted_path.exists():
            # Handle case where zip has subdirectory structure
            extracted_path = extract_dir / Path(tif_file).name
        
        print(f"  ✓ Extracted to: {extracted_path}")
        return extracted_path


def clip_to_wyoming(input_path: Path, output_path: Path):
    """Clip NLCD raster to Wyoming bounds."""
    print(f"\nClipping NLCD to Wyoming bounds...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Create Wyoming bounding box as GeoDataFrame
    wyoming_box = box(
        WYOMING_BOUNDS['west'],
        WYOMING_BOUNDS['south'],
        WYOMING_BOUNDS['east'],
        WYOMING_BOUNDS['north']
    )
    wyoming_gdf = gpd.GeoDataFrame({'geometry': [wyoming_box]}, crs='EPSG:4326')
    
    # Open input raster
    with rasterio.open(input_path) as src:
        print(f"  Original dimensions: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  NoData value: {src.nodata}")
        
        # Reproject Wyoming bounds to raster CRS
        wyoming_gdf_reprojected = wyoming_gdf.to_crs(src.crs)
        wyoming_geom = wyoming_gdf_reprojected.geometry.values[0]
        
        print(f"  Wyoming bounds (in raster CRS): {wyoming_geom.bounds}")
        
        # Clip using rasterio.mask
        print(f"  Clipping...")
        out_image, out_transform = mask(
            src,
            [wyoming_geom],
            crop=True,
            nodata=src.nodata
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'compress': 'lzw'  # Compress to save space
        })
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write clipped raster
        print(f"  Writing clipped raster...")
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(out_image)
        
        print(f"  ✓ Clipped and saved to: {output_path}")
        
        # Verify output
        with rasterio.open(output_path) as dst:
            print(f"\n  Output verification:")
            print(f"    Dimensions: {dst.width} x {dst.height}")
            print(f"    CRS: {dst.crs}")
            print(f"    Bounds: {dst.bounds}")
            
            # Check land cover classes
            data = dst.read(1)
            unique_classes = sorted(set(data[data != dst.nodata].flatten()))
            print(f"    Land cover classes: {len(unique_classes)}")
            print(f"      Classes: {unique_classes[:20]}{'...' if len(unique_classes) > 20 else ''}")
            
            # File size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clip CONUS NLCD dataset to Wyoming"
    )
    parser.add_argument(
        '--zip-file',
        type=Path,
        default=Path('data/landcover/Annual_NLCD_FctImp_2024_CU_C1V1.zip'),
        help='Path to CONUS NLCD zip file'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('data/landcover/nlcd.tif'),
        help='Output file path (default: data/landcover/nlcd.tif)'
    )
    parser.add_argument(
        '--keep-extracted',
        action='store_true',
        help='Keep extracted .tif file (do not delete after processing)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLIP CONUS NLCD TO WYOMING")
    print("=" * 60)
    print(f"\nZip file: {args.zip_file}")
    print(f"Output file: {args.output_file}")
    
    if not args.zip_file.exists():
        print(f"\n✗ ERROR: Zip file not found: {args.zip_file}")
        return 1
    
    # Check zip file size
    zip_size_gb = args.zip_file.stat().st_size / (1024 * 1024 * 1024)
    print(f"Zip file size: {zip_size_gb:.2f} GB")
    
    # Extract .tif from zip
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_tif = extract_tif_from_zip(args.zip_file, temp_path)
        
        # Clip to Wyoming
        clip_to_wyoming(extracted_tif, args.output_file)
        
        # Optionally keep extracted file
        if args.keep_extracted:
            keep_path = args.output_file.parent / extracted_tif.name
            shutil.copy2(extracted_tif, keep_path)
            print(f"\n  ✓ Kept extracted file: {keep_path}")
    
    print(f"\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\n✓ Clipped NLCD data saved to: {args.output_file}")
    print(f"\nNext steps:")
    print(f"1. Verify the file:")
    print(f"   python scripts/analyze_nlcd.py {args.output_file}")
    print(f"2. Re-run environmental feature integration:")
    print(f"   python scripts/integrate_environmental_features.py \\")
    print(f"       data/processed/combined_north_bighorn_presence_absence_cleaned.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

