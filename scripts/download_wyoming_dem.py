#!/usr/bin/env python3
"""
Download Wyoming DEM tiles from AWS S3 (USGS 3DEP 1 arc-second).

This script downloads all DEM tiles needed to cover Wyoming.
Much faster than downloading 300K+ individual files!

Usage:
    python scripts/download_wyoming_dem.py [--output-dir DIR] [--mosaic]

Requirements:
    pip install requests tqdm
"""

import argparse
import sys
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for progress bars: pip install tqdm")

# Wyoming bounding box tile coverage
# 1 arc-second DEM tiles are 1°x1° and named by northwest corner (nXXwYYY)
# Wyoming covers approximately these tiles:
WYOMING_TILES = [
    # Row 41°N
    'n41w111', 'n41w110', 'n41w109', 'n41w108', 'n41w107', 'n41w106', 'n41w105', 'n41w104',
    # Row 42°N
    'n42w111', 'n42w110', 'n42w109', 'n42w108', 'n42w107', 'n42w106', 'n42w105', 'n42w104',
    # Row 43°N
    'n43w111', 'n43w110', 'n43w109', 'n43w108', 'n43w107', 'n43w106', 'n43w105', 'n43w104',
    # Row 44°N
    'n44w111', 'n44w110', 'n44w109', 'n44w108', 'n44w107', 'n44w106', 'n44w105', 'n44w104',
    # Row 45°N
    'n45w111', 'n45w110', 'n45w109', 'n45w108', 'n45w107', 'n45w106', 'n45w105', 'n45w104',
]

# AWS S3 public bucket for USGS 3DEP
S3_BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current"


def download_tile(tile_name: str, output_dir: Path, max_retries: int = 3) -> tuple:
    """
    Download a single DEM tile from AWS S3.
    
    Args:
        tile_name: Tile name (e.g., 'n41w111')
        output_dir: Directory to save tile
        max_retries: Maximum retry attempts
        
    Returns:
        (tile_name, success: bool, error_message: str)
    """
    filename = f"USGS_1_{tile_name}.tif"
    url = f"{S3_BASE_URL}/{tile_name}/{filename}"
    output_file = output_dir / filename
    
    # Skip if already downloaded
    if output_file.exists():
        return (tile_name, True, "already exists")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Check file size (should be several MB)
            total_size = int(response.headers.get('content-length', 0))
            if total_size < 1000:  # Suspiciously small
                return (tile_name, False, f"File too small ({total_size} bytes)")
            
            # Download with progress
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify file size
            if output_file.stat().st_size < 1000:
                output_file.unlink()  # Delete bad file
                raise Exception("Downloaded file is too small")
            
            return (tile_name, True, f"{downloaded / 1024 / 1024:.1f} MB")
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return (tile_name, False, str(e))
        except Exception as e:
            if output_file.exists():
                output_file.unlink()  # Clean up on error
            return (tile_name, False, str(e))
    
    return (tile_name, False, "Max retries exceeded")


def mosaic_tiles(tiles_dir: Path, output_file: Path):
    """
    Mosaic all DEM tiles into a single file.
    
    Requires GDAL to be installed.
    """
    try:
        import subprocess
        import glob
        
        tile_files = list(tiles_dir.glob("USGS_1_*.tif"))
        if not tile_files:
            print("No tile files found to mosaic")
            return False
        
        print(f"\nMosaicking {len(tile_files)} tiles...")
        
        # Use gdal_merge.py
        cmd = [
            'gdal_merge.py',
            '-o', str(output_file),
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',  # Compress to save space
        ] + [str(f) for f in tile_files]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Mosaicked DEM saved to: {output_file}")
            return True
        else:
            print(f"✗ Mosaic failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ GDAL not found. Install with: brew install gdal (macOS) or apt-get install gdal-bin (Linux)")
        print("  Or use Python method (see script comments)")
        return False
    except Exception as e:
        print(f"✗ Mosaic error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Wyoming DEM tiles from AWS S3"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/dem/tiles'),
        help='Directory to save tiles (default: data/dem/tiles)'
    )
    parser.add_argument(
        '--mosaic',
        action='store_true',
        help='Mosaic tiles into single file after download'
    )
    parser.add_argument(
        '--mosaic-output',
        type=Path,
        default=Path('data/dem/wyoming_dem.tif'),
        help='Output file for mosaicked DEM (default: data/dem/wyoming_dem.tif)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel downloads (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Wyoming DEM Downloader")
    print("=" * 60)
    print(f"Tiles to download: {len(WYOMING_TILES)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel workers: {args.workers}")
    print()
    
    # Check existing files
    existing = list(args.output_dir.glob("USGS_1_*.tif"))
    if existing:
        print(f"Found {len(existing)} existing tiles, will skip those")
    
    # Download tiles
    print("Downloading tiles...")
    start_time = time.time()
    
    results = []
    iterator = WYOMING_TILES
    if HAS_TQDM:
        iterator = tqdm(WYOMING_TILES, desc="Downloading")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_tile, tile, args.output_dir): tile 
            for tile in WYOMING_TILES
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            tile_name, success, message = result
            if success:
                if HAS_TQDM:
                    iterator.set_postfix_str(f"✓ {tile_name}")
                else:
                    print(f"  ✓ {tile_name}: {message}")
            else:
                print(f"  ✗ {tile_name}: {message}")
    
    elapsed = time.time() - start_time
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {successful}/{len(WYOMING_TILES)}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f} seconds")
    
    if failed > 0:
        print("\nFailed tiles:")
        for tile_name, success, message in results:
            if not success:
                print(f"  {tile_name}: {message}")
    
    # Mosaic if requested
    if args.mosaic:
        print()
        mosaic_tiles(args.output_dir, args.mosaic_output)
    else:
        print()
        print("To mosaic tiles into one file, run:")
        print(f"  gdal_merge.py -o {args.mosaic_output} {args.output_dir}/USGS_1_*.tif")
        print()
        print("Or re-run with --mosaic flag:")
        print(f"  python {sys.argv[0]} --mosaic")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

