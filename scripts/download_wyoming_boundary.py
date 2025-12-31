#!/usr/bin/env python3
"""
Download Wyoming state boundary shapefile from US Census Bureau.

This script downloads the Wyoming state boundary from the US Census Bureau's
Cartographic Boundary Files and extracts it to the data/boundaries/ directory.
"""

import argparse
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_wyoming_boundary(
    output_dir: Path,
    year: int = 2023,
    resolution: str = "500k"
) -> Path:
    """
    Download Wyoming state boundary shapefile from US Census Bureau.
    
    Args:
        output_dir: Directory to save the shapefile
        year: Year of the boundary file (2020, 2021, 2022, 2023, etc.)
        resolution: Resolution level ("500k", "5m", "20m")
        
    Returns:
        Path to the downloaded Wyoming shapefile
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # US Census Bureau Cartographic Boundary Files URL
    # Format: https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_state_{resolution}.zip
    base_url = "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_state_{resolution}.zip"
    url = base_url.format(year=year, resolution=resolution)
    
    logger.info(f"Downloading Wyoming boundary from US Census Bureau...")
    logger.info(f"URL: {url}")
    
    # Download the ZIP file
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = Path(tmp_file.name)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            logger.info(f"Downloading {total_size / 1024 / 1024:.1f} MB...")
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"  Progress: {percent:.1f}%")
        
        logger.info("✓ Download complete")
        
        # Extract ZIP file
        logger.info("Extracting ZIP file...")
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_ref.extractall(tmp_dir)
                
                # Find the shapefile
                tmp_path = Path(tmp_dir)
                shp_files = list(tmp_path.glob("*.shp"))
                
                if not shp_files:
                    raise FileNotFoundError("No shapefile found in ZIP archive")
                
                # Load all states and filter for Wyoming
                logger.info("Filtering for Wyoming...")
                all_states = gpd.read_file(shp_files[0])
                
                # Filter for Wyoming (check common name columns)
                name_cols = ['NAME', 'STUSPS', 'STATEFP']
                wyoming = None
                
                for col in name_cols:
                    if col in all_states.columns:
                        if col == 'STUSPS':  # State abbreviation
                            wyoming = all_states[all_states[col] == 'WY'].copy()
                        elif col == 'NAME':  # Full name
                            wyoming = all_states[all_states[col].str.upper() == 'WYOMING'].copy()
                        elif col == 'STATEFP':  # FIPS code (Wyoming = 56)
                            wyoming = all_states[all_states[col] == '56'].copy()
                        
                        if wyoming is not None and len(wyoming) > 0:
                            break
                
                if wyoming is None or len(wyoming) == 0:
                    raise ValueError("Wyoming not found in shapefile")
                
                # Save Wyoming shapefile
                output_path = output_dir / "wyoming_state.shp"
                wyoming.to_file(output_path)
                
                logger.info(f"✓ Wyoming boundary saved to: {output_path}")
        
        # Clean up
        tmp_zip_path.unlink()
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        logger.info("\nAlternative: Download manually from:")
        logger.info("  https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        raise
    except Exception as e:
        logger.error(f"Error processing shapefile: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download Wyoming state boundary shapefile"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/boundaries',
        help='Directory to save shapefile (default: data/boundaries)'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='Year of boundary file (default: 2023)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='500k',
        choices=['500k', '5m', '20m'],
        help='Resolution level (default: 500k)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = download_wyoming_boundary(
            Path(args.output_dir),
            year=args.year,
            resolution=args.resolution
        )
        print(f"\n✓ Success! Wyoming boundary saved to: {output_path}")
        print(f"\nYou can now use it with:")
        print(f"  python scripts/generate_absence_data.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        print("2. Navigate to 'Cartographic Boundary Files' > 'States'")
        print(f"3. Download: cb_{args.year}_us_state_{args.resolution}.zip")
        print("4. Extract and filter for Wyoming")
        print(f"5. Save to: {Path(args.output_dir) / 'wyoming_state.shp'}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

