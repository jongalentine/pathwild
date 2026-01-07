#!/usr/bin/env python3
"""
Download and process TIGER/Line roads for Wyoming.

This script:
1. Downloads TIGER/Line roads shapefile for Wyoming (FIPS code 56)
2. Filters to relevant road types (primary, secondary, tertiary, unclassified)
3. Clips to Wyoming boundary
4. Converts to GeoJSON format
5. Saves to data/infrastructure/roads.geojson

Usage:
    python scripts/download_tiger_roads.py

Optional arguments:
    --year: Year of TIGER/Line data (default: 2023)
    --output: Output GeoJSON path (default: data/infrastructure/roads.geojson)
    --no-clip: Skip clipping to Wyoming boundary (not recommended)
    --road-types: Comma-separated list of road types to include
                  (default: S1100,S1200,S1400,S1500,S1710,S1720,S1730,S1740,S1750,S1780,S1820,S1830)
    --simplify: Simplify geometries to reduce file size (default: False)
"""

import argparse
import sys
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional
import logging

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("ERROR: geopandas required")
    print("Install with: conda install geopandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wyoming FIPS code
WYOMING_FIPS = '56'

# TIGER/Line MTFCC (MAF/TIGER Feature Class Code) for roads
# See: https://www.census.gov/geographies/reference-files/time-series/geo/tiger-line-file.html
# S1100: Primary roads (Interstate highways)
# S1200: Primary roads (US highways)
# S1400: Secondary roads (State highways)
# S1500: Local neighborhood roads
# S1710: Local roads (paved)
# S1720: Local roads (unpaved)
# S1730: Local roads (4WD)
# S1740: Local roads (private)
# S1750: Local roads (other)
# S1780: Parking lot roads
# S1820: Service drives
# S1830: Walkways/pedestrian paths (exclude for roads)
ROAD_TYPES = {
    'S1100': 'primary',      # Primary roads (Interstate)
    'S1200': 'primary',      # Primary roads (US highways)
    'S1400': 'secondary',    # Secondary roads (State highways)
    'S1500': 'tertiary',      # Local neighborhood roads
    'S1710': 'unclassified', # Local roads (paved)
    'S1720': 'unclassified', # Local roads (unpaved)
    'S1730': 'unclassified', # Local roads (4WD)
    'S1740': 'unclassified', # Local roads (private)
    'S1750': 'unclassified', # Local roads (other)
    'S1780': 'unclassified', # Parking lot roads
    'S1820': 'unclassified', # Service drives
    # S1830: Walkways - exclude (these are pedestrian paths, not roads)
}

# Default road types to include (exclude walkways)
DEFAULT_ROAD_TYPES = [
    'S1100', 'S1200', 'S1400', 'S1500',
    'S1710', 'S1720', 'S1730', 'S1740', 'S1750', 'S1780', 'S1820'
]


def process_tiger_roads_shapefile(
    input_path: Path,
    output_path: Optional[Path] = None,
    clip_to_wyoming: bool = True,
    road_types: Optional[List[str]] = None,
    simplify: bool = False
) -> Path:
    """
    Process a manually downloaded TIGER/Line roads shapefile.
    
    Args:
        input_path: Path to input roads shapefile
        output_path: Path to save GeoJSON file
        clip_to_wyoming: Whether to clip to Wyoming boundary
        road_types: List of MTFCC codes to include (default: all except walkways)
        simplify: Whether to simplify geometries
        
    Returns:
        Path to saved GeoJSON file
    """
    if output_path is None:
        output_path = Path("data/infrastructure/roads.geojson")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Use default road types if not specified
    if road_types is None:
        road_types = DEFAULT_ROAD_TYPES
    
    logger.info(f"Processing TIGER/Line roads from: {input_path}")
    
    # Load roads shapefile
    logger.info("Loading roads shapefile...")
    roads = gpd.read_file(input_path)
    logger.info(f"  Loaded {len(roads):,} road segments")
    
    # Filter by road type (MTFCC code)
    if 'MTFCC' in roads.columns:
        logger.info(f"Filtering to road types: {', '.join(road_types)}")
        roads = roads[roads['MTFCC'].isin(road_types)].copy()
        logger.info(f"  Filtered to {len(roads):,} road segments")
    else:
        logger.warning("MTFCC column not found - including all roads")
    
    # Continue with clipping and processing (same as download function)
    return _process_roads_dataframe(roads, output_path, clip_to_wyoming, simplify)


def _process_roads_dataframe(
    roads: gpd.GeoDataFrame,
    output_path: Path,
    clip_to_wyoming: bool,
    simplify: bool
) -> Path:
    """
    Process a roads GeoDataFrame (clipping, conversion, etc.).
    
    This is a helper function used by both download and manual processing.
    """
    # Clip to Wyoming boundary if requested
    if clip_to_wyoming:
        wyoming_boundary_path = Path("data/boundaries/wyoming_state.shp")
        if wyoming_boundary_path.exists():
            logger.info("Clipping to Wyoming boundary...")
            wyoming = gpd.read_file(wyoming_boundary_path)
            
            # Ensure same CRS
            if roads.crs != wyoming.crs:
                logger.info(f"  Reprojecting roads from {roads.crs} to {wyoming.crs}")
                roads = roads.to_crs(wyoming.crs)
            
            # Clip
            roads = gpd.clip(roads, wyoming)
            logger.info(f"  Clipped to {len(roads):,} road segments within Wyoming")
        else:
            logger.warning(f"Wyoming boundary not found at {wyoming_boundary_path}")
            logger.warning("  Skipping clip - roads may extend beyond Wyoming")
    
    # Ensure WGS84 CRS for GeoJSON
    if roads.crs is None:
        logger.warning("Roads have no CRS - assuming EPSG:4326 (WGS84)")
        roads.set_crs('EPSG:4326', inplace=True)
    elif roads.crs != 'EPSG:4326':
        logger.info(f"Reprojecting from {roads.crs} to EPSG:4326 (WGS84)")
        roads = roads.to_crs('EPSG:4326')
    
    # Simplify geometries if requested
    if simplify:
        logger.info("Simplifying geometries...")
        # Use tolerance of ~10 meters (0.0001 degrees ≈ 11m at Wyoming latitude)
        roads['geometry'] = roads.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        logger.info("  Geometries simplified")
    
    # Add road type attribute if MTFCC exists
    if 'MTFCC' in roads.columns:
        roads['road_type'] = roads['MTFCC'].map(ROAD_TYPES)
        # Fill NaN with 'unclassified'
        roads['road_type'] = roads['road_type'].fillna('unclassified')
    
    # Select only necessary columns to reduce file size
    columns_to_keep = ['geometry']
    if 'MTFCC' in roads.columns:
        columns_to_keep.append('MTFCC')
    if 'road_type' in roads.columns:
        columns_to_keep.append('road_type')
    if 'FULLNAME' in roads.columns:
        columns_to_keep.append('FULLNAME')
    if 'RTTYP' in roads.columns:  # Route type (I, U, S, etc.)
        columns_to_keep.append('RTTYP')
    
    roads = roads[columns_to_keep]
    
    # Save to GeoJSON
    logger.info(f"Saving to GeoJSON: {output_path}")
    roads.to_file(output_path, driver='GeoJSON')
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ROADS PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total road segments: {len(roads):,}")
    
    if 'road_type' in roads.columns:
        road_type_counts = roads['road_type'].value_counts()
        logger.info("\nRoad type distribution:")
        for road_type, count in road_type_counts.items():
            pct = (count / len(roads)) * 100
            logger.info(f"  {road_type:15s}: {count:8,} ({pct:5.1f}%)")
    
    # Calculate total road length
    if roads.crs == 'EPSG:4326':
        # Convert to UTM for accurate length calculation
        roads_utm = roads.to_crs('EPSG:32612')  # UTM Zone 12N (Wyoming)
        total_length_m = roads_utm.geometry.length.sum()
        total_length_mi = total_length_m / 1609.34
        logger.info(f"\nTotal road length: {total_length_mi:,.1f} miles")
    
    logger.info("="*60)
    
    return output_path


def download_tiger_roads(
    year: int = 2023,
    output_path: Optional[Path] = None,
    clip_to_wyoming: bool = True,
    road_types: Optional[List[str]] = None,
    simplify: bool = False
) -> Path:
    """
    Download and process TIGER/Line roads for Wyoming.
    
    Args:
        year: Year of TIGER/Line data
        output_path: Path to save GeoJSON file
        clip_to_wyoming: Whether to clip to Wyoming boundary
        road_types: List of MTFCC codes to include (default: all except walkways)
        simplify: Whether to simplify geometries
        
    Returns:
        Path to saved GeoJSON file
    """
    if output_path is None:
        output_path = Path("data/infrastructure/roads.geojson")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use default road types if not specified
    if road_types is None:
        road_types = DEFAULT_ROAD_TYPES
    
    # TIGER/Line roads URL patterns (try multiple in case structure changed)
    # The Census Bureau sometimes changes URL structures
    url_patterns = [
        f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{WYOMING_FIPS}_roads.zip",
        f"https://www2.census.gov/geo/tiger/TIGER{year}/tl_{year}_{WYOMING_FIPS}_roads.zip",
        f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{WYOMING_FIPS:0>2}_roads.zip",  # Zero-padded FIPS
    ]
    
    logger.info(f"Downloading TIGER/Line roads for Wyoming (FIPS {WYOMING_FIPS})...")
    logger.info(f"Year: {year}")
    
    # Try each URL pattern until one works
    response = None
    successful_url = None
    
    for url in url_patterns:
        logger.info(f"Trying URL: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                successful_url = url
                logger.info(f"✓ Found valid URL: {url}")
                break
            else:
                logger.debug(f"  URL returned status {response.status_code}, trying next...")
        except requests.exceptions.RequestException as e:
            logger.debug(f"  URL failed: {e}, trying next...")
            continue
    
    if response is None or successful_url is None:
        # All URLs failed, provide helpful error message
        logger.error("All URL patterns failed. The Census Bureau may have changed their URL structure.")
        logger.error("\nPlease download manually:")
        logger.error("1. Visit: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        logger.error(f"2. Navigate to: TIGER/Line Shapefiles > {year} > Roads")
        logger.error(f"3. Download: tl_{year}_{WYOMING_FIPS}_roads.zip")
        logger.error("4. Extract and process manually, or update the script with the correct URL")
        raise FileNotFoundError(f"Could not download TIGER/Line roads. Tried {len(url_patterns)} URL patterns.")
    
    # Download the ZIP file
    try:
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading {total_size / 1024 / 1024:.1f} MB...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = Path(tmp_file.name)
            
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                        percent = (downloaded / total_size) * 100
                        logger.info(f"  Progress: {percent:.1f}%")
        
        logger.info("✓ Download complete")
        
        # Extract ZIP file
        logger.info("Extracting ZIP file...")
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_ref.extractall(tmp_dir)
                
                # Find the shapefile
                tmp_path = Path(tmp_dir)
                shp_files = list(tmp_path.glob("*.shp"))
                
                if not shp_files:
                    raise FileNotFoundError("No shapefile found in ZIP archive")
                
                shp_path = shp_files[0]
                logger.info(f"Found shapefile: {shp_path.name}")
                
                # Load roads shapefile
                logger.info("Loading roads shapefile...")
                roads = gpd.read_file(shp_path)
                logger.info(f"  Loaded {len(roads):,} road segments")
                
                # Filter by road type (MTFCC code)
                if 'MTFCC' in roads.columns:
                    logger.info(f"Filtering to road types: {', '.join(road_types)}")
                    roads = roads[roads['MTFCC'].isin(road_types)].copy()
                    logger.info(f"  Filtered to {len(roads):,} road segments")
                else:
                    logger.warning("MTFCC column not found - including all roads")
                
                # Process the roads dataframe (clipping, conversion, etc.)
                _process_roads_dataframe(roads, output_path, clip_to_wyoming, simplify)
        
        # Clean up
        tmp_zip_path.unlink()
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        logger.error("\nThe Census Bureau URL structure may have changed.")
        logger.error("\nPlease download manually:")
        logger.error("1. Visit: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        logger.error(f"2. Navigate to: TIGER/Line Shapefiles > {year} > Roads")
        logger.error(f"3. Download: tl_{year}_{WYOMING_FIPS}_roads.zip")
        logger.error("4. Extract to a temporary directory")
        logger.error("5. Process with:")
        logger.error(f"   python scripts/download_tiger_roads.py --input path/to/tl_{year}_{WYOMING_FIPS}_roads.shp")
        raise
    except Exception as e:
        logger.error(f"Error processing roads: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and process TIGER/Line roads for Wyoming"
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='Year of TIGER/Line data (default: 2023)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/infrastructure/roads.geojson',
        help='Output GeoJSON path (default: data/infrastructure/roads.geojson)'
    )
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Skip clipping to Wyoming boundary (not recommended)'
    )
    parser.add_argument(
        '--road-types',
        type=str,
        help='Comma-separated list of MTFCC codes to include (default: all except walkways)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify geometries to reduce file size'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to manually downloaded roads shapefile (skip download, process local file)'
    )
    
    args = parser.parse_args()
    
    # If input file provided, process it instead of downloading
    if args.input:
        try:
            output_path = process_tiger_roads_shapefile(
                input_path=Path(args.input),
                output_path=Path(args.output),
                clip_to_wyoming=not args.no_clip,
                road_types=None if not args.road_types else [rt.strip() for rt in args.road_types.split(',')],
                simplify=args.simplify
            )
            print(f"\n✓ Success! Roads saved to: {output_path}")
            return 0
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return 1
    
    # Parse road types if provided
    road_types = None
    if args.road_types:
        road_types = [rt.strip() for rt in args.road_types.split(',')]
    
    try:
        output_path = download_tiger_roads(
            year=args.year,
            output_path=Path(args.output),
            clip_to_wyoming=not args.no_clip,
            road_types=road_types,
            simplify=args.simplify
        )
        print(f"\n✓ Success! Roads saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Verify the file loads correctly:")
        print(f"     python -c \"import geopandas as gpd; gdf = gpd.read_file('{output_path}'); print(f'Loaded {{len(gdf)}} roads')\"")
        print(f"  2. Test integration with:")
        print(f"     python scripts/integrate_environmental_features.py data/processed/combined_southern_bighorn_presence_absence.csv --limit 10")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

