#!/usr/bin/env python3
"""
Download and process USGS trails for Wyoming.

This script provides multiple methods to acquire trail data:
1. Manual download from USGS National Map Downloader (recommended)
2. OpenStreetMap Overpass API (alternative, more complete coverage)

Usage:
    # Method 1: Manual download (recommended)
    # 1. Download trails from National Map Downloader
    # 2. Extract to data/raw/infrastructure/trails/
    # 3. Run this script to process:
    python scripts/download_usgs_trails.py --input data/raw/infrastructure/trails/trails.shp

    # Method 2: OpenStreetMap (alternative)
    python scripts/download_usgs_trails.py --osm

Optional arguments:
    --input: Path to downloaded trails shapefile (for manual method)
    --output: Output GeoJSON path (default: data/infrastructure/trails.geojson)
    --osm: Use OpenStreetMap Overpass API instead of USGS
    --no-clip: Skip clipping to Wyoming boundary (not recommended)
    --simplify: Simplify geometries to reduce file size (default: False)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
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

# Wyoming bounding box (WGS84)
WYOMING_BOUNDS = box(-111.0, 41.0, -104.0, 45.0)


def process_trails_shapefile(
    input_path: Path,
    output_path: Optional[Path] = None,
    clip_to_wyoming: bool = True,
    simplify: bool = False
) -> Path:
    """
    Process trails shapefile and convert to GeoJSON.
    
    Args:
        input_path: Path to input trails shapefile
        output_path: Path to save GeoJSON file
        clip_to_wyoming: Whether to clip to Wyoming boundary
        simplify: Whether to simplify geometries
        
    Returns:
        Path to saved GeoJSON file
    """
    if output_path is None:
        output_path = Path("data/infrastructure/trails.geojson")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading trails from: {input_path}")
    trails = gpd.read_file(input_path)
    logger.info(f"  Loaded {len(trails):,} trail segments")
    
    # Clip to Wyoming boundary if requested
    if clip_to_wyoming:
        wyoming_boundary_path = Path("data/boundaries/wyoming_state.shp")
        if wyoming_boundary_path.exists():
            logger.info("Clipping to Wyoming boundary...")
            wyoming = gpd.read_file(wyoming_boundary_path)
            
            # Ensure same CRS
            if trails.crs != wyoming.crs:
                logger.info(f"  Reprojecting trails from {trails.crs} to {wyoming.crs}")
                trails = trails.to_crs(wyoming.crs)
            
            # Clip
            trails = gpd.clip(trails, wyoming)
            logger.info(f"  Clipped to {len(trails):,} trail segments within Wyoming")
        else:
            logger.warning(f"Wyoming boundary not found at {wyoming_boundary_path}")
            logger.warning("  Using bounding box clip instead")
            # Use bounding box as fallback
            if trails.crs != 'EPSG:4326':
                trails = trails.to_crs('EPSG:4326')
            trails = trails[trails.geometry.intersects(WYOMING_BOUNDS)]
            logger.info(f"  Clipped to {len(trails):,} trail segments within Wyoming bounds")
    
    # Ensure WGS84 CRS for GeoJSON
    if trails.crs is None:
        logger.warning("Trails have no CRS - assuming EPSG:4326 (WGS84)")
        trails.set_crs('EPSG:4326', inplace=True)
    elif trails.crs != 'EPSG:4326':
        logger.info(f"Reprojecting from {trails.crs} to EPSG:4326 (WGS84)")
        trails = trails.to_crs('EPSG:4326')
    
    # Simplify geometries if requested
    if simplify:
        logger.info("Simplifying geometries...")
        # Use tolerance of ~10 meters (0.0001 degrees ≈ 11m at Wyoming latitude)
        trails['geometry'] = trails.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        logger.info("  Geometries simplified")
    
    # Select only necessary columns to reduce file size
    columns_to_keep = ['geometry']
    if 'NAME' in trails.columns:
        columns_to_keep.append('NAME')
    if 'TYPE' in trails.columns:
        columns_to_keep.append('TYPE')
    if 'FCODE' in trails.columns:
        columns_to_keep.append('FCODE')
    
    trails = trails[columns_to_keep]
    
    # Save to GeoJSON
    logger.info(f"Saving to GeoJSON: {output_path}")
    trails.to_file(output_path, driver='GeoJSON')
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAILS PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total trail segments: {len(trails):,}")
    
    # Calculate total trail length
    if trails.crs == 'EPSG:4326':
        # Convert to UTM for accurate length calculation
        trails_utm = trails.to_crs('EPSG:32612')  # UTM Zone 12N (Wyoming)
        total_length_m = trails_utm.geometry.length.sum()
        total_length_mi = total_length_m / 1609.34
        logger.info(f"Total trail length: {total_length_mi:,.1f} miles")
    
    logger.info("="*60)
    
    return output_path


def download_trails_osm(
    output_path: Optional[Path] = None,
    clip_to_wyoming: bool = True,
    simplify: bool = False
) -> Path:
    """
    Download trails from OpenStreetMap using Overpass API.
    
    Args:
        output_path: Path to save GeoJSON file
        clip_to_wyoming: Whether to clip to Wyoming boundary
        simplify: Whether to simplify geometries
        
    Returns:
        Path to saved GeoJSON file
    """
    try:
        import requests
        import json
    except ImportError:
        raise ImportError("requests package required for OSM download. Install with: pip install requests")
    
    if output_path is None:
        output_path = Path("data/infrastructure/trails.geojson")
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading trails from OpenStreetMap Overpass API...")
    logger.info("  This may take a few minutes for Wyoming...")
    
    # Overpass API query for trails in Wyoming
    # Query for: highway=path, highway=footway, highway=bridleway, route=hiking, route=horse
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Wyoming bounding box (south, west, north, east)
    query = f"""
    [out:json][timeout:300];
    (
      way["highway"="path"](41.0,-111.0,45.0,-104.0);
      way["highway"="footway"](41.0,-111.0,45.0,-104.0);
      way["highway"="bridleway"](41.0,-111.0,45.0,-104.0);
      relation["route"="hiking"](41.0,-111.0,45.0,-104.0);
      relation["route"="horse"](41.0,-111.0,45.0,-104.0);
    );
    out geom;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=600)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"  Received {len(data.get('elements', []))} OSM elements")
        
        # Convert OSM data to GeoDataFrame
        # This is a simplified conversion - you may want to use osmnx for more robust handling
        features = []
        for element in data.get('elements', []):
            if element['type'] == 'way' and 'geometry' in element:
                # Extract coordinates
                coords = [[point['lon'], point['lat']] for point in element['geometry']]
                if len(coords) >= 2:
                    from shapely.geometry import LineString
                    geom = LineString(coords)
                    
                    # Extract tags
                    tags = element.get('tags', {})
                    feature = {
                        'geometry': geom,
                        'name': tags.get('name', ''),
                        'highway': tags.get('highway', ''),
                        'route': tags.get('route', ''),
                        'osm_id': element.get('id', '')
                    }
                    features.append(feature)
        
        if not features:
            raise ValueError("No trail features found in OSM data")
        
        trails = gpd.GeoDataFrame(features, crs='EPSG:4326')
        logger.info(f"  Converted to {len(trails):,} trail segments")
        
        # Clip to Wyoming boundary if requested
        if clip_to_wyoming:
            wyoming_boundary_path = Path("data/boundaries/wyoming_state.shp")
            if wyoming_boundary_path.exists():
                logger.info("Clipping to Wyoming boundary...")
                wyoming = gpd.read_file(wyoming_boundary_path)
                
                # Ensure same CRS
                if trails.crs != wyoming.crs:
                    trails = trails.to_crs(wyoming.crs)
                
                # Clip
                trails = gpd.clip(trails, wyoming)
                logger.info(f"  Clipped to {len(trails):,} trail segments within Wyoming")
        
        # Simplify geometries if requested
        if simplify:
            logger.info("Simplifying geometries...")
            trails['geometry'] = trails.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        
        # Save to GeoJSON
        logger.info(f"Saving to GeoJSON: {output_path}")
        trails.to_file(output_path, driver='GeoJSON')
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAILS PROCESSING COMPLETE (OSM)")
        logger.info("="*60)
        logger.info(f"Output file: {output_path}")
        logger.info(f"Total trail segments: {len(trails):,}")
        
        # Calculate total trail length
        trails_utm = trails.to_crs('EPSG:32612')  # UTM Zone 12N (Wyoming)
        total_length_m = trails_utm.geometry.length.sum()
        total_length_mi = total_length_m / 1609.34
        logger.info(f"Total trail length: {total_length_mi:,.1f} miles")
        
        logger.info("="*60)
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OSM API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing OSM data: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and process trails for Wyoming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Method 1: Process manually downloaded USGS trails
  python scripts/download_usgs_trails.py --input data/raw/infrastructure/trails/trails.shp

  # Method 2: Download from OpenStreetMap
  python scripts/download_usgs_trails.py --osm

For manual USGS download:
  1. Visit: https://apps.nationalmap.gov/downloader/
  2. Select Wyoming extent
  3. Choose "Trails" layer
  4. Download and extract to data/raw/infrastructure/trails/
  5. Run this script with --input pointing to the .shp file
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to downloaded trails shapefile (for manual USGS method)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/infrastructure/trails.geojson',
        help='Output GeoJSON path (default: data/infrastructure/trails.geojson)'
    )
    parser.add_argument(
        '--osm',
        action='store_true',
        help='Use OpenStreetMap Overpass API instead of USGS'
    )
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Skip clipping to Wyoming boundary (not recommended)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify geometries to reduce file size'
    )
    
    args = parser.parse_args()
    
    if args.osm and args.input:
        parser.error("Cannot use both --osm and --input. Choose one method.")
    
    if not args.osm and not args.input:
        parser.error("Must specify either --input (for USGS) or --osm (for OpenStreetMap)")
    
    try:
        if args.osm:
            output_path = download_trails_osm(
                output_path=Path(args.output),
                clip_to_wyoming=not args.no_clip,
                simplify=args.simplify
            )
        else:
            output_path = process_trails_shapefile(
                input_path=Path(args.input),
                output_path=Path(args.output),
                clip_to_wyoming=not args.no_clip,
                simplify=args.simplify
            )
        
        print(f"\n✓ Success! Trails saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Verify the file loads correctly:")
        print(f"     python -c \"import geopandas as gpd; gdf = gpd.read_file('{output_path}'); print(f'Loaded {{len(gdf)}} trails')\"")
        print(f"  2. Test integration with:")
        print(f"     python scripts/integrate_environmental_features.py data/processed/combined_southern_bighorn_presence_absence.csv --limit 10")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

