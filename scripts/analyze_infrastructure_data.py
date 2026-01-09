#!/usr/bin/env python3
"""
Analyze roads and trails infrastructure data for quality checks.

This script validates the processed infrastructure data files.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

try:
    import geopandas as gpd
    import pandas as pd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("ERROR: geopandas required")
    print("Install with: conda install geopandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_roads(roads_path: Path) -> dict:
    """Analyze roads GeoJSON file."""
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING ROADS DATA")
    logger.info('='*60)
    
    if not roads_path.exists():
        logger.error(f"Roads file not found: {roads_path}")
        return {}
    
    try:
        gdf = gpd.read_file(roads_path)
        
        results = {
            'file_path': str(roads_path),
            'total_features': len(gdf),
            'crs': str(gdf.crs) if gdf.crs else 'None',
            'columns': list(gdf.columns),
            'geometry_types': gdf.geom_type.value_counts().to_dict(),
            'has_valid_geometry': gdf.geometry.is_valid.all(),
            'invalid_geometries': (~gdf.geometry.is_valid).sum() if hasattr(gdf.geometry, 'is_valid') else 0,
        }
        
        # Check bounding box (should be within Wyoming)
        bounds = gdf.total_bounds
        results['bounding_box'] = {
            'min_lon': bounds[0],
            'min_lat': bounds[1],
            'max_lon': bounds[2],
            'max_lat': bounds[3]
        }
        
        # Check if within Wyoming bounds (approximately -111 to -104, 41 to 45)
        wyoming_bounds = {'min_lon': -111.1, 'min_lat': 41.0, 'max_lon': -104.0, 'max_lat': 45.1}
        results['within_wyoming'] = (
            bounds[0] >= wyoming_bounds['min_lon'] - 0.1 and
            bounds[1] >= wyoming_bounds['min_lat'] - 0.1 and
            bounds[2] <= wyoming_bounds['max_lon'] + 0.1 and
            bounds[3] <= wyoming_bounds['max_lat'] + 0.1
        )
        
        # Calculate total length if in WGS84
        if gdf.crs == 'EPSG:4326' or str(gdf.crs) == 'CRS84':
            gdf_utm = gdf.to_crs('EPSG:32612')  # UTM Zone 12N
            total_length_m = gdf_utm.geometry.length.sum()
            total_length_mi = total_length_m / 1609.34
            results['total_length_miles'] = total_length_mi
        
        # Check for road_type column
        if 'road_type' in gdf.columns:
            results['road_type_distribution'] = gdf['road_type'].value_counts().to_dict()
        
        # Check for MTFCC column
        if 'MTFCC' in gdf.columns:
            results['mtfcc_distribution'] = gdf['MTFCC'].value_counts().head(10).to_dict()
        
        # Print results
        logger.info(f"File: {roads_path}")
        logger.info(f"Total features: {results['total_features']:,}")
        logger.info(f"CRS: {results['crs']}")
        logger.info(f"Columns: {', '.join(results['columns'])}")
        logger.info(f"\nGeometry types:")
        for geom_type, count in results['geometry_types'].items():
            logger.info(f"  {geom_type}: {count:,}")
        
        logger.info(f"\nBounding box:")
        logger.info(f"  Longitude: {bounds[0]:.6f} to {bounds[2]:.6f}")
        logger.info(f"  Latitude: {bounds[1]:.6f} to {bounds[3]:.6f}")
        logger.info(f"  Within Wyoming bounds: {'✓' if results['within_wyoming'] else '✗'}")
        
        if 'total_length_miles' in results:
            logger.info(f"\nTotal road length: {results['total_length_miles']:,.1f} miles")
        
        if 'road_type_distribution' in results:
            logger.info(f"\nRoad type distribution:")
            for road_type, count in sorted(results['road_type_distribution'].items()):
                pct = (count / results['total_features']) * 100
                logger.info(f"  {road_type:15s}: {count:8,} ({pct:5.1f}%)")
        
        if 'invalid_geometries' in results and results['invalid_geometries'] > 0:
            logger.warning(f"\n⚠ Warning: {results['invalid_geometries']} invalid geometries found")
        else:
            logger.info(f"\n✓ All geometries are valid")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing roads: {e}")
        import traceback
        traceback.print_exc()
        return {}


def analyze_trails(trails_path: Path) -> dict:
    """Analyze trails shapefile."""
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING TRAILS DATA")
    logger.info('='*60)
    
    if not trails_path.exists():
        logger.error(f"Trails file not found: {trails_path}")
        return {}
    
    try:
        gdf = gpd.read_file(trails_path)
        
        results = {
            'file_path': str(trails_path),
            'total_features': len(gdf),
            'crs': str(gdf.crs) if gdf.crs else 'None',
            'columns': list(gdf.columns),
            'geometry_types': gdf.geom_type.value_counts().to_dict(),
        }
        
        # Check bounding box
        bounds = gdf.total_bounds
        results['bounding_box'] = {
            'min_lon': bounds[0],
            'min_lat': bounds[1],
            'max_lon': bounds[2],
            'max_lat': bounds[3]
        }
        
        # Calculate total length
        if gdf.crs:
            gdf_utm = gdf.to_crs('EPSG:32612')  # UTM Zone 12N
            total_length_m = gdf_utm.geometry.length.sum()
            total_length_mi = total_length_m / 1609.34
            results['total_length_miles'] = total_length_mi
        
        # Print results
        logger.info(f"File: {trails_path}")
        logger.info(f"Total features: {results['total_features']:,}")
        logger.info(f"CRS: {results['crs']}")
        logger.info(f"Columns: {', '.join(results['columns'])}")
        logger.info(f"\nGeometry types:")
        for geom_type, count in results['geometry_types'].items():
            logger.info(f"  {geom_type}: {count:,}")
        
        logger.info(f"\nBounding box:")
        logger.info(f"  Longitude: {bounds[0]:.6f} to {bounds[2]:.6f}")
        logger.info(f"  Latitude: {bounds[1]:.6f} to {bounds[3]:.6f}")
        
        if 'total_length_miles' in results:
            logger.info(f"\nTotal trail length: {results['total_length_miles']:,.1f} miles")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing trails: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze roads and trails infrastructure data"
    )
    parser.add_argument(
        '--roads',
        type=str,
        default='data/infrastructure/roads.geojson',
        help='Path to roads GeoJSON file'
    )
    parser.add_argument(
        '--trails',
        type=str,
        help='Path to trails shapefile or GeoJSON file'
    )
    
    args = parser.parse_args()
    
    results = {}
    
    # Analyze roads
    if args.roads:
        results['roads'] = analyze_roads(Path(args.roads))
    
    # Analyze trails
    if args.trails:
        results['trails'] = analyze_trails(Path(args.trails))
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info('='*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


