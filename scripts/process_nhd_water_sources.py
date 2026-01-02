#!/usr/bin/env python3
"""
Process NHD (National Hydrography Dataset) water sources and convert to GeoJSON.

This script:
1. Loads NHD shapefiles (NHDFlowline, NHDWaterbody, etc.)
2. Filters for relevant water types
3. Clips to Wyoming bounds (if needed)
4. Converts to WGS84
5. Adds water type and reliability attributes
6. Saves as GeoJSON

Usage:
    python scripts/process_nhd_water_sources.py \
        --nhd-flowline path/to/NHDFlowline.shp \
        --nhd-waterbody path/to/NHDWaterbody.shp \
        --output data/hydrology/water_sources.geojson

Optional arguments:
    --nhd-area: Path to NHDArea.shp (large water areas)
    --nhd-spring: Path to NHDSpring.shp (springs and seeps)
    --no-clip: Do not clip to Wyoming bounds (clipping is enabled by default)
    --simplify: Simplify geometries to reduce file size (default: False)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from shapely.ops import unary_union
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

# NHD Feature Type codes for water features
# See: https://nhd.usgs.gov/userGuide/Robohelpfiles/NHD_User_Guide/Feature_Catalog/Feature_Catalog.htm
WATER_FEATURE_TYPES = {
    # Flowline (streams, rivers)
    460: 'stream',  # StreamRiver
    558: 'canal',   # CanalDitch
    336: 'stream',  # ArtificialPath
    334: 'stream',  # Connector
    
    # Waterbody (lakes, ponds)
    390: 'lake',    # LakePond
    436: 'reservoir',  # Reservoir
    
    # Area (large water areas)
    390: 'lake',    # LakePond (also in area)
    436: 'reservoir',  # Reservoir (also in area)
    
    # Spring
    388: 'spring',  # SpringSeep
}

# Reliability mapping based on water type
RELIABILITY_MAP = {
    'spring': 1.0,      # Permanent springs
    'lake': 1.0,        # Permanent lakes
    'reservoir': 1.0,    # Permanent reservoirs
    'stream': 0.7,      # Perennial streams (may be seasonal in some areas)
    'canal': 0.9,       # Managed canals (usually permanent)
}


def load_nhd_flowline(flowline_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and filter NHD flowline (streams, rivers)."""
    if not flowline_path.exists():
        logger.warning(f"NHDFlowline not found: {flowline_path}")
        return None
    
    logger.info(f"Loading NHDFlowline: {flowline_path}")
    gdf = gpd.read_file(flowline_path)
    
    # Filter for water features
    if 'FType' in gdf.columns:
        # Filter by FType code
        water_ftypes = [460, 558, 336, 334]  # StreamRiver, CanalDitch, ArtificialPath, Connector
        gdf = gdf[gdf['FType'].isin(water_ftypes)]
        logger.info(f"  Filtered to {len(gdf)} water flowlines")
    else:
        logger.warning("  No 'FType' column found, keeping all features")
    
    # Add water type
    if 'FType' in gdf.columns:
        gdf['water_type'] = gdf['FType'].map({
            460: 'stream',
            558: 'canal',
            336: 'stream',
            334: 'stream',
        }).fillna('stream')
    else:
        gdf['water_type'] = 'stream'
    
    # Add reliability
    gdf['reliability'] = gdf['water_type'].map(RELIABILITY_MAP).fillna(0.7)
    
    # Add name if available
    if 'GNIS_Name' in gdf.columns:
        gdf['name'] = gdf['GNIS_Name']
    elif 'GNIS_NAME' in gdf.columns:
        gdf['name'] = gdf['GNIS_NAME']
    else:
        gdf['name'] = None
    
    logger.info(f"  ✓ Loaded {len(gdf)} flowline features")
    return gdf


def load_nhd_waterbody(waterbody_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and filter NHD waterbody (lakes, ponds)."""
    if not waterbody_path.exists():
        logger.warning(f"NHDWaterbody not found: {waterbody_path}")
        return None
    
    logger.info(f"Loading NHDWaterbody: {waterbody_path}")
    gdf = gpd.read_file(waterbody_path)
    
    # Filter for water features
    if 'FType' in gdf.columns:
        # Filter by FType code
        water_ftypes = [390, 436]  # LakePond, Reservoir
        gdf = gdf[gdf['FType'].isin(water_ftypes)]
        logger.info(f"  Filtered to {len(gdf)} waterbodies")
    else:
        logger.warning("  No 'FType' column found, keeping all features")
    
    # Add water type
    if 'FType' in gdf.columns:
        gdf['water_type'] = gdf['FType'].map({
            390: 'lake',
            436: 'reservoir',
        }).fillna('lake')
    else:
        gdf['water_type'] = 'lake'
    
    # Add reliability
    gdf['reliability'] = gdf['water_type'].map(RELIABILITY_MAP).fillna(1.0)
    
    # Add name if available
    if 'GNIS_Name' in gdf.columns:
        gdf['name'] = gdf['GNIS_Name']
    elif 'GNIS_NAME' in gdf.columns:
        gdf['name'] = gdf['GNIS_NAME']
    else:
        gdf['name'] = None
    
    logger.info(f"  ✓ Loaded {len(gdf)} waterbody features")
    return gdf


def load_nhd_area(area_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and filter NHD area (large water areas)."""
    if not area_path.exists():
        logger.warning(f"NHDArea not found: {area_path}")
        return None
    
    logger.info(f"Loading NHDArea: {area_path}")
    gdf = gpd.read_file(area_path)
    
    # Filter for water features
    if 'FType' in gdf.columns:
        water_ftypes = [390, 436]  # LakePond, Reservoir
        gdf = gdf[gdf['FType'].isin(water_ftypes)]
        logger.info(f"  Filtered to {len(gdf)} water areas")
    else:
        logger.warning("  No 'FType' column found, keeping all features")
    
    # Add water type
    if 'FType' in gdf.columns:
        gdf['water_type'] = gdf['FType'].map({
            390: 'lake',
            436: 'reservoir',
        }).fillna('lake')
    else:
        gdf['water_type'] = 'lake'
    
    # Add reliability
    gdf['reliability'] = gdf['water_type'].map(RELIABILITY_MAP).fillna(1.0)
    
    # Add name if available
    if 'GNIS_Name' in gdf.columns:
        gdf['name'] = gdf['GNIS_Name']
    elif 'GNIS_NAME' in gdf.columns:
        gdf['name'] = gdf['GNIS_NAME']
    else:
        gdf['name'] = None
    
    logger.info(f"  ✓ Loaded {len(gdf)} area features")
    return gdf


def load_nhd_spring(spring_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and filter NHD spring (springs, seeps)."""
    if not spring_path.exists():
        logger.warning(f"NHDSpring not found: {spring_path}")
        return None
    
    logger.info(f"Loading NHDSpring: {spring_path}")
    gdf = gpd.read_file(spring_path)
    
    # Filter for spring features
    if 'FType' in gdf.columns:
        gdf = gdf[gdf['FType'] == 388]  # SpringSeep
        logger.info(f"  Filtered to {len(gdf)} springs")
    else:
        logger.warning("  No 'FType' column found, keeping all features")
    
    # Add water type
    gdf['water_type'] = 'spring'
    
    # Add reliability
    gdf['reliability'] = 1.0  # Springs are permanent
    
    # Add name if available
    if 'GNIS_Name' in gdf.columns:
        gdf['name'] = gdf['GNIS_Name']
    elif 'GNIS_NAME' in gdf.columns:
        gdf['name'] = gdf['GNIS_NAME']
    else:
        gdf['name'] = None
    
    logger.info(f"  ✓ Loaded {len(gdf)} spring features")
    return gdf


def clip_to_wyoming(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to Wyoming bounds."""
    logger.info("Clipping to Wyoming bounds...")
    
    # Check bounds before clipping
    bounds = gdf.total_bounds
    logger.info(f"  Data bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    logger.info(f"  Wyoming bounds: [-111.0, 41.0, -104.0, 45.0]")
    
    # Ensure CRS is WGS84
    if gdf.crs != 'EPSG:4326':
        logger.info(f"  Converting CRS from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
        # Re-check bounds after conversion
        bounds = gdf.total_bounds
        logger.info(f"  Bounds after CRS conversion: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    
    # Use spatial index for better performance
    from shapely.geometry import box
    
    # Create Wyoming bounding box
    wyoming_bbox = box(-111.0, 41.0, -104.0, 45.0)
    
    # Check if any features intersect Wyoming
    # First, check if bounds overlap at all
    data_bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    if not data_bbox.intersects(wyoming_bbox):
        logger.warning(f"  ⚠ Data bounds don't overlap Wyoming bounds!")
        logger.warning(f"     This might mean the data is from a different region.")
        logger.warning(f"     Skipping clipping - keeping all features.")
        return gdf
    
    # Clip to Wyoming bounds using spatial query
    # Use intersects for better performance with spatial index
    gdf_clipped = gdf[gdf.geometry.intersects(wyoming_bbox)]
    
    logger.info(f"  ✓ Clipped to {len(gdf_clipped)} features in Wyoming (from {len(gdf)} total)")
    
    if len(gdf_clipped) == 0 and len(gdf) > 0:
        logger.warning(f"  ⚠ WARNING: Clipping resulted in 0 features!")
        logger.warning(f"     This might indicate:")
        logger.warning(f"     1. Data is from outside Wyoming")
        logger.warning(f"     2. CRS transformation issue")
        logger.warning(f"     3. Bounds check is incorrect")
        logger.warning(f"     Keeping all features without clipping.")
        return gdf
    
    return gdf_clipped


def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float = 0.0001) -> gpd.GeoDataFrame:
    """Simplify geometries to reduce file size."""
    logger.info(f"Simplifying geometries (tolerance: {tolerance})...")
    
    gdf['geometry'] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    
    logger.info("  ✓ Geometries simplified")
    return gdf


def process_nhd_water_sources(
    flowline_path: Optional[Path],
    waterbody_path: Optional[Path],
    area_path: Optional[Path],
    spring_path: Optional[Path],
    output_path: Path,
    clip_wyoming: bool = True,
    simplify: bool = False,
    input_dir: Optional[Path] = None
):
    """
    Process NHD water sources and save as GeoJSON.
    
    Args:
        flowline_path: Path to single NHDFlowline.shp (or None if using input_dir)
        waterbody_path: Path to single NHDWaterbody.shp (or None if using input_dir)
        area_path: Path to single NHDArea.shp (or None if using input_dir)
        spring_path: Path to single NHDSpring.shp (or None if using input_dir)
        output_path: Output GeoJSON file path
        clip_wyoming: Clip to Wyoming bounds
        simplify: Simplify geometries to reduce file size
        input_dir: Directory containing multiple NHD shapefiles (alternative to individual paths)
    """
    
    logger.info("Processing NHD water sources...")
    
    # Load all NHD components
    gdfs = []
    
    # If input_dir is provided, find all shapefiles in that directory
    if input_dir and input_dir.exists():
        logger.info(f"Searching for NHD shapefiles in: {input_dir}")
        
        # Find all flowline files
        flowline_files = list(input_dir.rglob('NHDFlowline.shp'))
        if flowline_files:
            logger.info(f"  Found {len(flowline_files)} NHDFlowline file(s)")
            for f in flowline_files:
                gdf = load_nhd_flowline(f)
                if gdf is not None:
                    gdfs.append(gdf)
        
        # Find all waterbody files
        waterbody_files = list(input_dir.rglob('NHDWaterbody.shp'))
        if waterbody_files:
            logger.info(f"  Found {len(waterbody_files)} NHDWaterbody file(s)")
            for f in waterbody_files:
                gdf = load_nhd_waterbody(f)
                if gdf is not None:
                    gdfs.append(gdf)
        
        # Find all area files
        area_files = list(input_dir.rglob('NHDArea.shp'))
        if area_files:
            logger.info(f"  Found {len(area_files)} NHDArea file(s)")
            for f in area_files:
                gdf = load_nhd_area(f)
                if gdf is not None:
                    gdfs.append(gdf)
        
        # Find all spring files
        spring_files = list(input_dir.rglob('NHDSpring.shp'))
        if spring_files:
            logger.info(f"  Found {len(spring_files)} NHDSpring file(s)")
            for f in spring_files:
                gdf = load_nhd_spring(f)
                if gdf is not None:
                    gdfs.append(gdf)
    
    # Otherwise, use individual file paths
    else:
        if flowline_path:
            gdf = load_nhd_flowline(flowline_path)
            if gdf is not None:
                gdfs.append(gdf)
        
        if waterbody_path:
            gdf = load_nhd_waterbody(waterbody_path)
            if gdf is not None:
                gdfs.append(gdf)
        
        if area_path:
            gdf = load_nhd_area(area_path)
            if gdf is not None:
                gdfs.append(gdf)
        
        if spring_path:
            gdf = load_nhd_spring(spring_path)
            if gdf is not None:
                gdfs.append(gdf)
    
    if not gdfs:
        logger.error("No NHD data loaded. Check file paths.")
        return False
    
    # Combine all GeoDataFrames
    logger.info("Combining all water features...")
    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs
    )
    
    logger.info(f"  ✓ Combined {len(combined)} total features")
    
    # Clip to Wyoming if requested
    if clip_wyoming:
        combined = clip_to_wyoming(combined)
    
    # Simplify if requested
    if simplify:
        combined = simplify_geometries(combined)
    
    # Ensure CRS is WGS84
    if combined.crs != 'EPSG:4326':
        logger.info(f"Converting CRS from {combined.crs} to EPSG:4326")
        combined = combined.to_crs('EPSG:4326')
    
    # Select only necessary columns
    columns_to_keep = ['geometry', 'water_type', 'reliability', 'name']
    available_columns = [col for col in columns_to_keep if col in combined.columns]
    combined = combined[available_columns]
    
    # Save to GeoJSON
    logger.info(f"Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver='GeoJSON')
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Processing complete!")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total features: {len(combined):,}")
    
    if 'water_type' in combined.columns:
        logger.info("\nFeature types:")
        type_counts = combined['water_type'].value_counts()
        for water_type, count in type_counts.items():
            pct = (count / len(combined)) * 100
            logger.info(f"  {water_type}: {count:,} ({pct:.1f}%)")
    
    if 'reliability' in combined.columns:
        logger.info(f"\nReliability range: {combined['reliability'].min():.2f} to {combined['reliability'].max():.2f}")
        logger.info(f"Mean reliability: {combined['reliability'].mean():.2f}")
    
    logger.info("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Process NHD water sources and convert to GeoJSON'
    )
    parser.add_argument(
        '--nhd-flowline',
        type=Path,
        help='Path to NHDFlowline.shp'
    )
    parser.add_argument(
        '--nhd-waterbody',
        type=Path,
        help='Path to NHDWaterbody.shp'
    )
    parser.add_argument(
        '--nhd-area',
        type=Path,
        help='Path to NHDArea.shp (optional)'
    )
    parser.add_argument(
        '--nhd-spring',
        type=Path,
        help='Path to NHDSpring.shp (optional)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output GeoJSON file path'
    )
    parser.add_argument(
        '--clip-wyoming',
        action='store_true',
        default=False,
        help='Explicitly enable clipping to Wyoming bounds (default: True)'
    )
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Disable clipping to Wyoming bounds (overrides --clip-wyoming)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify geometries to reduce file size'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Directory containing multiple NHD shapefiles (alternative to individual --nhd-* paths)'
    )
    
    args = parser.parse_args()
    
    # Handle clip_wyoming flag logic:
    # - If --no-clip is passed, disable clipping (overrides --clip-wyoming)
    # - If --clip-wyoming is passed, enable clipping
    # - If neither is passed, default to True (clipping enabled)
    if args.no_clip:
        clip_wyoming = False
    elif args.clip_wyoming:
        clip_wyoming = True
    else:
        clip_wyoming = True  # Default: clipping enabled
    
    # Process water sources
    success = process_nhd_water_sources(
        flowline_path=args.nhd_flowline,
        waterbody_path=args.nhd_waterbody,
        area_path=args.nhd_area,
        spring_path=args.nhd_spring,
        output_path=args.output,
        clip_wyoming=clip_wyoming,
        simplify=args.simplify,
        input_dir=getattr(args, 'input_dir', None)
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

