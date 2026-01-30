#!/usr/bin/env python3
"""
Process raw presence data files into standardized presence points CSV.

This script processes raw elk GPS collar data from various sources (shapefiles,
CSV files) and converts them into a standardized format for downstream processing.

Supported formats:
- Shapefiles (migration routes)
- CSV files (GPS collar points)

Usage:
    python scripts/process_raw_presence_data.py [--input-dir PATH] [--output-dir PATH] [--dataset NAME]
"""

import argparse
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_shapefile(input_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Process a shapefile containing migration routes into presence points.
    
    Converts coordinates to WGS84 (lat/lon) if the shapefile is in a different CRS.
    
    Args:
        input_path: Path to shapefile (.shp)
        dataset_name: Name of the dataset (for metadata)
    
    Returns:
        DataFrame with presence points in WGS84 (lat/lon)
    """
    logger.info(f"Processing shapefile: {input_path}")
    
    # Load shapefile
    gdf = gpd.read_file(input_path)
    logger.info(f"  Loaded {len(gdf):,} features")
    logger.info(f"  Original CRS: {gdf.crs}")
    
    # Transform to WGS84 (EPSG:4326) if not already in that CRS
    # This ensures we extract lat/lon coordinates, not UTM or other projected coordinates
    if gdf.crs is None:
        logger.warning("  Shapefile has no CRS defined. Assuming WGS84.")
        gdf.set_crs('EPSG:4326', inplace=True)
    elif gdf.crs != 'EPSG:4326':
        logger.info(f"  Transforming from {gdf.crs} to WGS84 (EPSG:4326)...")
        gdf = gdf.to_crs('EPSG:4326')
        logger.info("  ✓ Transformation complete")
    else:
        logger.info("  Shapefile already in WGS84 (EPSG:4326)")
    
    # Extract points from geometries
    # Shapefiles may contain LineStrings (routes) or Points
    points = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom.geom_type == 'Point':
            # Already a point - extract lat/lon (y/x in WGS84)
            point_data = {
                'latitude': geom.y,
                'longitude': geom.x,
                'route_id': idx
            }
            # Add all other columns
            for col in gdf.columns:
                if col != 'geometry':
                    point_data[col] = row[col]
            points.append(point_data)
        
        elif geom.geom_type in ['LineString', 'MultiLineString']:
            # Extract points from line
            if geom.geom_type == 'MultiLineString':
                coords = []
                for line in geom.geoms:
                    coords.extend(list(line.coords))
            else:
                coords = list(geom.coords)
            
            # Create a point for each coordinate
            # In WGS84, coords are (lon, lat) = (x, y)
            for i, (lon, lat) in enumerate(coords):
                point_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'route_id': idx,
                    'point_index': i
                }
                # Add all other columns
                for col in gdf.columns:
                    if col != 'geometry':
                        point_data[col] = row[col]
                points.append(point_data)
        
        else:
            logger.warning(f"  Skipping geometry type: {geom.geom_type}")
    
    df = pd.DataFrame(points)
    logger.info(f"  Extracted {len(df):,} presence points")
    
    # Validate coordinates are in valid lat/lon range
    invalid_coords = (
        (~df['latitude'].between(-90, 90)) |
        (~df['longitude'].between(-180, 180))
    ).sum()
    if invalid_coords > 0:
        logger.warning(f"  Found {invalid_coords:,} points with invalid lat/lon coordinates")
        logger.warning(f"    Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
        logger.warning(f"    Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    
    return df


def process_csv(input_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Process a CSV file containing GPS collar points.
    
    Supports both lat/lon and UTM (Easting/Northing) coordinate formats.
    If UTM coordinates are detected, they are converted to lat/lon.
    
    Args:
        input_path: Path to CSV file
        dataset_name: Name of the dataset (for metadata)
    
    Returns:
        DataFrame with presence points in lat/lon
    """
    logger.info(f"Processing CSV: {input_path}")
    
    # Load CSV
    df = pd.read_csv(input_path)
    logger.info(f"  Loaded {len(df):,} rows")
    
    # Check for UTM coordinates first (Easting/Northing)
    has_easting = 'Easting' in df.columns or 'easting' in df.columns
    has_northing = 'Northing' in df.columns or 'northing' in df.columns
    
    if has_easting and has_northing:
        # UTM coordinates detected - convert to lat/lon
        logger.info("  Detected UTM coordinates (Easting/Northing), converting to lat/lon")
        
        easting_col = 'Easting' if 'Easting' in df.columns else 'easting'
        northing_col = 'Northing' if 'Northing' in df.columns else 'northing'
        
        # Determine UTM zone (default to Zone 12N for Wyoming, EPSG:32612)
        # Could be enhanced to auto-detect based on coordinate values
        utm_zone = 'EPSG:32612'  # UTM Zone 12N (covers most of Wyoming)
        logger.info(f"  Using UTM zone: {utm_zone}")
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(utm_zone, 'EPSG:4326', always_xy=True)
            
            # Convert UTM to lat/lon
            lon, lat = transformer.transform(
                df[easting_col].values,
                df[northing_col].values
            )
            
            df['longitude'] = lon
            df['latitude'] = lat
            
            logger.info(f"  Converted {len(df):,} UTM coordinates to lat/lon")
            
        except ImportError:
            raise ImportError(
                "pyproj is required to convert UTM coordinates. "
                "Install with: conda install pyproj or pip install pyproj"
            )
        except Exception as e:
            raise ValueError(
                f"Error converting UTM coordinates to lat/lon: {e}. "
                f"Please check that Easting/Northing values are valid UTM coordinates."
            )
    
    else:
        # Standardize column names for lat/lon
        # Common variations: Lat/Latitude, Lon/Longitude, Long
        lat_cols = ['latitude', 'Lat', 'Latitude', 'LAT', 'y', 'Y']
        lon_cols = ['longitude', 'Longitude', 'Lon', 'LONG', 'Long', 'x', 'X']
        
        lat_col = None
        lon_col = None
        
        for col in lat_cols:
            if col in df.columns:
                lat_col = col
                break
        
        for col in lon_cols:
            if col in df.columns:
                lon_col = col
                break
        
        if lat_col is None or lon_col is None:
            raise ValueError(
                f"Could not find latitude/longitude or Easting/Northing columns in {input_path}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Rename to standard names
        df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
    
    # Validate coordinates
    valid_mask = (
        df['latitude'].between(-90, 90) &
        df['longitude'].between(-180, 180) &
        df['latitude'].notna() &
        df['longitude'].notna()
    )
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"  Found {invalid_count:,} rows with invalid coordinates, removing")
        df = df[valid_mask].copy()
    
    logger.info(f"  Processed {len(df):,} valid presence points")
    
    return df


def process_dataset(
    input_dir: Path,
    dataset_name: str,
    output_dir: Path,
    limit: Optional[int] = None
) -> Optional[Path]:
    """
    Process a single dataset from raw data to presence points.
    
    Args:
        input_dir: Directory containing raw data files
        dataset_name: Name of the dataset (e.g., 'northern_bighorn', 'national_refuge')
        output_dir: Directory to save processed presence points
    
    Returns:
        Path to output file if successful, None otherwise
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*70}")
    
    # Look for data files
    shapefiles = list(input_dir.glob("*.shp"))
    csv_files = list(input_dir.glob("*.csv"))
    
    if not shapefiles and not csv_files:
        logger.error(f"  No data files found in {input_dir}")
        return None
    
    # Prefer shapefile if both exist
    if shapefiles:
        input_file = shapefiles[0]
        logger.info(f"  Found shapefile: {input_file.name}")
        try:
            df = process_shapefile(input_file, dataset_name)
        except Exception as e:
            logger.error(f"  Error processing shapefile: {e}")
            return None
    elif csv_files:
        input_file = csv_files[0]
        logger.info(f"  Found CSV file: {input_file.name}")
        try:
            df = process_csv(input_file, dataset_name)
        except Exception as e:
            logger.error(f"  Error processing CSV: {e}")
            return None
    else:
        logger.error(f"  No supported files found")
        return None
    
    # Add dataset metadata
    df['dataset'] = dataset_name
    
    # Standardize date column: copy firstdate/lastdate to standard 'date' column
    # This ensures all presence data has a consistent 'date' column for downstream processing
    # while preserving original metadata columns (firstdate, lastdate) for reference
    if 'date' not in df.columns:
        # Look for alternative date column names (in priority order)
        date_columns = ['firstdate', 'lastdate', 'Date_Time_MST', 'DT', 'timestamp']
        date_col_found = None
        
        for col in date_columns:
            if col in df.columns:
                # Check if column has valid dates
                date_series = pd.to_datetime(df[col], errors='coerce')
                if date_series.notna().sum() > 0:
                    # Copy to standard 'date' column
                    df['date'] = date_series
                    date_col_found = col
                    logger.info(f"  Copied {col} to standard 'date' column ({date_series.notna().sum():,} valid dates)")
                    break
        
        if date_col_found is None:
            logger.warning("  No valid date column found - presence points will have missing temporal metadata")
    else:
        # Date column already exists - ensure it's datetime type
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df['date'].notna().sum()
        logger.info(f"  Standardized existing 'date' column ({valid_dates:,} valid dates)")
    
    # Ensure required columns exist
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"  Missing required columns: {missing_cols}")
        return None
    
    # Apply limit if specified (for testing)
    if limit is not None:
        original_count = len(df)
        df = df.head(limit)
        logger.info(f"  ⚠️  TEST MODE: Limited presence points from {original_count:,} to {len(df):,}")
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use _test suffix when limit is set
    if limit is not None:
        output_file = output_dir / f"{dataset_name}_points_test.csv"
    else:
        output_file = output_dir / f"{dataset_name}_points.csv"
    
    logger.info(f"  Saving {len(df):,} presence points to {output_file}")
    df.to_csv(output_file, index=False)
    
    logger.info(f"  ✓ Successfully processed {dataset_name}")
    logger.info(f"    Output: {output_file}")
    logger.info(f"    Points: {len(df):,}")
    
    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process raw presence data files into standardized presence points"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/raw'),
        help='Directory containing raw data subdirectories (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory to save processed presence points (default: data/processed)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Specific dataset to process (e.g., "northern_bighorn"). If not specified, processes all datasets.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of datasets to process. If not specified, auto-detects from input directory.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of presence points to process per dataset (for testing). Creates *_test.csv files instead of overwriting originals.'
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = args.datasets
    else:
        # Auto-detect datasets from subdirectories
        datasets = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith('elk_'):
                dataset_name = subdir.name.replace('elk_', '')
                datasets.append(dataset_name)
        
        if not datasets:
            logger.error(f"No elk dataset directories found in {input_dir}")
            logger.info(f"Expected directories like: elk_northern_bighorn/, elk_southern_bighorn/, etc.")
            return 1
    
    logger.info(f"Found {len(datasets)} dataset(s) to process: {datasets}")
    
    # Process each dataset
    processed = []
    failed = []
    
    for dataset_name in datasets:
        dataset_input_dir = input_dir / f"elk_{dataset_name}"
        
        if not dataset_input_dir.exists():
            logger.warning(f"  Dataset directory not found: {dataset_input_dir}")
            failed.append(dataset_name)
            continue
        
        output_file = process_dataset(dataset_input_dir, dataset_name, output_dir, limit=args.limit)
        
        if output_file:
            processed.append((dataset_name, output_file))
        else:
            failed.append(dataset_name)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed: {len(processed)}/{len(datasets)}")
    
    if processed:
        logger.info("\nProcessed datasets:")
        for dataset_name, output_file in processed:
            logger.info(f"  ✓ {dataset_name}: {output_file}")
    
    if failed:
        logger.warning("\nFailed datasets:")
        for dataset_name in failed:
            logger.warning(f"  ✗ {dataset_name}")
        # Return 0 if at least some datasets succeeded, 1 only if all failed
        if len(processed) == 0:
            return 1  # All datasets failed
        # Some succeeded, some failed - return 0 to allow pipeline to continue
        logger.info(f"\nNote: {len(processed)} dataset(s) processed successfully despite {len(failed)} failure(s)")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

