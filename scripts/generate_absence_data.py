#!/usr/bin/env python3
"""
Generate balanced absence dataset for PathWild training.

Combines multiple strategies to create high-quality absence points
that complement existing presence data from elk GPS collars.

Usage:
    python scripts/generate_absence_data.py [--presence-file PATH] [--output-file PATH]
"""

import argparse
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import box
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.absence_generators import (
    EnvironmentalPseudoAbsenceGenerator,
    UnsuitableHabitatAbsenceGenerator,
    RandomBackgroundGenerator,
    TemporalAbsenceGenerator
)
from src.data.processors import DataContextBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_wyoming_bbox() -> gpd.GeoDataFrame:
    """
    Create Wyoming bounding box as fallback study area.
    
    Wyoming approximate bounds (WGS84):
    - Latitude: 41.0°N to 45.0°N
    - Longitude: -111.0°W to -104.0°W
    """
    wyoming_bbox = box(-111.0, 41.0, -104.0, 45.0)
    return gpd.GeoDataFrame(geometry=[wyoming_bbox], crs="EPSG:4326")


def load_study_area(data_dir: Path) -> gpd.GeoDataFrame:
    """
    Load study area boundary (Wyoming state or hunt area).
    
    Tries to load Wyoming boundary shapefile, falls back to bounding box.
    """
    # Try Wyoming state boundary
    wyoming_path = data_dir / "boundaries" / "wyoming_state.shp"
    if wyoming_path.exists():
        logger.info(f"Loading Wyoming boundary from {wyoming_path}")
        return gpd.read_file(wyoming_path)
    
    # Try hunt area as fallback
    hunt_area_path = data_dir / "raw" / "hunt_areas" / "Area_048.shp"
    if hunt_area_path.exists():
        logger.info(f"Using Area 048 as study area: {hunt_area_path}")
        return gpd.read_file(hunt_area_path)
    
    # Fallback to Wyoming bounding box
    logger.warning("No boundary file found, using Wyoming bounding box")
    return create_wyoming_bbox()


def validate_absence_data(
    presence_gdf: gpd.GeoDataFrame,
    absence_gdf: gpd.GeoDataFrame
) -> bool:
    """
    Validate that absence data meets quality requirements.
    
    Args:
        presence_gdf: GeoDataFrame with presence points
        absence_gdf: GeoDataFrame with absence points
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating absence data...")
    
    # Convert to UTM for distance calculations
    utm_crs = "EPSG:32613"
    presence_utm = presence_gdf.to_crs(utm_crs)
    absence_utm = absence_gdf.to_crs(utm_crs)
    
    # Check 1: Spatial separation
    min_distances = []
    for absence_point in absence_utm.geometry:
        distances = presence_utm.geometry.distance(absence_point)
        min_distances.append(distances.min())
    
    min_dist_array = np.array(min_distances)
    mean_dist = min_dist_array.mean()
    median_dist = np.median(min_dist_array)
    min_dist = min_dist_array.min()
    
    logger.info(f"Minimum distances from absences to presences:")
    logger.info(f"  Mean: {mean_dist:.0f}m")
    logger.info(f"  Median: {median_dist:.0f}m")
    logger.info(f"  Min: {min_dist:.0f}m")
    
    if mean_dist < 1000:
        logger.warning(f"Mean distance {mean_dist:.0f}m is less than 1000m threshold")
        return False
    
    # Check 2: Geographic coverage
    presence_bounds = presence_gdf.total_bounds
    absence_bounds = absence_gdf.total_bounds
    
    logger.info(f"\nGeographic coverage:")
    logger.info(f"  Presence bounds: {presence_bounds}")
    logger.info(f"  Absence bounds: {absence_bounds}")
    
    # Check 3: Class balance
    n_presence = len(presence_gdf)
    n_absence = len(absence_gdf)
    ratio = n_presence / n_absence if n_absence > 0 else 0
    
    logger.info(f"\nClass balance:")
    logger.info(f"  Presence: {n_presence:,}")
    logger.info(f"  Absence: {n_absence:,}")
    logger.info(f"  Ratio: {ratio:.2f}")
    
    if ratio < 0.5 or ratio > 2.0:
        logger.warning(f"Class ratio {ratio:.2f} is outside recommended range [0.5, 2.0]")
    
    logger.info("\n✓ Validation complete")
    return True


def enrich_with_features(
    df: pd.DataFrame,
    data_dir: Path,
    date_column: Optional[str] = None,
    default_date: str = "2024-10-15"
) -> pd.DataFrame:
    """
    Enrich dataframe with environmental features using DataContextBuilder.
    
    Args:
        df: DataFrame with latitude, longitude columns
        data_dir: Path to data directory
        date_column: Optional column name with dates
        default_date: Default date to use if date_column not available
        
    Returns:
        DataFrame with added environmental features
    """
    logger.info("Enriching with environmental features...")
    
    context_builder = DataContextBuilder(data_dir)
    
    enriched_rows = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processing {idx:,}/{len(df):,}")
        
        location = {"lat": row['latitude'], "lon": row['longitude']}
        
        # Use date from column if available, otherwise default
        if date_column and date_column in row and pd.notna(row[date_column]):
            date = str(pd.to_datetime(row[date_column]).date())
        else:
            date = default_date
        
        try:
            context = context_builder.build_context(location, date)
            
            # Create new row with all features
            new_row = row.to_dict()
            for key, value in context.items():
                # Skip geometry objects
                if key not in ['dem_grid', 'water_sources']:
                    new_row[key] = value
            
            enriched_rows.append(new_row)
        except Exception as e:
            logger.warning(f"Error enriching row {idx}: {e}")
            # Add row with defaults
            new_row = row.to_dict()
            enriched_rows.append(new_row)
    
    return pd.DataFrame(enriched_rows)


def main():
    """Main function to generate absence data."""
    parser = argparse.ArgumentParser(
        description="Generate absence data for PathWild training dataset"
    )
    parser.add_argument(
        '--presence-file',
        type=str,
        default='data/processed/south_bighorn_points.csv',
        help='Path to presence data CSV file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/processed/combined_presence_absence.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Ratio of absence to presence points (default: 1.0)'
    )
    parser.add_argument(
        '--skip-enrichment',
        action='store_true',
        help='Skip environmental feature enrichment (faster for testing)'
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: auto-detect, max 8)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of presence points to process (for testing). Reduces absence generation accordingly.'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    presence_file = Path(args.presence_file)
    output_file = Path(args.output_file)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load presence data
    logger.info(f"Loading presence data from {presence_file}")
    if not presence_file.exists():
        logger.error(f"Presence file not found: {presence_file}")
        return 1
    
    presence_df = pd.read_csv(presence_file)
    
    # Limit presence data if --limit is set (for testing)
    if args.limit is not None:
        original_count = len(presence_df)
        presence_df = presence_df.head(args.limit)
        logger.info(f"⚠️  TEST MODE: Limited presence data from {original_count:,} to {len(presence_df):,} points")
    
    logger.info(f"Loaded {len(presence_df):,} presence points")
    
    # Create GeoDataFrame
    presence_gdf = gpd.GeoDataFrame(
        presence_df,
        geometry=gpd.points_from_xy(
            presence_df.longitude,
            presence_df.latitude
        ),
        crs="EPSG:4326"
    )
    presence_gdf['elk_present'] = 1
    
    n_presence = len(presence_gdf)
    
    # Load study area
    study_area = load_study_area(data_dir)
    
    # Calculate absence targets
    n_total_absences = int(n_presence * args.ratio)
    n_environmental = int(n_total_absences * 0.40)
    n_unsuitable = int(n_total_absences * 0.30)
    n_background = int(n_total_absences * 0.20)
    n_temporal = int(n_total_absences * 0.10)
    
    logger.info(f"\nGenerating {n_total_absences:,} absence points:")
    logger.info(f"  Environmental pseudo-absences: {n_environmental:,}")
    logger.info(f"  Unsuitable habitat absences: {n_unsuitable:,}")
    logger.info(f"  Random background absences: {n_background:,}")
    logger.info(f"  Temporal absences: {n_temporal:,}")
    
    # Generate absences using each strategy
    all_absences = []
    
    # Environmental pseudo-absences
    if n_environmental > 0:
        env_gen = EnvironmentalPseudoAbsenceGenerator(
            presence_gdf,
            study_area,
            data_dir=data_dir
        )
        env_absences = env_gen.generate(n_environmental, n_processes=args.n_processes)
        all_absences.append(env_absences)
        logger.info(f"✓ Generated {len(env_absences):,} environmental absences")
    
    # Unsuitable habitat absences
    if n_unsuitable > 0:
        unsuit_gen = UnsuitableHabitatAbsenceGenerator(
            presence_gdf,
            study_area,
            data_dir=data_dir
        )
        unsuit_absences = unsuit_gen.generate(n_unsuitable, n_processes=args.n_processes)
        all_absences.append(unsuit_absences)
        logger.info(f"✓ Generated {len(unsuit_absences):,} unsuitable absences")
    
    # Random background absences
    if n_background > 0:
        bg_gen = RandomBackgroundGenerator(presence_gdf, study_area)
        bg_absences = bg_gen.generate(n_background, n_processes=args.n_processes)
        all_absences.append(bg_absences)
        logger.info(f"✓ Generated {len(bg_absences):,} background absences")
    
    # Temporal absences (if applicable)
    if n_temporal > 0:
        # Check if we have date information
        date_columns = ['firstdate', 'lastdate', 'date', 'timestamp']
        date_column = None
        for col in date_columns:
            if col in presence_df.columns:
                date_column = col
                break
        
        if date_column:
            # Use firstdate or lastdate as the date
            temp_gen = TemporalAbsenceGenerator(
                presence_gdf,
                study_area,
                date_column=date_column
            )
            temp_absences = temp_gen.generate(n_temporal)
            all_absences.append(temp_absences)
            logger.info(f"✓ Generated {len(temp_absences):,} temporal absences")
        else:
            logger.warning("No date column found, skipping temporal absences")
    
    # Combine all absences
    if all_absences:
        all_absences_gdf = pd.concat(all_absences, ignore_index=True)
        all_absences_gdf['elk_present'] = 0
    else:
        logger.error("No absences generated!")
        return 1
    
    # Validate
    validate_absence_data(presence_gdf, all_absences_gdf)
    
    # Combine presence + absence
    logger.info("\nCombining presence and absence data...")
    
    # Select common columns
    presence_cols = ['latitude', 'longitude', 'elk_present']
    absence_cols = ['latitude', 'longitude', 'elk_present']
    
    # Add absence_strategy if present (useful metadata)
    if 'absence_strategy' in all_absences_gdf.columns:
        absence_cols.append('absence_strategy')
    
    # Add any additional columns from presence data
    for col in presence_df.columns:
        if col not in ['latitude', 'longitude'] and col not in presence_cols:
            presence_cols.append(col)
    
    training_data = pd.concat([
        presence_gdf[presence_cols],
        all_absences_gdf[absence_cols]
    ], ignore_index=True)
    
    # Shuffle
    training_data = training_data.sample(
        frac=1.0,
        random_state=42
    ).reset_index(drop=True)
    
    # Enrich with environmental features
    if not args.skip_enrichment:
        training_data = enrich_with_features(
            training_data,
            data_dir,
            date_column=date_column if 'date_column' in locals() else None
        )
    else:
        logger.info("Skipping environmental feature enrichment")
    
    # Save
    logger.info(f"\nSaving combined dataset to {output_file}")
    training_data.to_csv(output_file, index=False)
    
    # Summary
    logger.info(f"\n✓ Dataset generation complete!")
    logger.info(f"  Total samples: {len(training_data):,}")
    logger.info(f"  Presence (elk=1): {(training_data.elk_present == 1).sum():,}")
    logger.info(f"  Absence (elk=0): {(training_data.elk_present == 0).sum():,}")
    
    n_pres = (training_data.elk_present == 1).sum()
    n_abs = (training_data.elk_present == 0).sum()
    if n_abs > 0:
        ratio = n_pres / n_abs
        logger.info(f"  Ratio: {ratio:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

