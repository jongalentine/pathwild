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

# Import enhanced temporal generators (optional - enabled via config)
try:
    from src.data.temporal_absence_generators import (
        TemporallyMatchedAbsenceGenerator,
        SeasonalSegregationAbsenceGenerator,
        UnsuitableTemporalEnvironmentalAbsenceGenerator,
        RandomTemporalBackgroundGenerator
    )
    TEMPORAL_GENERATORS_AVAILABLE = True
except ImportError as e:
    TEMPORAL_GENERATORS_AVAILABLE = False
    # Logger not available yet at import time, will log later
    _temporal_generators_error = e

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
    parser.add_argument(
        '--use-temporal-strategies',
        action='store_true',
        default=False,
        help='Use enhanced temporal absence strategies (all absences get temporal metadata). '
             'If not specified, temporal strategies are enabled by default if available.'
    )
    parser.add_argument(
        '--no-temporal-strategies',
        action='store_true',
        default=False,
        help='Disable enhanced temporal strategies and use legacy generators'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file with absence generation settings'
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
    
    # Determine if we should use parallel processing
    # Standardized threshold: sequential for datasets < 5000 rows
    # This matches the logic in integrate_environmental_features.py
    PARALLEL_THRESHOLD = 5000
    
    # Determine effective n_processes
    # If explicitly set to 1, always use sequential
    # Otherwise, use sequential for small datasets (< threshold)
    if args.n_processes == 1:
        effective_n_processes = 1
        logger.info("Using sequential processing (n_processes=1)")
    elif n_presence < PARALLEL_THRESHOLD:
        effective_n_processes = 1
        logger.info(f"Dataset size ({n_presence:,}) < {PARALLEL_THRESHOLD:,} rows - using sequential processing")
    else:
        effective_n_processes = args.n_processes
        if effective_n_processes is None or effective_n_processes > 1:
            logger.info(f"Dataset size ({n_presence:,}) >= {PARALLEL_THRESHOLD:,} rows - using parallel processing")
        else:
            logger.info("Using sequential processing")
    
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
    
    # Check if temporal strategies should be used
    # Default: enabled if available, unless explicitly disabled
    if args.no_temporal_strategies:
        use_temporal_strategies = False
        logger.info("Temporal strategies explicitly disabled - using legacy generators")
    elif args.use_temporal_strategies:
        use_temporal_strategies = True
        if not TEMPORAL_GENERATORS_AVAILABLE:
            logger.error("Enhanced temporal generators not available but --use-temporal-strategies was specified!")
            logger.debug(f"Import error: {_temporal_generators_error}")
            return 1
    else:
        # Default: enable if available
        use_temporal_strategies = TEMPORAL_GENERATORS_AVAILABLE
        if TEMPORAL_GENERATORS_AVAILABLE:
            logger.info("Using enhanced temporal strategies by default (available and enabled)")
        else:
            logger.info("Temporal strategies not available - using legacy generators")
            logger.debug(f"Import error: {_temporal_generators_error}")
    
    if use_temporal_strategies and TEMPORAL_GENERATORS_AVAILABLE:
        # Use enhanced temporal strategies (all absences get temporal metadata)
        logger.info("\n" + "="*60)
        logger.info("USING ENHANCED TEMPORAL STRATEGIES")
        logger.info("="*60)
        logger.info("All absence points will include complete temporal metadata (date, year, month, day_of_year)")
        
        all_absences = []
        
        # Strategy 1: Temporally-Matched Environmental Absences (40%)
        if n_environmental > 0:
            logger.info(f"\nStrategy 1: Temporally-Matched Environmental ({n_environmental:,} points)")
            temp_matched_gen = TemporallyMatchedAbsenceGenerator(
                presence_gdf,
                study_area,
                data_dir=data_dir,
                min_distance_meters=2000.0
            )
            env_absences = temp_matched_gen.generate(n_environmental)
            all_absences.append(env_absences)
        
        # Strategy 2: Seasonal Segregation Absences (30%)
        if n_unsuitable > 0:
            logger.info(f"\nStrategy 2: Seasonal Segregation ({n_unsuitable:,} points)")
            # Check if we have date information
            date_columns = ['firstdate', 'lastdate', 'date', 'timestamp', 'DT']
            date_column = None
            for col in date_columns:
                if col in presence_df.columns:
                    date_column = col
                    break
            
            if date_column:
                seasonal_gen = SeasonalSegregationAbsenceGenerator(
                    presence_gdf,
                    study_area,
                    date_column=date_column,
                    offset_months=6  # Default 6 months for opposite season
                )
                seasonal_absences = seasonal_gen.generate(n_unsuitable)
                all_absences.append(seasonal_absences)
            else:
                logger.warning("No date column found for seasonal segregation, skipping")
                # Fallback to unsuitable habitat generator
                unsuit_gen = UnsuitableHabitatAbsenceGenerator(
                    presence_gdf,
                    study_area,
                    data_dir=data_dir
                )
                unsuit_absences = unsuit_gen.generate(n_unsuitable, n_processes=effective_n_processes)
                # Add temporal metadata
                unsuit_absences = unsuit_gen._add_temporal_metadata(unsuit_absences, 'unsuitable')
                all_absences.append(unsuit_absences)
        
        # Strategy 3: Unsuitable Temporal-Environmental Absences (20%)
        if n_background > 0:
            logger.info(f"\nStrategy 3: Unsuitable Temporal-Environmental ({n_background:,} points)")
            unsuitable_temp_gen = UnsuitableTemporalEnvironmentalAbsenceGenerator(
                presence_gdf,
                study_area,
                data_dir=data_dir
            )
            unsuitable_temp_absences = unsuitable_temp_gen.generate(n_background)
            all_absences.append(unsuitable_temp_absences)
        
        # Strategy 4: Random Temporal Background (10%)
        if n_temporal > 0:
            logger.info(f"\nStrategy 4: Random Temporal Background ({n_temporal:,} points)")
            random_temp_gen = RandomTemporalBackgroundGenerator(
                presence_gdf,
                study_area,
                min_distance_meters=500.0
            )
            random_temp_absences = random_temp_gen.generate(n_temporal)
            all_absences.append(random_temp_absences)
    
    else:
        # Use legacy generators (backward compatibility)
        logger.info("\nUsing legacy absence generators (backward compatibility mode)")
        logger.info("Note: Legacy generators may not include temporal metadata for all absences")
        logger.info("      Use --use-temporal-strategies flag to enable enhanced temporal strategies")
        
        all_absences = []
        
        # Environmental pseudo-absences
        if n_environmental > 0:
            env_gen = EnvironmentalPseudoAbsenceGenerator(
                presence_gdf,
                study_area,
                data_dir=data_dir
            )
            env_absences = env_gen.generate(n_environmental, n_processes=effective_n_processes)
            # Add temporal metadata if available
            env_absences = env_gen._add_temporal_metadata(env_absences, 'environmental')
            all_absences.append(env_absences)
            logger.info(f"✓ Generated {len(env_absences):,} environmental absences")
        
        # Unsuitable habitat absences
        if n_unsuitable > 0:
            unsuit_gen = UnsuitableHabitatAbsenceGenerator(
                presence_gdf,
                study_area,
                data_dir=data_dir
            )
            unsuit_absences = unsuit_gen.generate(n_unsuitable, n_processes=effective_n_processes)
            # Add temporal metadata if available
            unsuit_absences = unsuit_gen._add_temporal_metadata(unsuit_absences, 'unsuitable')
            all_absences.append(unsuit_absences)
            logger.info(f"✓ Generated {len(unsuit_absences):,} unsuitable absences")
        
        # Random background absences
        if n_background > 0:
            bg_gen = RandomBackgroundGenerator(presence_gdf, study_area)
            bg_absences = bg_gen.generate(n_background, n_processes=effective_n_processes)
            # Add temporal metadata if available
            bg_absences = bg_gen._add_temporal_metadata(bg_absences, 'background')
            all_absences.append(bg_absences)
            logger.info(f"✓ Generated {len(bg_absences):,} background absences")
        
        # Temporal absences (if applicable)
        if n_temporal > 0:
            # Check if we have date information
            date_columns = ['firstdate', 'lastdate', 'date', 'timestamp', 'DT']
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
        # Ensure all are GeoDataFrames
        absences_gdfs = []
        for absence in all_absences:
            if isinstance(absence, gpd.GeoDataFrame):
                absences_gdfs.append(absence)
            else:
                # Convert DataFrame to GeoDataFrame
                if 'geometry' not in absence.columns:
                    absence_gdf = gpd.GeoDataFrame(
                        absence,
                        geometry=gpd.points_from_xy(absence.longitude, absence.latitude),
                        crs="EPSG:4326"
                    )
                else:
                    absence_gdf = gpd.GeoDataFrame(absence, crs="EPSG:4326")
                absences_gdfs.append(absence_gdf)
        
        all_absences_gdf = pd.concat(absences_gdfs, ignore_index=True)
        
        # Ensure elk_present is set
        if 'elk_present' not in all_absences_gdf.columns:
            all_absences_gdf['elk_present'] = 0
        else:
            all_absences_gdf['elk_present'] = 0  # Overwrite to ensure all are 0
        
        logger.info(f"\n✓ Combined {len(all_absences_gdf):,} total absence points")
        
        # Log temporal metadata completeness
        if 'month' in all_absences_gdf.columns:
            missing_month = all_absences_gdf['month'].isna().sum()
            logger.info(f"  Temporal metadata completeness:")
            logger.info(f"    Missing month: {missing_month:,} ({missing_month/len(all_absences_gdf)*100:.1f}%)")
        if 'year' in all_absences_gdf.columns:
            missing_year = all_absences_gdf['year'].isna().sum()
            logger.info(f"    Missing year: {missing_year:,} ({missing_year/len(all_absences_gdf)*100:.1f}%)")
        if 'date' in all_absences_gdf.columns:
            missing_date = all_absences_gdf['date'].isna().sum()
            logger.info(f"    Missing date: {missing_date:,} ({missing_date/len(all_absences_gdf)*100:.1f}%)")
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
    
    # Add temporal columns if present (CRITICAL for model accuracy)
    temporal_cols = ['date', 'year', 'month', 'day_of_year', 'season']
    for col in temporal_cols:
        if col in all_absences_gdf.columns:
            absence_cols.append(col)
        if col in presence_df.columns:
            presence_cols.append(col)
    
    # Add dataset if present
    if 'dataset' in all_absences_gdf.columns:
        absence_cols.append('dataset')
    if 'dataset' in presence_df.columns:
        presence_cols.append('dataset')
    
    # Add any additional columns from presence data (but not geometry columns)
    for col in presence_df.columns:
        if col not in ['latitude', 'longitude', 'geometry'] and col not in presence_cols:
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
    
    # Note: Environmental feature enrichment is now handled by integrate_environmental_features.py
    # This avoids duplication and reduces processing time
    
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

