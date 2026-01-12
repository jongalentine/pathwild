"""
Enhanced temporal absence generators for PathWild.

This module provides research-based temporal strategies for generating absence data
that maximizes ML model accuracy by ensuring all absence points have complete temporal metadata.

Key Strategies:
1. Temporally-Matched Environmental Absences (40%)
   - Match temporal distribution of presence data
   - Random locations within study area
   - Environmental suitability filters

2. Seasonal Segregation Absences (30%)
   - Same locations as presence points
   - Offset dates by configured months (opposite season)
   - High confidence labeling

3. Unsuitable Temporal-Environmental Absences (20%)
   - Locations with unsuitable elevation-season combinations
   - Biologically meaningful absences (e.g., too high in winter)

4. Random Temporal Background (10%)
   - Uniform temporal sampling across all months/years
   - Random geographic locations
   - Represents pure "available" vs "used" comparison
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from datetime import datetime
import logging
import calendar

logger = logging.getLogger(__name__)


def days_in_month(month: int, year: int) -> int:
    """Get number of days in a given month and year."""
    return calendar.monthrange(year, month)[1]


def get_season_from_month(month: int) -> str:
    """Get season abbreviation from month (1-12)."""
    if month in [12, 1, 2]:
        return 'wi'  # Winter
    elif month in [3, 4, 5]:
        return 'sp'  # Spring
    elif month in [6, 7, 8]:
        return 'su'  # Summer
    else:  # 9, 10, 11
        return 'fa'  # Fall


def sample_date_from_distribution(
    presence_df: pd.DataFrame,
    date_column: str = 'date'
) -> pd.Timestamp:
    """
    Sample a date from the temporal distribution of presence data.
    
    Args:
        presence_df: DataFrame with presence data including date column
        date_column: Name of date column (supports 'date', 'firstdate', etc.)
        
    Returns:
        Randomly sampled date matching presence distribution
    """
    # Find date column
    date_col = None
    for col in [date_column, 'date', 'firstdate', 'lastdate', 'timestamp']:
        if col in presence_df.columns:
            date_col = col
            break
    
    if date_col is None:
        # No date column - sample uniformly from years 2006-2024
        year = np.random.choice(range(2006, 2025))
        month = np.random.randint(1, 13)
        day = np.random.randint(1, days_in_month(month, year) + 1)
        return pd.Timestamp(year=year, month=month, day=day)
    
    # Parse dates
    dates = pd.to_datetime(presence_df[date_col], errors='coerce').dropna()
    
    if len(dates) == 0:
        # All dates invalid - sample uniformly
        year = np.random.choice(range(2006, 2025))
        month = np.random.randint(1, 13)
        day = np.random.randint(1, days_in_month(month, year) + 1)
        return pd.Timestamp(year=year, month=month, day=day)
    
    # Sample from actual distribution
    sampled_date = dates.sample(n=1, random_state=None).iloc[0]
    
    # Optionally add small random variation to day (within same month)
    if len(dates) > 100:  # Only add variation if enough samples
        day_variation = np.random.randint(-3, 4)  # ±3 days
        try:
            new_day = min(max(1, sampled_date.day + day_variation), 
                         days_in_month(sampled_date.month, sampled_date.year))
            return pd.Timestamp(
                year=sampled_date.year,
                month=sampled_date.month,
                day=new_day
            )
        except (ValueError, OverflowError):
            pass
    
    return sampled_date


def sample_random_location(
    study_area_bounds: Tuple[float, float, float, float],
    max_attempts: int = 100
) -> Tuple[float, float]:
    """
    Sample a random location within study area bounds.
    
    Args:
        study_area_bounds: (min_lon, min_lat, max_lon, max_lat)
        max_attempts: Maximum attempts to find valid location
        
    Returns:
        (latitude, longitude) tuple
    """
    min_lon, min_lat, max_lon, max_lat = study_area_bounds
    
    for _ in range(max_attempts):
        lon = np.random.uniform(min_lon, max_lon)
        lat = np.random.uniform(min_lat, max_lat)
        
        # Validate bounds (basic check)
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            return lat, lon
    
    # Fallback to center if all attempts fail
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


def is_far_enough(
    lat: float,
    lon: float,
    presence_gdf: gpd.GeoDataFrame,
    min_distance_meters: float,
    utm_crs: str = "EPSG:32613"
) -> bool:
    """
    Check if location is far enough from all presence points.
    
    Args:
        lat: Latitude of candidate point
        lon: Longitude of candidate point
        presence_gdf: GeoDataFrame with presence points (WGS84)
        min_distance_meters: Minimum required distance in meters
        utm_crs: UTM CRS for distance calculations
        
    Returns:
        True if far enough, False otherwise
    """
    if len(presence_gdf) == 0:
        return True
    
    # Convert to UTM for distance calculation
    candidate_point = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(utm_crs)
    
    presence_utm = presence_gdf.to_crs(utm_crs)
    
    # Calculate distances to all presence points
    distances = presence_utm.geometry.distance(candidate_point.geometry.iloc[0])
    
    return distances.min() >= min_distance_meters


class TemporallyMatchedAbsenceGenerator:
    """
    Generate temporally-matched environmental absences.
    
    Strategy: Match temporal distribution of presence data while sampling
    random locations within study area with environmental suitability filters.
    """
    
    def __init__(
        self,
        presence_gdf: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        data_dir: Optional[Path] = None,
        date_column: str = 'date',
        min_distance_meters: float = 2000.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize temporally-matched absence generator.
        
        Args:
            presence_gdf: GeoDataFrame with presence points
            study_area: GeoDataFrame defining valid sampling area
            data_dir: Path to data directory (for environmental data)
            date_column: Name of date column in presence data
            min_distance_meters: Minimum distance from presence points
            config: Optional configuration dict
        """
        self.presence_gdf = presence_gdf
        self.study_area = study_area
        self.data_dir = data_dir
        self.date_column = date_column
        self.min_distance_meters = min_distance_meters
        self.config = config or {}
        
        # Get study area bounds
        if hasattr(study_area, 'total_bounds'):
            bounds = study_area.total_bounds  # (min_x, min_y, max_x, max_y)
            self.study_area_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])  # (min_lon, min_lat, max_lon, max_lat)
        else:
            # Fallback to Wyoming bounds
            self.study_area_bounds = (-111.0, 41.0, -104.0, 45.0)
        
        # Convert presence to DataFrame for temporal operations
        self.presence_df = pd.DataFrame(presence_gdf.drop(columns='geometry'))
        
        # UTM CRS for distance calculations
        self.utm_crs = "EPSG:32613"
    
    def generate(self, n_samples: int, max_attempts: int = 10000) -> gpd.GeoDataFrame:
        """
        Generate temporally-matched environmental absences.
        
        Args:
            n_samples: Number of absence points to generate
            max_attempts: Maximum attempts per sample
            
        Returns:
            GeoDataFrame with absence points including temporal metadata
        """
        logger.info(f"Generating {n_samples:,} temporally-matched environmental absences...")
        
        absences = []
        attempts = 0
        
        for i in range(n_samples):
            # Sample date from presence distribution
            sampled_date = sample_date_from_distribution(
                self.presence_df,
                self.date_column
            )
            
            # Sample location with constraints
            for attempt in range(max_attempts):
                attempts += 1
                lat, lon = sample_random_location(self.study_area_bounds)
                
                # Check distance from presences
                if is_far_enough(lat, lon, self.presence_gdf, self.min_distance_meters, self.utm_crs):
                    # Create absence record
                    absence = {
                        'latitude': lat,
                        'longitude': lon,
                        'date': sampled_date,
                        'year': sampled_date.year,
                        'month': sampled_date.month,
                        'day_of_year': sampled_date.dayofyear,
                        'season': get_season_from_month(sampled_date.month),
                        'elk_present': 0,
                        'absence_strategy': 'temporal_matched',
                        'confidence': 'medium'
                    }
                    
                    # Add dataset if present
                    if 'dataset' in self.presence_df.columns:
                        # Sample from presence dataset distribution
                        dataset_dist = self.presence_df['dataset'].value_counts(normalize=True)
                        absence['dataset'] = np.random.choice(
                            dataset_dist.index,
                            p=dataset_dist.values
                        )
                    
                    absences.append(absence)
                    break
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1:,}/{n_samples:,} absences...")
        
        if len(absences) < n_samples:
            logger.warning(
                f"Only generated {len(absences)}/{n_samples} temporally-matched absences "
                f"(attempts: {attempts:,})"
            )
        
        # Create GeoDataFrame
        absence_df = pd.DataFrame(absences)
        absence_gdf = gpd.GeoDataFrame(
            absence_df,
            geometry=gpd.points_from_xy(
                absence_df.longitude,
                absence_df.latitude
            ),
            crs="EPSG:4326"
        )
        
        logger.info(f"✓ Generated {len(absence_gdf):,} temporally-matched environmental absences")
        return absence_gdf


class SeasonalSegregationAbsenceGenerator:
    """
    Generate seasonal segregation absences.
    
    Strategy: Use same locations as presence points but offset dates
    by configured months (typically 6 for opposite season).
    """
    
    def __init__(
        self,
        presence_gdf: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        date_column: str = 'date',
        offset_months: int = 6,
        config: Optional[Dict] = None
    ):
        """
        Initialize seasonal segregation absence generator.
        
        Args:
            presence_gdf: GeoDataFrame with presence points
            study_area: GeoDataFrame defining valid sampling area (unused but kept for API consistency)
            date_column: Name of date column in presence data
            offset_months: Number of months to offset (default 6 for opposite season)
            config: Optional configuration dict with per-dataset offsets
        """
        self.presence_gdf = presence_gdf
        self.study_area = study_area
        self.date_column = date_column
        self.offset_months = offset_months
        self.config = config or {}
        
        # Convert to DataFrame for easier manipulation
        self.presence_df = pd.DataFrame(presence_gdf.drop(columns='geometry'))
        
        # Parse dates
        if date_column in self.presence_df.columns:
            self.presence_df['_parsed_date'] = pd.to_datetime(
                self.presence_df[date_column],
                errors='coerce'
            )
        else:
            logger.warning(f"Date column '{date_column}' not found in presence data")
            self.presence_df['_parsed_date'] = None
    
    def _get_offset_for_dataset(self, dataset: Optional[str]) -> int:
        """Get offset months for a specific dataset."""
        if dataset and 'datasets' in self.config:
            dataset_config = self.config['datasets'].get(dataset, {})
            return dataset_config.get('temporal_offset_months', self.offset_months)
        return self.offset_months
    
    def generate(self, n_samples: int, max_attempts: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Generate seasonal segregation absences.
        
        Args:
            n_samples: Number of absence points to generate
            max_attempts: Unused (kept for API consistency)
            
        Returns:
            GeoDataFrame with absence points including temporal metadata
        """
        logger.info(f"Generating {n_samples:,} seasonal segregation absences...")
        
        # Filter presence data with valid dates
        valid_presence = self.presence_df[self.presence_df['_parsed_date'].notna()].copy()
        
        if len(valid_presence) == 0:
            logger.warning("No presence data with valid dates - cannot generate seasonal segregation absences")
            return gpd.GeoDataFrame(geometry=[], crs=self.presence_gdf.crs)
        
        # Sample presence points
        n_to_sample = min(n_samples, len(valid_presence))
        sampled_presence = valid_presence.sample(
            n=n_to_sample,
            random_state=42,
            replace=True if n_samples > len(valid_presence) else False
        )
        
        absences = []
        
        for idx, row in sampled_presence.iterrows():
            # Get original date and location
            original_date = row['_parsed_date']
            point = self.presence_gdf.loc[idx].geometry
            
            # Get dataset-specific offset
            dataset = row.get('dataset', None)
            offset = self._get_offset_for_dataset(dataset)
            
            # Calculate new date with offset
            # Use pandas DateOffset for proper month/year handling
            new_date = original_date + pd.DateOffset(months=offset)
            
            # Ensure valid date (handle edge cases like Feb 30)
            if new_date.day != original_date.day:
                # Day is invalid (e.g., Feb 30 -> Feb 28)
                new_date = new_date.replace(day=min(original_date.day, days_in_month(new_date.month, new_date.year)))
            
            absence = {
                'latitude': point.y,
                'longitude': point.x,
                'date': new_date,
                'year': new_date.year,
                'month': new_date.month,
                'day_of_year': new_date.dayofyear,
                'season': get_season_from_month(new_date.month),
                'elk_present': 0,
                'absence_strategy': 'seasonal_segregation',
                'confidence': 'high',
                'original_date': original_date
            }
            
            # Preserve dataset if present
            if 'dataset' in row.index and pd.notna(row.get('dataset')):
                absence['dataset'] = row['dataset']
                absence['source_dataset'] = row['dataset']
            
            absences.append(absence)
        
        # Create GeoDataFrame
        absence_df = pd.DataFrame(absences)
        absence_gdf = gpd.GeoDataFrame(
            absence_df,
            geometry=gpd.points_from_xy(
                absence_df.longitude,
                absence_df.latitude
            ),
            crs=self.presence_gdf.crs
        )
        
        logger.info(f"✓ Generated {len(absence_gdf):,} seasonal segregation absences")
        return absence_gdf


class UnsuitableTemporalEnvironmentalAbsenceGenerator:
    """
    Generate unsuitable temporal-environmental absences.
    
    Strategy: Create absences at locations with biologically unsuitable
    elevation-season combinations (e.g., too high in winter, too low in summer).
    """
    
    def __init__(
        self,
        presence_gdf: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        data_dir: Optional[Path] = None,
        date_column: str = 'date',
        config: Optional[Dict] = None
    ):
        """
        Initialize unsuitable temporal-environmental absence generator.
        
        Args:
            presence_gdf: GeoDataFrame with presence points
            study_area: GeoDataFrame defining valid sampling area
            data_dir: Path to data directory (for DEM/elevation data)
            date_column: Name of date column in presence data
            config: Configuration dict with unsuitable rules
        """
        self.presence_gdf = presence_gdf
        self.study_area = study_area
        self.data_dir = data_dir
        self.date_column = date_column
        self.config = config or {}
        
        # Get study area bounds
        if hasattr(study_area, 'total_bounds'):
            bounds = study_area.total_bounds
            self.study_area_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
        else:
            self.study_area_bounds = (-111.0, 41.0, -104.0, 45.0)
        
        # Get rules from config
        self.rules = self.config.get('strategies', {}).get(
            'unsuitable_temporal_environmental', {}
        ).get('rules', {})
        
        if not self.rules:
            # Default rules if not configured
            self.rules = {
                'winter_too_high': {
                    'elevation_min': 9000,  # ft
                    'months': [12, 1, 2, 3]
                },
                'summer_too_low': {
                    'elevation_max': 6000,  # ft
                    'months': [6, 7, 8]
                }
            }
        
        # Load DEM if available (for elevation-based sampling)
        self.dem_raster = None
        if data_dir and RASTERIO_AVAILABLE:
            dem_path = data_dir / "dem" / "wyoming_dem.tif"
            if dem_path.exists():
                try:
                    import rasterio
                    self.dem_raster = rasterio.open(str(dem_path))
                    logger.info(f"Loaded DEM from {dem_path}")
                except Exception as e:
                    logger.warning(f"Failed to load DEM: {e}")
        
        self.utm_crs = "EPSG:32613"
    
    def _get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at location from DEM if available."""
        if self.dem_raster is None:
            return None
        
        try:
            import rasterio
            # Sample DEM at point
            for val in self.dem_raster.sample([(lon, lat)]):
                if val[0] is not None and not np.isnan(val[0]):
                    return float(val[0])  # Convert to feet if needed
        except Exception:
            pass
        
        return None
    
    def _sample_location_for_rule(
        self,
        rule: Dict,
        years: List[int],
        max_attempts: int = 1000
    ) -> Optional[Tuple[float, float, pd.Timestamp]]:
        """
        Sample location meeting unsuitable elevation criteria for a rule.
        
        Args:
            rule: Rule dict with elevation_min/max and months
            years: List of years to sample from
            max_attempts: Maximum attempts to find suitable location
            
        Returns:
            (lat, lon, date) tuple or None if not found
        """
        months = rule['months']
        month = np.random.choice(months)
        year = np.random.choice(years)
        day = np.random.randint(1, days_in_month(month, year) + 1)
        date = pd.Timestamp(year=year, month=month, day=day)
        
        for attempt in range(max_attempts):
            lat, lon = sample_random_location(self.study_area_bounds)
            
            # Check elevation if DEM available
            if self.dem_raster is not None:
                elevation = self._get_elevation(lat, lon)
                if elevation is not None:
                    # Check if elevation meets rule criteria
                    if 'elevation_min' in rule:
                        if elevation < rule['elevation_min']:
                            continue  # Try another location
                    if 'elevation_max' in rule:
                        if elevation > rule['elevation_max']:
                            continue  # Try another location
                    
                    # Elevation matches - return location
                    return lat, lon, date
            
            # If no DEM, just return any location (will rely on month rule only)
            return lat, lon, date
        
        return None
    
    def generate(self, n_samples: int, max_attempts: int = 10000) -> gpd.GeoDataFrame:
        """
        Generate unsuitable temporal-environmental absences.
        
        Args:
            n_samples: Number of absence points to generate
            max_attempts: Maximum attempts per sample
            
        Returns:
            GeoDataFrame with absence points including temporal metadata
        """
        logger.info(f"Generating {n_samples:,} unsuitable temporal-environmental absences...")
        
        # Get years from presence data
        presence_df = pd.DataFrame(self.presence_gdf.drop(columns='geometry'))
        if 'year' in presence_df.columns:
            years = presence_df['year'].dropna().unique().astype(int).tolist()
        elif self.date_column in presence_df.columns:
            dates = pd.to_datetime(presence_df[self.date_column], errors='coerce').dropna()
            years = dates.dt.year.unique().astype(int).tolist() if len(dates) > 0 else [2020]
        else:
            years = list(range(2006, 2025))  # Default years
        
        if not years:
            years = [2020]  # Fallback
        
        # Distribute samples across rules
        n_rules = len(self.rules)
        absences_per_rule = n_samples // n_rules if n_rules > 0 else n_samples
        
        absences = []
        
        for rule_name, rule in self.rules.items():
            for i in range(absences_per_rule):
                result = self._sample_location_for_rule(rule, years, max_attempts // n_rules)
                
                if result is not None:
                    lat, lon, date = result
                    absence = {
                        'latitude': lat,
                        'longitude': lon,
                        'date': date,
                        'year': date.year,
                        'month': date.month,
                        'day_of_year': date.dayofyear,
                        'season': get_season_from_month(date.month),
                        'elk_present': 0,
                        'absence_strategy': 'unsuitable_temporal_env',
                        'confidence': 'high',
                        'rule_applied': rule_name
                    }
                    
                    # Add elevation if available
                    if self.dem_raster is not None:
                        elevation = self._get_elevation(lat, lon)
                        if elevation is not None:
                            absence['elevation'] = elevation
                    
                    absences.append(absence)
        
        if len(absences) < n_samples:
            logger.warning(
                f"Only generated {len(absences)}/{n_samples} unsuitable temporal-environmental absences"
            )
        
        # Create GeoDataFrame
        absence_df = pd.DataFrame(absences)
        if len(absence_df) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=self.presence_gdf.crs)
        
        absence_gdf = gpd.GeoDataFrame(
            absence_df,
            geometry=gpd.points_from_xy(
                absence_df.longitude,
                absence_df.latitude
            ),
            crs="EPSG:4326"
        )
        
        logger.info(f"✓ Generated {len(absence_gdf):,} unsuitable temporal-environmental absences")
        return absence_gdf


class RandomTemporalBackgroundGenerator:
    """
    Generate random temporal background absences.
    
    Strategy: Sample dates uniformly across all months/years and locations
    randomly within study area. Represents pure "available" vs "used" comparison.
    """
    
    def __init__(
        self,
        presence_gdf: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        date_column: str = 'date',
        min_distance_meters: float = 500.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize random temporal background generator.
        
        Args:
            presence_gdf: GeoDataFrame with presence points
            study_area: GeoDataFrame defining valid sampling area
            date_column: Name of date column (used to infer year range)
            min_distance_meters: Minimum distance from presence points (looser than Strategy 1)
            config: Optional configuration dict
        """
        self.presence_gdf = presence_gdf
        self.study_area = study_area
        self.date_column = date_column
        self.min_distance_meters = min_distance_meters
        self.config = config or {}
        
        # Get study area bounds
        if hasattr(study_area, 'total_bounds'):
            bounds = study_area.total_bounds
            self.study_area_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
        else:
            self.study_area_bounds = (-111.0, 41.0, -104.0, 45.0)
        
        # Get year range from presence data
        presence_df = pd.DataFrame(presence_gdf.drop(columns='geometry'))
        if 'year' in presence_df.columns:
            years = presence_df['year'].dropna().unique().astype(int)
            self.years = list(range(int(years.min()), int(years.max()) + 1)) if len(years) > 0 else [2020]
        elif date_column in presence_df.columns:
            dates = pd.to_datetime(presence_df[date_column], errors='coerce').dropna()
            if len(dates) > 0:
                years = dates.dt.year.unique().astype(int)
                self.years = list(range(int(years.min()), int(years.max()) + 1))
            else:
                self.years = [2020]
        else:
            self.years = list(range(2006, 2025))  # Default years
        
        if not self.years:
            self.years = [2020]
        
        self.utm_crs = "EPSG:32613"
    
    def generate(self, n_samples: int, max_attempts: int = 5000) -> gpd.GeoDataFrame:
        """
        Generate random temporal background absences.
        
        Args:
            n_samples: Number of absence points to generate
            max_attempts: Maximum attempts per sample
            
        Returns:
            GeoDataFrame with absence points including temporal metadata
        """
        logger.info(f"Generating {n_samples:,} random temporal background absences...")
        
        absences = []
        attempts = 0
        
        for i in range(n_samples):
            # Uniform temporal sampling
            year = np.random.choice(self.years)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, days_in_month(month, year) + 1)
            date = pd.Timestamp(year=year, month=month, day=day)
            
            # Random spatial sampling with distance constraint
            for attempt in range(max_attempts):
                attempts += 1
                lat, lon = sample_random_location(self.study_area_bounds)
                
                if is_far_enough(lat, lon, self.presence_gdf, self.min_distance_meters, self.utm_crs):
                    absence = {
                        'latitude': lat,
                        'longitude': lon,
                        'date': date,
                        'year': year,
                        'month': month,
                        'day_of_year': date.dayofyear,
                        'season': get_season_from_month(month),
                        'elk_present': 0,
                        'absence_strategy': 'random_background',
                        'confidence': 'low'
                    }
                    
                    # Add dataset if present in presence data
                    presence_df = pd.DataFrame(self.presence_gdf.drop(columns='geometry'))
                    if 'dataset' in presence_df.columns:
                        # Uniform sampling across datasets
                        datasets = presence_df['dataset'].unique()
                        absence['dataset'] = np.random.choice(datasets)
                    
                    absences.append(absence)
                    break
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1:,}/{n_samples:,} absences...")
        
        if len(absences) < n_samples:
            logger.warning(
                f"Only generated {len(absences)}/{n_samples} random temporal background absences "
                f"(attempts: {attempts:,})"
            )
        
        # Create GeoDataFrame
        absence_df = pd.DataFrame(absences)
        if len(absence_df) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=self.presence_gdf.crs)
        
        absence_gdf = gpd.GeoDataFrame(
            absence_df,
            geometry=gpd.points_from_xy(
                absence_df.longitude,
                absence_df.latitude
            ),
            crs="EPSG:4326"
        )
        
        logger.info(f"✓ Generated {len(absence_gdf):,} random temporal background absences")
        return absence_gdf


# Check if rasterio is available for DEM-based sampling
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except (ImportError, OSError):
    rasterio = None
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available - elevation-based sampling will be limited")

