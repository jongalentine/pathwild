"""
Absence data generators for PathWild training dataset.

This module provides multiple strategies for generating absence points (locations
where elk were NOT present) to balance presence data for binary classification.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import logging

logger = logging.getLogger(__name__)

# Lazy import rasterio to avoid import errors in sandboxed environments
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except (ImportError, OSError) as e:
    rasterio = None
    RASTERIO_AVAILABLE = False
    logger.warning(f"rasterio not available: {e}. Some features may be limited.")


def _static_generate_worker(
    n_samples: int,
    max_attempts: int,
    seed: int,
    study_area_bounds: Tuple[float, float, float, float],
    presence_points_list: list,
    min_distance_meters: float,
    crs: str,
    utm_crs: str,
    dem_path: Optional[str],
    slope_path: Optional[str],
    water_path: Optional[str],
    strategy_name: str
) -> list:
    """
    Static worker function for multiprocessing that doesn't require self.
    
    Opens rasters inside the worker process to avoid pickling issues.
    """
    import logging
    worker_logger = logging.getLogger(f"{__name__}.worker.{seed}")
    
    try:
        worker_logger.info(f"Worker {seed} starting: {n_samples} samples, {max_attempts} max attempts")
        
        if seed is not None:
            np.random.seed(seed)
        
        absence_points = []
        attempts = 0
        
        # Open rasters inside worker (avoids pickling issues)
        worker_logger.debug(f"Worker {seed}: Opening rasters...")
        dem_raster = None
        slope_raster = None
        water_gdf = None
        
        if dem_path and RASTERIO_AVAILABLE:
            try:
                dem_raster = rasterio.open(dem_path)
                worker_logger.debug(f"Worker {seed}: DEM opened")
            except Exception as e:
                worker_logger.warning(f"Worker {seed}: Failed to open DEM: {e}")
        
        if slope_path and RASTERIO_AVAILABLE:
            try:
                slope_raster = rasterio.open(slope_path)
                worker_logger.debug(f"Worker {seed}: Slope opened")
            except Exception as e:
                worker_logger.warning(f"Worker {seed}: Failed to open slope: {e}")
        
        # Only load water sources if needed for this strategy
        if water_path and strategy_name in ['environmental', 'unsuitable']:
            try:
                worker_logger.debug(f"Worker {seed}: Loading water sources ({water_path})...")
                water_gdf = gpd.read_file(water_path)
                if water_gdf.crs != crs:
                    water_gdf = water_gdf.to_crs(crs)
                worker_logger.debug(f"Worker {seed}: Loaded {len(water_gdf)} water features")
                # Don't build spatial index here - do it lazily when needed
            except Exception as e:
                worker_logger.warning(f"Worker {seed}: Failed to load water sources: {e}")
                water_gdf = None
        else:
            worker_logger.debug(f"Worker {seed}: Skipping water sources (not needed for {strategy_name})")
        
        # Convert presence points to GeoDataFrame for distance calculations
        # Note: presence_points_list is already in UTM coordinates from self.presence_utm
        worker_logger.debug(f"Worker {seed}: Creating presence GeoDataFrame from {len(presence_points_list)} points...")
        presence_gdf_utm = None
        if presence_points_list:
            try:
                from shapely.geometry import Point as ShapelyPoint
                presence_points_utm = [ShapelyPoint(x, y) for x, y in presence_points_list]
                presence_gdf_utm = gpd.GeoDataFrame(geometry=presence_points_utm, crs=utm_crs)
                worker_logger.debug(f"Worker {seed}: Presence GeoDataFrame created")
                # Try to build spatial index (but don't fail if it doesn't work)
                try:
                    if len(presence_gdf_utm) > 1000:
                        _ = presence_gdf_utm.sindex  # Access to build index
                except Exception:
                    pass  # Spatial index optional
            except Exception as e:
                worker_logger.warning(f"Worker {seed}: Failed to create presence GeoDataFrame: {e}")
                presence_gdf_utm = None
        
        # Create study area polygon from bounds
        try:
            from shapely.geometry import box
            study_area_poly = box(*study_area_bounds)
        except Exception:
            study_area_poly = None
        
        # Define helper functions inside try block
        def sample_raster_static(raster, lon: float, lat: float, default: float = 0.0) -> float:
            """Sample value from raster at point (static version)."""
            if raster is None or not RASTERIO_AVAILABLE:
                return default
            try:
                row, col = raster.index(lon, lat)
                window = rasterio.windows.Window(col, row, 1, 1)
                data = raster.read(1, window=window)
                value = float(data[0, 0])
                if value == raster.nodata or np.isnan(value):
                    return default
                return value
            except Exception:
                return default
        
        def calculate_water_distance_static(point: Point) -> float:
            """Calculate distance to nearest water source in miles (static version)."""
            if water_gdf is None or len(water_gdf) == 0:
                return 0.5
            try:
                # Convert point and water_gdf to UTM for accurate distance calculation
                # This avoids the geographic CRS warning and is more accurate
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs=crs)
                point_utm = point_gdf.to_crs(utm_crs).geometry.iloc[0]
                
                # Ensure water_gdf is in UTM (convert if needed)
                water_gdf_utm = water_gdf
                if water_gdf.crs != utm_crs:
                    water_gdf_utm = water_gdf.to_crs(utm_crs)
                
                # For very large datasets, use sampling for performance
                # unary_union on 958K+ features is extremely slow, and spatial index building can hang
                if len(water_gdf_utm) > 10000:
                    # Always use sampling for large datasets - much faster and avoids index building
                    sample_size = min(1000, len(water_gdf_utm))
                    sample_gdf = water_gdf_utm.sample(n=sample_size, random_state=42)
                    # Distance in UTM is in meters, convert to miles
                    distances_m = sample_gdf.geometry.distance(point_utm)
                    distances_mi = distances_m / 1609.34  # meters to miles
                    return float(distances_mi.min())
                else:
                    # Small dataset: calculate distance to all
                    # Distance in UTM is in meters, convert to miles
                    distances_m = water_gdf_utm.geometry.distance(point_utm)
                    distances_mi = distances_m / 1609.34  # meters to miles
                    return float(distances_mi.min())
            except Exception as e:
                # Return default if calculation fails
                worker_logger.debug(f"Worker {seed}: Water distance calculation failed: {e}")
                return 0.5
        
        def check_distance_constraint_static(point: Point) -> bool:
            """Check if point meets distance constraint (static version)."""
            if presence_gdf_utm is None or len(presence_gdf_utm) == 0:
                return True
            try:
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs=crs).to_crs(utm_crs)
                point_utm = point_gdf.geometry.iloc[0]
                
                # For large presence datasets, use spatial index if available
                if len(presence_gdf_utm) > 1000:
                    try:
                        # Build spatial index if not already available
                        if presence_gdf_utm.sindex is None:
                            # Force index creation (geopandas does this automatically, but sometimes fails)
                            presence_gdf_utm.sindex
                    except Exception:
                        pass
                    
                    # Use spatial index for efficient query
                    try:
                        if presence_gdf_utm.sindex is not None:
                            # Query nearby points within a buffer
                            buffer_m = min_distance_meters * 1.1  # Add 10% buffer
                            point_bounds = (
                                point_utm.x - buffer_m,
                                point_utm.y - buffer_m,
                                point_utm.x + buffer_m,
                                point_utm.y + buffer_m
                            )
                            nearby_indices = list(presence_gdf_utm.sindex.intersection(point_bounds))
                            if nearby_indices:
                                # Only check distances to nearby candidates
                                candidates = presence_gdf_utm.iloc[nearby_indices]
                                distances = candidates.geometry.distance(point_utm)
                                min_distance = distances.min()
                                return min_distance >= min_distance_meters
                            else:
                                # No nearby points, constraint satisfied
                                return True
                    except Exception:
                        # Fall back to full distance calculation
                        pass
                
                # Standard distance calculation (for small datasets or if spatial index fails)
                distances = presence_gdf_utm.geometry.distance(point_utm)
                min_distance = distances.min()
                return min_distance >= min_distance_meters
            except Exception as e:
                worker_logger.debug(f"Worker {seed}: Distance constraint check failed: {e}")
                return True
        
        def is_environmentally_suitable_static(point: Point) -> bool:
            """Check environmental suitability (static version)."""
            lon, lat = point.x, point.y
            
            # Check elevation
            elevation_m = sample_raster_static(dem_raster, lon, lat, default=2500.0)
            elevation_ft = elevation_m * 3.28084
            if not (6000 <= elevation_ft <= 13500):
                return False
            
            # Check slope
            slope_deg = sample_raster_static(slope_raster, lon, lat, default=15.0)
            if slope_deg >= 45.0:
                return False
            
            # Check water distance
            water_dist_mi = calculate_water_distance_static(point)
            if water_dist_mi > 5.0:
                return False
            
            return True
        
        def is_unsuitable_static(point: Point) -> bool:
            """Check if point is unsuitable habitat (static version)."""
            lon, lat = point.x, point.y
            
            # Check elevation (too low)
            elevation_m = sample_raster_static(dem_raster, lon, lat, default=2500.0)
            elevation_ft = elevation_m * 3.28084
            if elevation_ft < 4000:
                return True
            
            # Check slope (too steep)
            slope_deg = sample_raster_static(slope_raster, lon, lat, default=15.0)
            if slope_deg >= 45.0:
                return True
            
            # Check water distance (too far)
            water_dist_mi = calculate_water_distance_static(point)
            if water_dist_mi > 10.0:
                return True
            
            return False
        
        worker_logger.info(f"Worker {seed}: Starting generation loop...")
        
        # Generate points
        while len(absence_points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Log progress every 1000 attempts to show worker is alive
            if attempts % 1000 == 0:
                worker_logger.info(f"Worker {seed}: {attempts:,} attempts, {len(absence_points)}/{n_samples} points found")
            
            # Sample random point
            lon = np.random.uniform(study_area_bounds[0], study_area_bounds[2])
            lat = np.random.uniform(study_area_bounds[1], study_area_bounds[3])
            point = Point(lon, lat)
            
            # Check if point is within study area
            if study_area_poly and not study_area_poly.contains(point):
                continue
            
            # Check distance constraint
            if not check_distance_constraint_static(point):
                continue
            
            # Strategy-specific checks
            if strategy_name == 'environmental':
                if not is_environmentally_suitable_static(point):
                    continue
            elif strategy_name == 'unsuitable':
                if not is_unsuitable_static(point):
                    continue
            # For 'background' and 'temporal', no additional checks needed
            
            absence_points.append(point)
        
        worker_logger.info(f"Worker {seed}: Completed - found {len(absence_points)}/{n_samples} points in {attempts} attempts")
        
        # Close rasters
        if dem_raster:
            dem_raster.close()
        if slope_raster:
            slope_raster.close()
        
        return absence_points
        
    except Exception as e:
        worker_logger.error(f"Worker {seed}: Fatal error: {e}", exc_info=True)
        # Return whatever points we found before the error
        return absence_points


class AbsenceGenerator(ABC):
    """Abstract base class for generating absence points.
    
    All absence generators must implement the generate() method to create
    a specified number of absence points that meet spatial and environmental
    constraints.
    """
    
    def __init__(
        self,
        presence_data: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        min_distance_meters: float = 500.0,
        crs: str = "EPSG:4326"
    ):
        """
        Initialize absence generator.
        
        Args:
            presence_data: GeoDataFrame with presence points (elk_present=1)
            study_area: GeoDataFrame defining valid sampling area
            min_distance_meters: Minimum distance from presence points
            crs: Coordinate reference system (default WGS84)
        """
        self.presence_data = presence_data.copy()
        self.study_area = study_area.copy()
        self.min_distance_meters = min_distance_meters
        self.crs = crs
        
        # Ensure both are in same CRS
        if self.presence_data.crs != self.crs:
            self.presence_data = self.presence_data.to_crs(self.crs)
        if self.study_area.crs != self.crs:
            self.study_area = self.study_area.to_crs(self.crs)
        
        # For distance calculations, use UTM (meters)
        # Wyoming is in UTM Zone 12N or 13N
        self.utm_crs = "EPSG:32613"  # UTM Zone 13N (covers most of Wyoming)
        
        # Convert to UTM for accurate distance calculations
        self.presence_utm = self.presence_data.to_crs(self.utm_crs)
    
    @abstractmethod
    def generate(self, n_samples: int, max_attempts: int = 10000, n_processes: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Generate n absence points.
        
        Args:
            n_samples: Number of absence points to generate
            max_attempts: Maximum sampling attempts before giving up
            n_processes: Number of parallel processes (None = auto-detect)
            
        Returns:
            GeoDataFrame with absence points (elk_present=0)
        """
        pass
    
    def _calculate_adaptive_max_attempts(self, n_samples: int) -> int:
        """
        Calculate adaptive max_attempts based on dataset size and target samples.
        
        For large datasets, scale max_attempts proportionally to ensure
        enough attempts to find valid points.
        
        Args:
            n_samples: Target number of samples to generate
            
        Returns:
            Adaptive max_attempts value
        """
        n_presence = len(self.presence_data)
        
        # Base max_attempts
        base_max_attempts = 10000
        
        # Scale with dataset size (more presence points = harder to find absences)
        if n_presence > 50000:
            # Very large dataset: scale aggressively
            scale_factor = max(3.0, n_samples / 5000.0)
        elif n_presence > 10000:
            # Large dataset: moderate scaling
            scale_factor = max(2.0, n_samples / 10000.0)
        else:
            # Small dataset: minimal scaling
            scale_factor = max(1.0, n_samples / 10000.0)
        
        max_attempts = int(base_max_attempts * scale_factor)
        
        # Cap at reasonable maximum (1M attempts)
        max_attempts = min(max_attempts, 1000000)
        
        return max_attempts
    
    def _generate_worker(
        self,
        n_samples: int,
        max_attempts: int,
        seed: Optional[int] = None
    ) -> list:
        """
        Worker function for parallel generation.
        
        This function can be pickled for multiprocessing.
        Each worker generates a subset of the total samples.
        
        Args:
            n_samples: Number of samples to generate in this worker
            max_attempts: Maximum attempts for this worker
            seed: Random seed for reproducibility
            
        Returns:
            List of Point geometries
        """
        if seed is not None:
            np.random.seed(seed)
        
        absence_points = []
        attempts = 0
        
        while len(absence_points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample random point
            point = self._sample_random_point_in_study_area()
            if point is None:
                continue
            
            # Check distance constraint
            if not self.check_distance_constraint(point):
                continue
            
            # Additional constraints are checked in subclass overrides
            # This base implementation only checks distance
            pass
            
            absence_points.append(point)
        
        return absence_points
    
    def _generate_parallel(
        self,
        n_samples: int,
        max_attempts: int,
        n_processes: Optional[int] = None,
        strategy_name: str = "absence"
    ) -> gpd.GeoDataFrame:
        """
        Generate absence points using parallel processing.
        
        Args:
            n_samples: Total number of samples to generate
            max_attempts: Maximum total attempts (divided across workers)
            n_processes: Number of parallel processes (None = auto-detect)
            strategy_name: Name of the absence strategy
            
        Returns:
            GeoDataFrame with absence points
        """
        if n_processes is None:
            n_processes = min(cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        if n_processes == 1:
            # Fall back to sequential
            points = self._generate_worker(n_samples, max_attempts, seed=42)
        else:
            # Split work across processes
            samples_per_process = max(1, n_samples // n_processes)
            remaining_samples = n_samples - (samples_per_process * n_processes)
            
            # Divide max_attempts across workers (each worker gets a portion)
            attempts_per_worker = max(1, max_attempts // n_processes)
            
            # Prepare data that can be pickled for workers
            # Convert rasterio DatasetReader objects to file paths
            dem_path = None
            slope_path = None
            water_path = None
            
            if hasattr(self, 'data_dir') and self.data_dir:
                if hasattr(self, 'dem') and self.dem is not None:
                    dem_path = str(self.dem.name) if hasattr(self.dem, 'name') else None
                if hasattr(self, 'slope') and self.slope is not None:
                    slope_path = str(self.slope.name) if hasattr(self.slope, 'name') else None
                if hasattr(self, 'water_sources') and self.water_sources is not None:
                    # For GeoDataFrames, we need to save to a temp file or pass the path
                    # Since water_sources is already loaded from a file, use the original path
                    water_path = str(self.data_dir / "hydrology" / "water_sources.geojson")
            
            # Prepare study area and presence data for pickling (convert to dict/shapely)
            study_area_bounds = self.study_area.total_bounds
            presence_points_list = [(p.x, p.y) for p in self.presence_utm.geometry]
            
            # Distribute remaining samples
            worker_args = []
            for i in range(n_processes):
                worker_n_samples = samples_per_process
                if i < remaining_samples:
                    worker_n_samples += 1
                
                # Use different seeds for each worker
                seed = 42 + i
                worker_args.append((
                    worker_n_samples,
                    attempts_per_worker,
                    seed,
                    study_area_bounds,
                    presence_points_list,
                    self.min_distance_meters,
                    self.crs,
                    self.utm_crs,
                    dem_path,
                    slope_path,
                    water_path,
                    strategy_name
                ))
            
            logger.info(f"Using {n_processes} parallel processes for {strategy_name} generation")
            logger.info(f"Workers will process {len(worker_args)} batches")
            
            # Use a static worker function that doesn't require self
            # Generate in parallel with timeout protection
            try:
                with Pool(processes=n_processes) as pool:
                    # Use map_async with timeout to detect stuck workers
                    logger.debug("Starting parallel worker pool...")
                    async_result = pool.starmap_async(_static_generate_worker, worker_args)
                    
                    # Wait with a reasonable timeout (5 minutes per worker should be plenty)
                    # For 10,000 attempts, should complete in < 1 minute even if slow
                    timeout = max(300, max_attempts // 100)  # At least 5 minutes, or based on attempts
                    logger.debug(f"Waiting for workers (timeout: {timeout}s)...")
                    results = async_result.get(timeout=timeout)
                    logger.info(f"All workers completed successfully")
            except TimeoutError:
                logger.error(f"Workers timed out after {timeout}s - some workers may be stuck")
                pool.terminate()  # Force kill stuck workers
                raise
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                raise
            
            # Combine results
            points = []
            for result in results:
                points.extend(result)
            
            # If we got more than needed, trim
            if len(points) > n_samples:
                points = points[:n_samples]
        
        # Create GeoDataFrame
        if len(points) == 0:
            logger.warning(f"No {strategy_name} points generated")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)
        
        absence_gdf = gpd.GeoDataFrame(
            geometry=points,
            crs=self.crs
        )
        absence_gdf['latitude'] = absence_gdf.geometry.y
        absence_gdf['longitude'] = absence_gdf.geometry.x
        absence_gdf['absence_strategy'] = strategy_name
        
        return absence_gdf
    
    def check_distance_constraint(
        self,
        candidate_point: Point,
        min_distance_meters: Optional[float] = None
    ) -> bool:
        """
        Check if candidate point is far enough from all presence points.
        
        Args:
            candidate_point: Point to check (in WGS84)
            min_distance_meters: Override default minimum distance
            
        Returns:
            True if candidate meets distance constraint
        """
        if min_distance_meters is None:
            min_distance_meters = self.min_distance_meters
        
        # Convert candidate to UTM for distance calculation
        candidate_gdf = gpd.GeoDataFrame(
            geometry=[candidate_point],
            crs=self.crs
        ).to_crs(self.utm_crs)
        
        candidate_utm = candidate_gdf.geometry.iloc[0]
        
        # Calculate distances to all presence points
        distances = self.presence_utm.geometry.distance(candidate_utm)
        min_distance = distances.min()
        
        return min_distance >= min_distance_meters
    
    def _sample_random_point_in_study_area(self) -> Optional[Point]:
        """
        Sample a random point within the study area bounds.
        
        Returns:
            Random Point in WGS84, or None if sampling fails
        """
        # Get bounding box
        bounds = self.study_area.total_bounds  # [minx, miny, maxx, maxy]
        
        # Sample random point
        for _ in range(100):  # Try up to 100 times
            lon = np.random.uniform(bounds[0], bounds[2])
            lat = np.random.uniform(bounds[1], bounds[3])
            point = Point(lon, lat)
            
            # Check if point is actually within study area (not just bbox)
            if self.study_area.contains(point).any():
                return point
        
        # Fallback: return point from bbox (may be outside actual polygon)
        lon = np.random.uniform(bounds[0], bounds[2])
        lat = np.random.uniform(bounds[1], bounds[3])
        return Point(lon, lat)
    
    def _is_in_study_area(self, point: Point) -> bool:
        """Check if point is within study area."""
        return self.study_area.contains(point).any()


class EnvironmentalPseudoAbsenceGenerator(AbsenceGenerator):
    """
    Generate pseudo-absences from environmentally suitable but unused habitat.
    
    Strategy: Sample random points that:
    - Are ≥2000m from any presence point
    - Meet basic habitat suitability (elevation 6,000-13,500 ft, slope <45°)
    - Are within 5 miles of water
    - Are within study area
    
    Note: Elk use elevations up to 13,500+ ft in summer (high alpine meadows).
    
    This represents "available but unused" habitat.
    """
    
    def __init__(
        self,
        presence_data: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        data_dir: Optional[Path] = None,
        min_distance_meters: float = 2000.0
    ):
        """
        Initialize environmental pseudo-absence generator.
        
        Args:
            presence_data: GeoDataFrame with presence points
            study_area: GeoDataFrame defining valid sampling area
            data_dir: Path to data directory (for loading DEM, water sources)
            min_distance_meters: Minimum distance from presence (default 2000m)
        """
        super().__init__(presence_data, study_area, min_distance_meters)
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Load environmental data if available
        self.dem = None
        self.slope = None
        self.water_sources = None
        
        if self.data_dir:
            self._load_environmental_data()
    
    def _load_environmental_data(self):
        """Load DEM, slope, and water sources for environmental filtering."""
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available, skipping raster data loading")
            return
        
        try:
            # DEM
            dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
            if dem_path.exists():
                self.dem = rasterio.open(dem_path)
                logger.info(f"Loaded DEM: {dem_path}")
            
            # Slope
            slope_path = self.data_dir / "terrain" / "slope.tif"
            if slope_path.exists():
                self.slope = rasterio.open(slope_path)
                logger.info(f"Loaded slope: {slope_path}")
            
            # Water sources
            water_path = self.data_dir / "hydrology" / "water_sources.geojson"
            if water_path.exists():
                self.water_sources = gpd.read_file(water_path)
                if self.water_sources.crs != self.crs:
                    self.water_sources = self.water_sources.to_crs(self.crs)
                logger.info(f"Loaded water sources: {len(self.water_sources)} features")
        except Exception as e:
            logger.warning(f"Error loading environmental data: {e}")
    
    def _sample_raster(self, raster, lon: float, lat: float, default: float = 0.0) -> float:
        """Sample value from raster at point."""
        if raster is None or not RASTERIO_AVAILABLE:
            return default
        
        try:
            row, col = raster.index(lon, lat)
            window = rasterio.windows.Window(col, row, 1, 1)
            data = raster.read(1, window=window)
            value = float(data[0, 0])
            
            if value == raster.nodata or np.isnan(value):
                return default
            return value
        except Exception:
            return default
    
    def _calculate_water_distance(self, point: Point) -> float:
        """Calculate distance to nearest water source in miles."""
        if self.water_sources is None:
            return 0.5  # Default: assume water nearby
        
        try:
            nearest_geom = nearest_points(point, self.water_sources.unary_union)[1]
            distance_m = point.distance(nearest_geom) * 111139  # degrees to meters
            distance_mi = distance_m / 1609.34
            return distance_mi
        except Exception:
            return 0.5
    
    def _is_environmentally_suitable(self, point: Point) -> bool:
        """
        Check if point meets environmental suitability criteria.
        
        Criteria:
        - Elevation: 6,000-13,500 ft (1,829-4,115 m; elk use high alpine areas in summer)
        - Slope: <45 degrees
        - Water distance: <5 miles
        """
        lon, lat = point.x, point.y
        
        # Check elevation
        elevation_m = self._sample_raster(self.dem, lon, lat, default=2500.0)
        elevation_ft = elevation_m * 3.28084
        if not (6000 <= elevation_ft <= 13500):
            return False
        
        # Check slope
        slope_deg = self._sample_raster(self.slope, lon, lat, default=15.0)
        if slope_deg >= 45.0:
            return False
        
        # Check water distance
        water_dist_mi = self._calculate_water_distance(point)
        if water_dist_mi > 5.0:
            return False
        
        return True
    
    def generate(self, n_samples: int, max_attempts: Optional[int] = None, n_processes: Optional[int] = None) -> gpd.GeoDataFrame:
        """Generate environmental pseudo-absences."""
        logger.info(f"Generating {n_samples} environmental pseudo-absences...")
        
        # Use adaptive max_attempts if not specified
        if max_attempts is None:
            max_attempts = self._calculate_adaptive_max_attempts(n_samples)
        
        logger.info(f"Using max_attempts={max_attempts:,} (adaptive)")
        
        # Use parallel processing
        absence_gdf = self._generate_parallel(
            n_samples=n_samples,
            max_attempts=max_attempts,
            n_processes=n_processes,
            strategy_name='environmental'
        )
        
        if len(absence_gdf) < n_samples:
            logger.warning(
                f"Only generated {len(absence_gdf)}/{n_samples} environmental absences "
                f"(max_attempts={max_attempts:,})"
            )
        else:
            logger.info(f"✓ Generated {len(absence_gdf):,} environmental absences")
        
        return absence_gdf


class UnsuitableHabitatAbsenceGenerator(AbsenceGenerator):
    """
    Generate absences from areas elk cannot/will not be.
    
    Strategy: Sample from unsuitable habitat:
    - Elevation <4,000 ft OR >14,000 ft (very low or extreme high elevations)
    - Slope >60°
    - Urban areas, water bodies, barren land (NLCD codes)
    - Distance to water >10 miles
    
    Note: Elk use elevations up to 13,500+ ft in summer (high alpine meadows),
    so only very extreme elevations (>14,000 ft) are considered unsuitable.
    """
    
    def __init__(
        self,
        presence_data: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        data_dir: Optional[Path] = None,
        min_distance_meters: float = 500.0
    ):
        """Initialize unsuitable habitat absence generator."""
        super().__init__(presence_data, study_area, min_distance_meters)
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Load environmental data
        self.dem = None
        self.slope = None
        self.landcover = None
        self.water_sources = None
        
        if self.data_dir:
            self._load_environmental_data()
    
    def _load_environmental_data(self):
        """Load environmental data for filtering."""
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available, skipping raster data loading")
            return
        
        try:
            # DEM
            dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
            if dem_path.exists():
                self.dem = rasterio.open(dem_path)
            
            # Slope
            slope_path = self.data_dir / "terrain" / "slope.tif"
            if slope_path.exists():
                self.slope = rasterio.open(slope_path)
            
            # Land cover
            landcover_path = self.data_dir / "landcover" / "nlcd.tif"
            if landcover_path.exists():
                self.landcover = rasterio.open(landcover_path)
            
            # Water sources
            water_path = self.data_dir / "hydrology" / "water_sources.geojson"
            if water_path.exists():
                self.water_sources = gpd.read_file(water_path)
                if self.water_sources.crs != self.crs:
                    self.water_sources = self.water_sources.to_crs(self.crs)
        except Exception as e:
            logger.warning(f"Error loading environmental data: {e}")
    
    def _sample_raster(self, raster, lon: float, lat: float, default: float = 0.0) -> float:
        """Sample value from raster at point."""
        if raster is None or not RASTERIO_AVAILABLE:
            return default
        
        try:
            row, col = raster.index(lon, lat)
            window = rasterio.windows.Window(col, row, 1, 1)
            data = raster.read(1, window=window)
            value = float(data[0, 0])
            
            if value == raster.nodata or np.isnan(value):
                return default
            return value
        except Exception:
            return default
    
    def _calculate_water_distance(self, point: Point) -> float:
        """Calculate distance to nearest water source in miles."""
        if self.water_sources is None:
            return 20.0  # Default: assume far from water
        
        try:
            nearest_geom = nearest_points(point, self.water_sources.unary_union)[1]
            distance_m = point.distance(nearest_geom) * 111139
            distance_mi = distance_m / 1609.34
            return distance_mi
        except Exception:
            return 20.0
    
    def _is_unsuitable(self, point: Point) -> bool:
        """
        Check if point is in unsuitable habitat.
        
        Criteria (elk cannot/will not be here):
        - Elevation <4,000 ft OR >14,000 ft (very low or extreme high elevations)
        - Slope >60°
        - Urban (NLCD 21-24), water (11-12), barren (31)
        - Water distance >10 miles
        
        Note: Elk use elevations up to 13,500+ ft in summer, so only extreme
        elevations (>14,000 ft) are considered unsuitable.
        """
        lon, lat = point.x, point.y
        
        # Check elevation
        elevation_m = self._sample_raster(self.dem, lon, lat, default=2500.0)
        elevation_ft = elevation_m * 3.28084
        if elevation_ft < 4000 or elevation_ft > 14000:
            return True
        
        # Check slope
        slope_deg = self._sample_raster(self.slope, lon, lat, default=15.0)
        if slope_deg > 60.0:
            return True
        
        # Check land cover (unsuitable types)
        landcover_code = int(self._sample_raster(self.landcover, lon, lat, default=0))
        unsuitable_codes = [11, 12, 21, 22, 23, 24, 31]  # Water, urban, barren
        if landcover_code in unsuitable_codes:
            return True
        
        # Check water distance (too far = unsuitable)
        water_dist_mi = self._calculate_water_distance(point)
        if water_dist_mi > 10.0:
            return True
        
        return False
    
    def _generate_worker(
        self,
        n_samples: int,
        max_attempts: int,
        seed: Optional[int] = None
    ) -> list:
        """Override to add unsuitable habitat check."""
        if seed is not None:
            np.random.seed(seed)
        
        absence_points = []
        attempts = 0
        
        while len(absence_points) < n_samples and attempts < max_attempts:
            attempts += 1
            
            point = self._sample_random_point_in_study_area()
            if point is None:
                continue
            
            if not self.check_distance_constraint(point):
                continue
            
            # Check if unsuitable
            if not self._is_unsuitable(point):
                continue
            
            absence_points.append(point)
        
        return absence_points
    
    def generate(self, n_samples: int, max_attempts: Optional[int] = None, n_processes: Optional[int] = None) -> gpd.GeoDataFrame:
        """Generate unsuitable habitat absences."""
        logger.info(f"Generating {n_samples} unsuitable habitat absences...")
        
        # Use adaptive max_attempts if not specified
        if max_attempts is None:
            max_attempts = self._calculate_adaptive_max_attempts(n_samples)
        
        logger.info(f"Using max_attempts={max_attempts:,} (adaptive)")
        
        # Use parallel processing
        absence_gdf = self._generate_parallel(
            n_samples=n_samples,
            max_attempts=max_attempts,
            n_processes=n_processes,
            strategy_name='unsuitable'
        )
        
        if len(absence_gdf) < n_samples:
            logger.warning(
                f"Only generated {len(absence_gdf)}/{n_samples} unsuitable absences "
                f"(max_attempts={max_attempts:,})"
            )
        else:
            logger.info(f"✓ Generated {len(absence_gdf):,} unsuitable absences")
        
        return absence_gdf


class RandomBackgroundGenerator(AbsenceGenerator):
    """
    Generate random background points across study area.
    
    Strategy: Pure random sampling with minimal constraints:
    - ≥500m from presence points
    - Within study area
    - No other filters
    
    Represents "available habitat" vs "used habitat" (presence).
    """
    
    def __init__(
        self,
        presence_data: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        min_distance_meters: float = 500.0
    ):
        """Initialize random background generator."""
        super().__init__(presence_data, study_area, min_distance_meters)
    
    def generate(self, n_samples: int, max_attempts: Optional[int] = None, n_processes: Optional[int] = None) -> gpd.GeoDataFrame:
        """Generate random background absences."""
        logger.info(f"Generating {n_samples} random background absences...")
        
        # Use adaptive max_attempts if not specified
        if max_attempts is None:
            max_attempts = self._calculate_adaptive_max_attempts(n_samples)
        
        logger.info(f"Using max_attempts={max_attempts:,} (adaptive)")
        
        # Use parallel processing
        absence_gdf = self._generate_parallel(
            n_samples=n_samples,
            max_attempts=max_attempts,
            n_processes=n_processes,
            strategy_name='background'
        )
        
        if len(absence_gdf) < n_samples:
            logger.warning(
                f"Only generated {len(absence_gdf)}/{n_samples} background absences "
                f"(max_attempts={max_attempts:,})"
            )
        else:
            logger.info(f"✓ Generated {len(absence_gdf):,} background absences")
        
        return absence_gdf


class TemporalAbsenceGenerator(AbsenceGenerator):
    """
    Generate temporal absences from presence locations at different times.
    
    Strategy: For datasets with timestamps:
    - Use same locations as presence points
    - But different time periods (e.g., if elk was there in summer, mark as absent in winter)
    - Helps model learn seasonal patterns
    
    Note: Requires individual GPS fixes with timestamps, not migration routes.
    """
    
    def __init__(
        self,
        presence_data: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame,
        date_column: str = 'date',
        min_distance_meters: float = 0.0  # Same location, different time
    ):
        """
        Initialize temporal absence generator.
        
        Args:
            presence_data: GeoDataFrame with presence points and date column
            study_area: GeoDataFrame defining valid sampling area
            date_column: Name of column with date/timestamp info
        """
        super().__init__(presence_data, study_area, min_distance_meters)
        self.date_column = date_column
        
        # Check if date column exists
        if date_column not in presence_data.columns:
            logger.warning(
                f"Date column '{date_column}' not found. "
                "Temporal absences may not be generated."
            )
    
    def generate(self, n_samples: int, max_attempts: Optional[int] = None, n_processes: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Generate temporal absences.
        
        For each presence point, create an absence at the same location
        but during a different season (e.g., if summer presence, use winter date).
        """
        logger.info(f"Generating {n_samples} temporal absences...")
        
        if self.date_column not in self.presence_data.columns:
            logger.warning("Cannot generate temporal absences without date column")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)
        
        # Sample presence points to use as base locations
        n_to_sample = min(n_samples, len(self.presence_data))
        sampled_presence = self.presence_data.sample(
            n=n_to_sample,
            random_state=42,
            replace=True if n_samples > len(self.presence_data) else False
        )
        
        absence_points = []
        absence_dates = []
        
        for idx, row in sampled_presence.iterrows():
            point = row.geometry
            
            # Get original date
            original_date = pd.to_datetime(row[self.date_column])
            
            # Shift to opposite season
            # If summer (Jun-Aug), use winter (Dec-Feb)
            # If winter (Dec-Feb), use summer (Jun-Aug)
            # Otherwise, shift by 6 months
            month = original_date.month
            
            if month in [6, 7, 8]:  # Summer -> Winter
                new_month = np.random.choice([12, 1, 2])
                new_year = original_date.year if new_month == 12 else original_date.year + 1
            elif month in [12, 1, 2]:  # Winter -> Summer
                new_month = np.random.choice([6, 7, 8])
                new_year = original_date.year
            else:  # Spring/Fall -> opposite season
                new_month = (month + 6) % 12
                if new_month == 0:
                    new_month = 12
                new_year = original_date.year + (1 if month > 6 else 0)
            
            # Create new date (same day of month, or 15th if invalid)
            try:
                new_date = pd.Timestamp(year=new_year, month=new_month, day=min(original_date.day, 28))
            except:
                new_date = pd.Timestamp(year=new_year, month=new_month, day=15)
            
            absence_points.append(point)
            absence_dates.append(new_date)
        
        # Create GeoDataFrame
        absence_gdf = gpd.GeoDataFrame(
            geometry=absence_points,
            crs=self.crs
        )
        absence_gdf['latitude'] = absence_gdf.geometry.y
        absence_gdf['longitude'] = absence_gdf.geometry.x
        absence_gdf['absence_strategy'] = 'temporal'
        absence_gdf['date'] = absence_dates
        absence_gdf['original_date'] = sampled_presence[self.date_column].values
        
        return absence_gdf

