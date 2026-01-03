from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from shapely.geometry import Point, shape
import requests
from functools import lru_cache
import logging
import warnings

logger = logging.getLogger(__name__)

# Lazy import rasterio to avoid import errors in sandboxed environments
try:
    import rasterio
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
    # Suppress NotGeoreferencedWarning (non-critical, rasterio handles it with identity transform)
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
except (ImportError, OSError) as e:
    rasterio = None
    mask = None
    RASTERIO_AVAILABLE = False
    logger.warning(f"rasterio not available: {e}. Some features may be limited.")

class DataContextBuilder:
    """Builds comprehensive context for heuristic calculations"""
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir or (self.data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load static layers
        self._load_static_layers()
        
        # Initialize data loaders
        self.snotel_client = SNOTELClient(self.data_dir)
        self.weather_client = WeatherClient()
        self.satellite_client = SatelliteClient()
    
    def _load_static_layers(self):
        """Load data that doesn't change over time"""
        
        if not RASTERIO_AVAILABLE:
            print("Loading static data layers...")
            print("  ⚠ rasterio not available, skipping raster data loading")
            self.dem = None
            self.slope = None
            self.aspect = None
            self.landcover = None
            self.canopy = None
            # Still load vector data which doesn't require rasterio
            import geopandas as gpd
            # Water sources
            water_path = self.data_dir / "hydrology" / "water_sources.geojson"
            if water_path.exists():
                self.water_sources = gpd.read_file(water_path)
                print(f"  ✓ Water sources loaded: {len(self.water_sources)} features")
            else:
                self.water_sources = None
            # Roads
            roads_path = self.data_dir / "infrastructure" / "roads.geojson"
            if roads_path.exists():
                self.roads = gpd.read_file(roads_path)
                print(f"  ✓ Roads loaded: {len(self.roads)} features")
            else:
                self.roads = None
            # Hunt areas
            hunt_areas_path = self.data_dir / "hunt_areas" / "hunt_areas.geojson"
            if hunt_areas_path.exists():
                self.hunt_areas = gpd.read_file(hunt_areas_path)
                print(f"  ✓ Hunt areas loaded: {len(self.hunt_areas)} features")
            else:
                self.hunt_areas = None
            # Trails
            trails_path = self.data_dir / "infrastructure" / "trails.geojson"
            if trails_path.exists():
                self.trails = gpd.read_file(trails_path)
                print(f"  ✓ Trails loaded: {len(self.trails)} features")
            else:
                self.trails = None
            # Wolf pack territories
            wolf_path = self.data_dir / "wildlife" / "wolf_packs.geojson"
            if wolf_path.exists():
                self.wolf_packs = gpd.read_file(wolf_path)
                print(f"  ✓ Wolf territories loaded: {len(self.wolf_packs)} packs")
            else:
                self.wolf_packs = None
            # Bear activity
            bear_path = self.data_dir / "wildlife" / "bear_activity.geojson"
            if bear_path.exists():
                self.bear_activity = gpd.read_file(bear_path)
                print(f"  ✓ Bear activity loaded: {len(self.bear_activity)} features")
            else:
                self.bear_activity = None
            return
        
        print("Loading static data layers...")
        
        # Digital Elevation Model
        dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
        if dem_path.exists():
            self.dem = rasterio.open(dem_path)
            print(f"  ✓ DEM loaded: {dem_path}")
        else:
            print(f"  ✗ DEM not found: {dem_path}")
            self.dem = None
        
        # Slope (derived from DEM)
        slope_path = self.data_dir / "terrain" / "slope.tif"
        if slope_path.exists():
            self.slope = rasterio.open(slope_path)
            print(f"  ✓ Slope loaded")
        else:
            self.slope = None
        
        # Aspect
        aspect_path = self.data_dir / "terrain" / "aspect.tif"
        if aspect_path.exists():
            self.aspect = rasterio.open(aspect_path)
            print(f"  ✓ Aspect loaded")
        else:
            self.aspect = None
        
        # Land cover
        landcover_path = self.data_dir / "landcover" / "nlcd.tif"
        if landcover_path.exists():
            self.landcover = rasterio.open(landcover_path)
            print(f"  ✓ Land cover loaded")
        else:
            self.landcover = None
        
        # Canopy cover
        canopy_path = self.data_dir / "canopy" / "canopy_cover.tif"
        if canopy_path.exists():
            self.canopy = rasterio.open(canopy_path)
            print(f"  ✓ Canopy cover loaded")
        else:
            self.canopy = None
        
        # Water sources (vector data)
        water_path = self.data_dir / "hydrology" / "water_sources.geojson"
        if water_path.exists():
            import geopandas as gpd
            self.water_sources = gpd.read_file(water_path)
            # Cache projected version for efficient distance calculations
            if self.water_sources.crs is None:
                self.water_sources.set_crs('EPSG:4326', inplace=True)
            # Convert to UTM Zone 12N (covers most of Wyoming) for accurate distance calculations
            self.water_sources_proj = self.water_sources.to_crs('EPSG:32612')
            print(f"  ✓ Water sources loaded: {len(self.water_sources)} features")
        else:
            print(f"  ✗ Water sources not found")
            self.water_sources = None
            self.water_sources_proj = None
        
        # Roads (vector data)
        roads_path = self.data_dir / "infrastructure" / "roads.geojson"
        if roads_path.exists():
            import geopandas as gpd
            self.roads = gpd.read_file(roads_path)
            print(f"  ✓ Roads loaded: {len(self.roads)} features")
        else:
            self.roads = None
        
        # Trails
        trails_path = self.data_dir / "infrastructure" / "trails.geojson"
        if trails_path.exists():
            import geopandas as gpd
            self.trails = gpd.read_file(trails_path)
            print(f"  ✓ Trails loaded: {len(self.trails)} features")
        else:
            self.trails = None
        
        # Wolf pack territories
        wolf_path = self.data_dir / "wildlife" / "wolf_packs.geojson"
        if wolf_path.exists():
            import geopandas as gpd
            self.wolf_packs = gpd.read_file(wolf_path)
            print(f"  ✓ Wolf territories loaded: {len(self.wolf_packs)} packs")
        else:
            self.wolf_packs = None
        
        # Bear activity centers
        bear_path = self.data_dir / "wildlife" / "bear_activity.geojson"
        if bear_path.exists():
            import geopandas as gpd
            self.bear_activity = gpd.read_file(bear_path)
            print(f"  ✓ Bear activity loaded: {len(self.bear_activity)} locations")
        else:
            self.bear_activity = None
        
        print("Static data loading complete.\n")
    
    def build_context(self, location: Dict, date: str, 
                     buffer_km: float = 1.0) -> Dict:
        """
        Build complete context for a location and date
        
        Args:
            location: {"lat": float, "lon": float}
            date: ISO format date string
            buffer_km: Radius for neighborhood analysis
        
        Returns:
            Dictionary with all context data
        """
        lat, lon = location["lat"], location["lon"]
        point = Point(lon, lat)
        
        context = {}
        
        # --- STATIC TERRAIN DATA ---
        # Check if location is within Wyoming bounds before sampling
        # Wyoming boundaries: approximately 41-45°N, 104-111°W
        wyoming_bounds = {
            'north': 45.0,
            'south': 41.0,
            'east': -104.0,
            'west': -111.0
        }
        
        outside_bounds = (
            lat > wyoming_bounds['north'] or
            lat < wyoming_bounds['south'] or
            lon < wyoming_bounds['west'] or
            lon > wyoming_bounds['east']
        )
        
        if outside_bounds:
            # Location is outside Wyoming - use NaN to indicate "outside bounds" 
            # This is different from placeholder 8500.0, so we can distinguish
            # between "not processed yet" vs "outside bounds"
            context["elevation"] = np.nan
            logger.debug(f"Location ({lat:.6f}, {lon:.6f}) is outside Wyoming bounds - setting elevation to NaN")
        else:
            # Sample elevation from DEM
            elevation = self._sample_raster(self.dem, lon, lat, default=8500.0)
            context["elevation"] = elevation
            
            # Warn if using placeholder elevation (indicates DEM sampling failed)
            if elevation == 8500.0:
                # Check if near boundaries (within 0.1 degrees)
                tolerance = 0.1
                boundary_issues = []
                if lat > wyoming_bounds['north'] - tolerance:
                    boundary_issues.append(f"near northern boundary ({wyoming_bounds['north']}°N)")
                if lat < wyoming_bounds['south'] + tolerance:
                    boundary_issues.append(f"near southern boundary ({wyoming_bounds['south']}°N)")
                if lon < wyoming_bounds['west'] + tolerance:
                    boundary_issues.append(f"near western boundary ({wyoming_bounds['west']}°W)")
                if lon > wyoming_bounds['east'] - tolerance:
                    boundary_issues.append(f"near eastern boundary ({wyoming_bounds['east']}°W)")
                
                # Build warning message
                warning_msg = f"Using default elevation (8500 ft) for location ({lat:.6f}, {lon:.6f})"
                if boundary_issues:
                    warning_msg += f" - Location is {', '.join(boundary_issues)}"
                else:
                    warning_msg += " - DEM sampling failed (may be outside DEM bounds or DEM file issue)"
                
                logger.warning(warning_msg)
        
        if outside_bounds:
            # Location is outside Wyoming - set all terrain features to NaN
            context["slope_degrees"] = np.nan
            context["aspect_degrees"] = np.nan
            context["canopy_cover_percent"] = np.nan
            context["land_cover_code"] = np.nan
            context["land_cover_type"] = "outside_bounds"
        else:
            # Sample terrain features from rasters
            context["slope_degrees"] = self._sample_raster(self.slope, lon, lat, default=15.0)
            context["aspect_degrees"] = self._sample_raster(self.aspect, lon, lat, default=180.0)
            # Sample canopy cover and clamp to valid range (0-100%)
            canopy_value = self._sample_raster(
                self.canopy, lon, lat, default=30.0
            )
            # Clamp to valid percentage range (0-100%)
            context["canopy_cover_percent"] = max(0.0, min(100.0, canopy_value))
            
            # Land cover type
            landcover_code = self._sample_raster(self.landcover, lon, lat, default=0)
            context["land_cover_type"] = self._decode_landcover(landcover_code)
            context["land_cover_code"] = landcover_code
        
        # --- WATER DATA ---
        if self.water_sources is not None:
            water_dist, water_reliability = self._calculate_water_metrics(point)
            context["water_distance_miles"] = water_dist
            context["water_reliability"] = water_reliability
        else:
            context["water_distance_miles"] = 0.5  # Default
            context["water_reliability"] = 0.8
        
        # --- ROAD/TRAIL ACCESS ---
        if self.roads is not None:
            context["road_distance_miles"] = self._calculate_distance_to_nearest(
                point, self.roads
            )
        else:
            context["road_distance_miles"] = 2.0
        
        if self.trails is not None:
            context["trail_distance_miles"] = self._calculate_distance_to_nearest(
                point, self.trails
            )
        else:
            context["trail_distance_miles"] = 1.5
        
        # --- SECURITY HABITAT ---
        context["security_habitat_percent"] = self._calculate_security_habitat(
            point, buffer_km
        )
        
        # --- PREDATOR DATA ---
        if self.wolf_packs is not None:
            context["wolves_per_1000_elk"] = self._calculate_wolf_density(point)
            context["wolf_data_quality"] = 0.75
        else:
            context["wolves_per_1000_elk"] = 2.0  # Low default
            context["wolf_data_quality"] = 0.5
        
        if self.bear_activity is not None:
            context["bear_activity_distance_miles"] = \
                self._calculate_distance_to_nearest(point, self.bear_activity)
            context["bear_data_quality"] = 0.65
        else:
            context["bear_activity_distance_miles"] = 5.0
            context["bear_data_quality"] = 0.5
        
        # --- TEMPORAL DATA (depends on date) ---
        dt = datetime.fromisoformat(date)
        
        # Snow data - pass elevation for better estimation if no station nearby
        location_elevation = context.get("elevation", None)
        snow_data = self.snotel_client.get_snow_data(lat, lon, dt, elevation_ft=location_elevation)
        context["snow_depth_inches"] = snow_data.get("depth", 0.0)
        context["snow_water_equiv_inches"] = snow_data.get("swe", 0.0)
        context["snow_crust_detected"] = snow_data.get("crust", False)
        
        # Data quality tracking: indicate whether data is real SNOTEL or estimate
        snow_station = snow_data.get("station")
        context["snow_data_source"] = "snotel" if snow_station else "estimate"
        context["snow_station_name"] = snow_station
        context["snow_station_distance_km"] = snow_data.get("station_distance_km", None)
        
        # Weather data
        weather = self.weather_client.get_weather(lat, lon, dt)
        context["temperature_f"] = weather.get("temp", 45.0)
        context["precip_last_7_days_inches"] = weather.get("precip_7d", 0.0)
        context["cloud_cover_percent"] = weather.get("cloud_cover", 20)
        
        # Vegetation data
        ndvi_data = self.satellite_client.get_ndvi(lat, lon, dt)
        context["ndvi"] = ndvi_data.get("ndvi", 0.5)
        context["ndvi_age_days"] = ndvi_data.get("age_days", 8)
        context["irg"] = ndvi_data.get("irg", 0.0)
        
        # Summer integrated NDVI (for nutritional condition)
        if dt.month >= 9:  # After summer
            summer_start = datetime(dt.year, 6, 1)
            summer_end = datetime(dt.year, 9, 1)
            context["summer_integrated_ndvi"] = \
                self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
        else:
            # Use previous year
            summer_start = datetime(dt.year - 1, 6, 1)
            summer_end = datetime(dt.year - 1, 9, 1)
            context["summer_integrated_ndvi"] = \
                self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
        
        # Population/herd data (from state agencies)
        context["pregnancy_rate"] = 0.90  # Typical
        
        # Grid references for spatial operations
        context["dem_grid"] = self.dem
        context["water_sources"] = self.water_sources
        
        return context
    
    def _sample_raster(self, raster, lon: float, lat: float, 
                      default: float = 0.0) -> float:
        """Sample value from raster at point"""
        if raster is None or not RASTERIO_AVAILABLE:
            return default
        
        try:
            # Handle CRS transformation if needed
            # If raster is not in WGS84, transform coordinates
            if raster.crs and not raster.crs.is_geographic:
                # Raster is in projected CRS, need to transform lat/lon
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                row, col = raster.index(x, y)
            else:
                # Raster is in geographic CRS (WGS84), use directly
                row, col = raster.index(lon, lat)
            
            # Read value
            window = rasterio.windows.Window(col, row, 1, 1)
            data = raster.read(1, window=window)
            
            value = float(data[0, 0])
            
            # Check for nodata
            if raster.nodata is not None and (value == raster.nodata or np.isnan(value)):
                return default
            
            return value
        except Exception as e:
            logger.debug(f"Error sampling raster at ({lon}, {lat}): {e}")
            return default
    
    def _calculate_water_metrics(self, point: Point) -> Tuple[float, float]:
        """Calculate distance to water and reliability using spatial index for efficiency"""
        import geopandas as gpd
        import pandas as pd
        
        # Ensure water_sources has a CRS (should be EPSG:4326)
        if self.water_sources.crs is None:
            self.water_sources.set_crs('EPSG:4326', inplace=True)
        
        # Use cached projected version for efficient distance calculations
        utm_crs = 'EPSG:32612'
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
        point_gdf_proj = point_gdf.to_crs(utm_crs)
        point_proj = point_gdf_proj.geometry.iloc[0]
        
        try:
            # Use sjoin_nearest in projected CRS (no warnings, accurate distances)
            nearest = gpd.sjoin_nearest(
                point_gdf_proj,
                self.water_sources_proj,
                how='left',
                max_distance=100000,  # 100 km max in meters
                distance_col='distance_m'
            )
            
            if len(nearest) == 0 or pd.isna(nearest.iloc[0].get('index_right')):
                # No water source found within max_distance
                return 10.0, 0.5  # Default: 10 miles, medium reliability
            
            # Get the nearest water source (use original index from projected dataframe)
            nearest_idx = int(nearest.iloc[0]['index_right'])
            nearest_feature = self.water_sources.iloc[nearest_idx]
            
            # Distance is already in meters from sjoin_nearest
            distance_m = nearest.iloc[0].get('distance_m', point_proj.distance(self.water_sources_proj.iloc[nearest_idx].geometry))
            distance_mi = distance_m / 1609.34
            
        except (AttributeError, TypeError, KeyError):
            # Fallback: use spatial index directly
            if not hasattr(self.water_sources_proj, 'sindex') or self.water_sources_proj.sindex is None:
                self.water_sources_proj.sindex  # Build spatial index if needed
            
            # Find nearest using spatial index
            possible_matches_index = list(self.water_sources_proj.sindex.nearest(
                point_proj.bounds, num_results=1
            ))
            
            if len(possible_matches_index) == 0:
                return 10.0, 0.5  # Default
            
            # Get the nearest feature
            nearest_idx = possible_matches_index[0]
            nearest_feature = self.water_sources.iloc[nearest_idx]
            nearest_geom_proj = self.water_sources_proj.iloc[nearest_idx].geometry
            
            # Calculate distance in projected CRS (meters)
            distance_m = point_proj.distance(nearest_geom_proj)
            distance_mi = distance_m / 1609.34
        
        # Get water source attributes
        water_type = nearest_feature.get("water_type", nearest_feature.get("type", "stream"))
        if pd.notna(water_type):
            water_type = str(water_type).lower()
        else:
            water_type = "stream"
        
        reliability_map = {
            "spring": 1.0,
            "lake": 1.0,
            "pond": 0.9,
            "stream": 0.7,
            "creek": 0.7,
            "ephemeral": 0.4
        }
        reliability = reliability_map.get(water_type, 0.7)
        
        return distance_mi, reliability
    
    def _calculate_distance_to_nearest(self, point: Point, features) -> float:
        """Calculate distance to nearest feature in miles using spatial index"""
        import geopandas as gpd
        import pandas as pd
        
        # Ensure features have a CRS
        if features.crs is None:
            features.set_crs('EPSG:4326', inplace=True)
        
        # Convert to projected CRS (UTM Zone 12N for Wyoming) for accurate distance calculations
        utm_crs = 'EPSG:32612'
        features_proj = features.to_crs(utm_crs)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
        point_gdf_proj = point_gdf.to_crs(utm_crs)
        point_proj = point_gdf_proj.geometry.iloc[0]
        
        try:
            # Use sjoin_nearest in projected CRS (no warnings, accurate distances)
            nearest = gpd.sjoin_nearest(
                point_gdf_proj,
                features_proj,
                how='left',
                max_distance=100000,  # 100 km max in meters
                distance_col='distance_m'
            )
            
            if len(nearest) == 0 or pd.isna(nearest.iloc[0].get('index_right')):
                # No feature found within max_distance
                return 10.0  # Default: 10 miles
            
            # Get the nearest feature
            nearest_idx = int(nearest.iloc[0]['index_right'])
            
            # Distance is already in meters from sjoin_nearest
            distance_m = nearest.iloc[0].get('distance_m', point_proj.distance(features_proj.iloc[nearest_idx].geometry))
            distance_mi = distance_m / 1609.34
            
        except (AttributeError, TypeError, KeyError):
            # Fallback: use spatial index directly
            if not hasattr(features_proj, 'sindex') or features_proj.sindex is None:
                features_proj.sindex  # Build spatial index if needed
            
            # Find nearest using spatial index
            possible_matches_index = list(features_proj.sindex.nearest(
                point_proj.bounds, num_results=1
            ))
            
            if len(possible_matches_index) == 0:
                return 10.0  # Default
            
            # Get the nearest feature
            nearest_idx = possible_matches_index[0]
            nearest_geom_proj = features_proj.iloc[nearest_idx].geometry
            
            # Calculate distance in projected CRS (meters)
            distance_m = point_proj.distance(nearest_geom_proj)
            distance_mi = distance_m / 1609.34
        
        return distance_mi
    
    def _calculate_security_habitat(self, point: Point, buffer_km: float) -> float:
        """Calculate % of security habitat in buffer around point"""
        
        # Create buffer (in degrees, approximate)
        buffer_deg = buffer_km / 111.0  # 1 degree ≈ 111 km
        buffered = point.buffer(buffer_deg)
        
        # Sample terrain within buffer
        # Security criteria: slope > 40° OR canopy > 70% OR remote
        
        if self.slope is None:
            return 35.0  # Default moderate security
        
        if not RASTERIO_AVAILABLE or self.slope is None:
            return 35.0  # Default moderate security
        
        try:
            # Extract slope within buffer
            from rasterio.mask import mask as rio_mask
            
            geom = [buffered.__geo_interface__]
            # Suppress NotGeoreferencedWarning for this operation (non-critical)
            with warnings.catch_warnings():
                if RASTERIO_AVAILABLE:
                    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
                slope_data, _ = rio_mask(self.slope, geom, crop=True)
            
            # Calculate % meeting security criteria
            steep_pixels = np.sum(slope_data > 40)
            total_pixels = slope_data.size
            
            security_pct = (steep_pixels / total_pixels) * 100
            
            # TODO: Also check canopy cover and remoteness
            # For now, just use slope
            
            return float(security_pct)
        except Exception as e:
            return 35.0
    
    def _calculate_wolf_density(self, point: Point) -> float:
        """Calculate wolves per 1000 elk in area"""
        
        # Check if point is within any wolf pack territory
        for idx, pack in self.wolf_packs.iterrows():
            if pack.geometry.contains(point):
                pack_size = pack.get("pack_size", 6)
                territory_area_sqmi = pack.geometry.area * 111.0**2  # deg² to km² to mi²
                elk_in_territory = pack.get("elk_count", 2000)
                
                wolves_per_1000 = (pack_size / elk_in_territory) * 1000
                
                return wolves_per_1000
        
        # Not in any pack territory
        return 0.5  # Background level
    
    def _decode_landcover(self, code) -> str:
        """Decode NLCD land cover code to description"""
        import pandas as pd
        
        # Handle NaN/None values
        if pd.isna(code) or code is None:
            return "outside_bounds"
        
        landcover_map = {
            41: "deciduous_forest",
            42: "evergreen_forest",
            43: "mixed_forest",
            52: "shrub",
            71: "grassland",
            81: "pasture",
            90: "wetland",
            95: "emergent_wetland",
            # ... add more as needed
        }
        return landcover_map.get(int(code), "unknown")


class SNOTELClient:
    """Client for accessing SNOTEL snow data using snotelr R package"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.station_cache_path = self.cache_dir / "snotel_stations_wyoming.geojson"
        self._stations_gdf = None
        self._load_stations()
        
        # Two-level caching:
        # 1. Station data cache: station_id -> full DataFrame from snotelr (entire historical record)
        #    This avoids re-downloading the same station data for multiple locations/dates
        #    snotel_download() returns all historical data, so we cache by station ID only
        # 2. Request cache: (lat, lon, date) -> final result dict
        #    This provides fast lookup for exact same location/date queries
        self.station_data_cache = {}  # Cache full station downloads: {site_id: DataFrame}
        self.request_cache = {}  # Cache final results: {(lat, lon, date_str): result_dict}
        
        self._r_initialized = False
        self._snotelr_warned = False  # Track if we've already warned about snotelr being unavailable
        self._init_r_snotelr()
    
    def _init_r_snotelr(self):
        """
        Initialize R and load snotelr package.
        
        Note: rpy2 uses Python's contextvars to store conversion rules. In worker threads,
        these context variables are not automatically available. We try to initialize rpy2
        properly in each thread, but R itself is not thread-safe, so this may still fail.
        """
        try:
            import rpy2.robjects as ro  # type: ignore
            from rpy2.robjects.packages import importr  # type: ignore
            # Note: pandas2ri is imported later when needed (in _fetch_snow_data)
            # We don't need it here for initialization
            
            # Initialize R
            if not self._r_initialized:
                # Try to import snotelr
                try:
                    # Note: pandas2ri.activate() is deprecated in newer rpy2 versions
                    # We don't need to activate it here - it will be used with localconverter when needed
                    # Attempting to activate can cause issues in some environments
                    
                    self.snotelr = importr('snotelr')
                    self.ro = ro
                    self._r_initialized = True
                    logger.info("snotelr R package initialized successfully")
                except Exception as e:
                    error_msg = str(e)
                    import traceback
                    import threading
                    error_traceback = traceback.format_exc()
                    
                    # Check if we're in the main thread or a worker thread
                    is_main_thread = threading.current_thread() is threading.main_thread()
                    thread_info = "main thread" if is_main_thread else "worker thread"
                    
                    # Check if this is the rpy2 context issue with threading
                    if "contextvars.ContextVar" in error_msg or "Conversion rules" in error_msg:
                        # This is a known limitation: rpy2 context variables don't propagate to threads
                        # R itself is also not thread-safe, so even if we fix the context issue,
                        # using R in multiple threads simultaneously can cause crashes
                        if not is_main_thread:
                            logger.warning(
                                f"Could not load snotelr R package in {thread_info} (rpy2 context issue). "
                                "rpy2 uses contextvars that don't automatically propagate to worker threads. "
                                "Additionally, R is not thread-safe. "
                                "Falling back to elevation-based snow estimates for this thread."
                            )
                        else:
                            # This shouldn't happen in main thread, but log it anyway
                            logger.warning(
                                f"Could not load snotelr R package in {thread_info} (rpy2 context issue). "
                                "This is unexpected in the main thread. "
                                "Falling back to elevation-based snow estimates."
                            )
                        logger.debug(f"Full error: {e}")
                        logger.debug(f"Traceback:\n{error_traceback}")
                        if not is_main_thread:
                            logger.debug(
                                "To use snotelr with parallel processing, consider: "
                                "1) Use --workers 1 for sequential processing, or "
                                "2) Use multiprocessing instead of threading (but rasterio datasets can't be pickled)"
                            )
                    else:
                        # Other error - log it with more detail
                        logger.warning(
                            f"Could not load snotelr R package in {thread_info}: {e}"
                        )
                        logger.debug(f"Full traceback:\n{error_traceback}")
                        logger.info("Install with: R -e \"install.packages('snotelr', repos='https://cloud.r-project.org')\"")
                        logger.info("Or check if R and rpy2 are properly installed in your conda environment")
                        if is_main_thread:
                            logger.info("Since you're using --workers 1, this error is unexpected. Check R installation.")
                    self.snotelr = None
                    self.ro = None
        except ImportError:
            logger.warning("rpy2 not available. Install with: pip install rpy2")
            logger.info("Or use conda: conda install -c conda-forge rpy2")
            self.snotelr = None
            self.ro = None
            self._r_initialized = False
    
    def _load_stations(self):
        """Load SNOTEL station locations"""
        import geopandas as gpd
        from shapely.geometry import Point
        
        if self.station_cache_path.exists():
            try:
                self._stations_gdf = gpd.read_file(self.station_cache_path)
                logger.info(f"Loaded {len(self._stations_gdf)} SNOTEL stations")
            except Exception as e:
                logger.warning(f"Error loading stations: {e}")
                self._stations_gdf = None
        else:
            logger.warning(f"SNOTEL stations file not found: {self.station_cache_path}")
            logger.info("Run scripts/download_snotel_stations_manual.py to download stations")
            self._stations_gdf = None
    
    def _find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 100.0):
        """
        Find nearest SNOTEL station to location.
        
        Prioritizes mapped stations (those with snotelr_site_id) over unmapped ones.
        If multiple mapped stations are within range, returns the closest one.
        Only returns unmapped stations if no mapped stations are available.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum distance to search (default: 100 km to allow finding mapped stations)
        
        Returns:
            Dictionary with station info, or None if no station within range
        """
        if self._stations_gdf is None:
            return None
        
        from shapely.geometry import Point
        import geopandas as gpd
        import pandas as pd
        
        point = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        
        # Convert to UTM for distance calculation (Wyoming is in UTM Zone 12N)
        stations_utm = self._stations_gdf.to_crs("EPSG:32612").copy()
        point_utm = point_gdf.to_crs("EPSG:32612")
        
        # Calculate distances to all stations
        stations_utm['distance_m'] = stations_utm.geometry.distance(point_utm.geometry.iloc[0])
        stations_utm['distance_km'] = stations_utm['distance_m'] / 1000.0
        
        # Filter to stations within max distance
        within_range = stations_utm[stations_utm['distance_km'] <= max_distance_km].copy()
        
        if len(within_range) == 0:
            return None
        
        # Prioritize mapped stations (those with snotelr_site_id)
        mapped_stations = within_range[within_range['snotelr_site_id'].notna()]
        
        if len(mapped_stations) > 0:
            # Use nearest mapped station
            nearest_mapped = mapped_stations.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest_mapped.name]  # Use original index
            distance_km = nearest_mapped['distance_km']
        else:
            # No mapped stations available, use nearest unmapped station
            # (will fall back to elevation estimate anyway)
            nearest_unmapped = within_range.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest_unmapped.name]
            distance_km = nearest_unmapped['distance_km']
            logger.debug(f"Using unmapped station {station['name']} (no mapped stations within {max_distance_km} km)")
        
        return {
            "triplet": station["triplet"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "elevation_ft": station.get("elevation_ft", 0),
            "distance_km": distance_km,
            "snotelr_site_id": station.get("snotelr_site_id")  # May be None if not mapped
        }
    
    def get_snow_data(self, lat: float, lon: float, date: datetime, elevation_ft: Optional[float] = None) -> Dict:
        """
        Get snow data for location and date.
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date for snow data
            elevation_ft: Optional elevation in feet (from DEM). If provided, used for
                         better elevation-based estimation when no station is nearby.
        """
        import pandas as pd  # Import at function level for use throughout
        
        # Check request cache first (fast path for exact same location/date)
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.request_cache:
            logger.debug(f"Cache hit for location ({lat:.4f}, {lon:.4f}) on {date.strftime('%Y-%m-%d')}")
            return self.request_cache[cache_key]
        
        # Find nearest SNOTEL station
        station = self._find_nearest_station(lat, lon)
        
        if station is None:
            # No station nearby, use elevation-based estimate
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
            self.request_cache[cache_key] = result
            return result
        
        # Fetch data using snotelr R package
        if self.snotelr is None:
            # Only warn once per DataContextBuilder instance to avoid log spam
            if not self._snotelr_warned:
                logger.warning("snotelr not available, using elevation estimate for all snow data")
                logger.debug("This warning will only appear once per DataContextBuilder instance")
                self._snotelr_warned = True
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
            self.request_cache[cache_key] = result
            return result
        
        try:
            # Get snotelr site_id from mapped station data
            snotelr_site_id = station.get("snotelr_site_id")
            
            if snotelr_site_id is None or pd.isna(snotelr_site_id):
                # Station not mapped to snotelr site_id
                logger.warning(f"Station {station['name']} has no snotelr site_id mapping")
                logger.info("Run scripts/map_snotel_station_ids.py to create mapping")
                # Fall back to elevation estimate with available elevation
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
                self.request_cache[cache_key] = result
                return result
            
            snotelr_site_id = int(snotelr_site_id)
            
            # Convert date to R Date format
            date_str = date.strftime("%Y-%m-%d")
            
            # Check station data cache first - this is the key optimization!
            # snotel_download() returns the ENTIRE historical record for a station,
            # so we cache by station ID only (not station + date)
            # Multiple locations can use the same station, so we only download once per station
            station_cache_key = snotelr_site_id
            
            if station_cache_key in self.station_data_cache:
                # Use cached station data - no download needed!
                logger.debug(f"Using cached data for station {station['name']} (site_id {snotelr_site_id})")
                df = self.station_data_cache[station_cache_key]
            else:
                # Need to download station data
                logger.debug(f"Downloading historical data for station {station['name']} (site_id {snotelr_site_id})")
                
                # Get the download function
                snotel_download = self.snotelr.snotel_download
                
                # Call snotelr::snotel_download() with mapped site_id
                # This downloads the entire historical record for the station (one time per station)
                r_data = snotel_download(snotelr_site_id, internal=True)
                
                # Convert R data frame to pandas using new rpy2 API
                from rpy2.robjects import pandas2ri  # type: ignore
                from rpy2.robjects.conversion import localconverter  # type: ignore
                
                # Use context manager instead of activate/deactivate
                with localconverter(pandas2ri.converter):
                    df = pandas2ri.rpy2py(r_data)
                
                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                
                # Cache the full station dataframe - this is the key optimization!
                # Now any other location/date querying this same station can reuse the data
                self.station_data_cache[station_cache_key] = df
                logger.debug(f"Cached full station data for {station['name']} (site_id {snotelr_site_id}) - {len(df)} records covering {df['date'].min()} to {df['date'].max()}")
            
            # Filter to the specific date from cached dataframe
            date_data = df[df['date'].dt.date == date.date()]
            
            if len(date_data) == 0:
                # No data for this exact date, try to get closest date
                logger.debug(f"No data for {date_str}, using closest available date")
                # Get data within ±7 days
                date_range = pd.date_range(date - timedelta(days=7), date + timedelta(days=7))
                date_data = df[df['date'].isin(date_range)]
                if len(date_data) > 0:
                    # Use closest date
                    # idxmin() returns an index label, so use .loc with the label (not .iloc with position)
                    closest_date_label = (date_data['date'] - pd.Timestamp(date)).abs().idxmin()
                    date_data = date_data.loc[[closest_date_label]]
            
            if len(date_data) == 0:
                logger.warning(f"No SNOTEL data available for station {station['name']} (site_id {snotelr_site_id}) near {date_str}")
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
                self.request_cache[cache_key] = result
                return result
            
            # Extract snow depth and SWE
            # snotelr column names: snow_water_equivalent (SWE in mm), snow_depth (in mm)
            row = date_data.iloc[0]
            
            # Convert from mm to inches
            swe_mm = row.get('snow_water_equivalent', 0)
            if pd.isna(swe_mm):
                swe_mm = 0
            swe = float(swe_mm) / 25.4  # mm to inches
            
            # Snow depth (if available)
            snow_depth_mm = row.get('snow_depth', None)
            if snow_depth_mm is not None and not pd.isna(snow_depth_mm):
                snow_depth = float(snow_depth_mm) / 25.4  # mm to inches
            else:
                # Estimate snow depth from SWE (typical density ~0.25)
                snow_depth = swe / 0.25 if swe > 0 else 0.0
            
            # Detect crusting (SWE high relative to depth)
            density = swe / snow_depth if snow_depth > 0 else 0
            crust_detected = density > 0.35  # High density = crust
            
            result = {
                "depth": float(snow_depth),
                "swe": float(swe),
                "crust": crust_detected,
                "station": station["name"],
                "station_distance_km": station["distance_km"]
            }
            
            # Cache result for this specific location/date (fast lookup)
            self.request_cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.warning(f"Error retrieving SNOTEL data via snotelr: {e}, using elevation estimate")
            import traceback
            logger.debug(traceback.format_exc())
            # Pass elevation if available for better estimation
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
            self.request_cache[cache_key] = result
            return result
    
    def _estimate_snow_from_elevation(self, lat: float, lon: float, 
                                     date: datetime, elevation_ft: Optional[float] = None) -> Dict:
        """
        Estimate snow based on elevation and date (fallback).
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date for estimation
            elevation_ft: Optional elevation in feet. If None, attempts to sample from DEM,
                         otherwise defaults to 8500 ft.
        """
        # Use provided elevation, or try to sample from DEM, or use default
        if elevation_ft is not None:
            elevation = elevation_ft
        else:
            # Try to sample from DEM if available
            elevation = self._get_elevation_from_dem(lat, lon)
        
        month = date.month
        
        # Simple heuristic
        if month in [12, 1, 2]:  # Winter
            snow_depth = max(0, (elevation - 6000) / 100)
        elif month in [3, 4]:  # Spring
            snow_depth = max(0, (elevation - 7000) / 150)
        elif month in [10, 11]:  # Fall
            snow_depth = max(0, (elevation - 9000) / 200)
        else:  # Summer
            snow_depth = max(0, (elevation - 10000) / 100)
        
        return {
            "depth": snow_depth,
            "swe": snow_depth * 0.25,
            "crust": False,
            "station": None
        }
    
    def _get_elevation_from_dem(self, lat: float, lon: float) -> float:
        """
        Attempt to get elevation from DEM if available.
        
        Falls back to default elevation if DEM not accessible.
        """
        try:
            # Try to use DEM if available in data_dir
            dem_path = self.data_dir / "dem" / "wyoming_dem.tif"
            if dem_path.exists():
                try:
                    import rasterio
                    from rasterio.warp import transform_geom
                    from shapely.geometry import Point, mapping
                    
                    # Sample DEM at location
                    point = Point(lon, lat)
                    geom = mapping(point)
                    
                    # Transform to DEM CRS if needed (assuming DEM is in UTM or similar)
                    # For simplicity, assume DEM is in EPSG:4326 or can be sampled directly
                    with rasterio.open(dem_path) as src:
                        # Sample the point
                        for val in src.sample([(lon, lat)]):
                            if val[0] is not None and not (val[0] < -1000 or val[0] > 15000):
                                # Convert meters to feet if needed
                                # Assume DEM is in meters, convert to feet
                                elevation_m = float(val[0])
                                if elevation_m < 500:  # Likely in feet already
                                    return elevation_m
                                else:  # Likely in meters
                                    return elevation_m * 3.28084  # meters to feet
                except Exception as e:
                    logger.debug(f"Could not sample DEM for elevation: {e}")
                    pass
            
        except Exception as e:
            logger.debug(f"Error accessing DEM for elevation: {e}")
            pass
        
        # Default fallback
        return 8500.0


class WeatherClient:
    """Client for weather data"""
    
    def __init__(self):
        self.api_key = None  # Set from environment
        self.cache = {}
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather for location and date"""
        
        # Check if date is in future (forecast) or past (historical)
        today = datetime.now().date()
        target_date = date.date()
        
        if target_date > today:
            return self._get_forecast(lat, lon, date)
        else:
            return self._get_historical(lat, lon, date)
    
    def _get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast"""
        # In production: use NOAA API, weather.gov
        # Placeholder with reasonable defaults
        
        return {
            "temp": 45.0,
            "temp_high": 55.0,
            "temp_low": 35.0,
            "precip_7d": 0.3,
            "cloud_cover": 30,
            "wind_mph": 10
        }
    
    def _get_historical(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical weather"""
        # In production: use NOAA NCDC, weather archives
        # Placeholder
        
        return {
            "temp": 42.0,
            "temp_high": 52.0,
            "temp_low": 32.0,
            "precip_7d": 0.5,
            "cloud_cover": 40,
            "wind_mph": 12
        }


class SatelliteClient:
    """Client for satellite imagery (NDVI, etc.)"""
    
    def __init__(self):
        self.cache = {}
    
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get NDVI for location and date"""
        # In production: use Google Earth Engine, Landsat, Sentinel
        
        # Placeholder with seasonal variation
        month = date.month
        
        if month in [6, 7, 8]:  # Summer - high NDVI
            ndvi = 0.70
        elif month in [9, 10]:  # Fall - declining
            ndvi = 0.55
        elif month in [11, 12, 1, 2, 3]:  # Winter - low
            ndvi = 0.30
        else:  # Spring - increasing
            ndvi = 0.50
        
        return {
            "ndvi": ndvi,
            "age_days": 8,
            "irg": 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0,
            "cloud_free": True
        }
    
    def get_integrated_ndvi(self, lat: float, lon: float, 
                           start_date: datetime, end_date: datetime) -> float:
        """Get integrated NDVI over date range"""
        # In production: sum NDVI across all dates
        
        # Placeholder
        return 60.0  # Typical summer iNDVI