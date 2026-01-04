from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
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
        self.snotel_client = AWDBClient(self.data_dir)
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
                # Ensure scalar values to avoid deprecation warning
                x, y = transformer.transform(float(lon), float(lat))
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


class AWDBClient:
    """Client for accessing SNOTEL snow data using AWDB REST API"""
    
    # AWDB REST API base URL
    AWDB_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
    
    # Class-level sets to track warnings across all instances (shared by all workers)
    # This prevents duplicate warnings when multiple workers process the same stations
    _warned_stations = set()  # Set of (station_id, warning_type) tuples
    _warned_api_failures = set()  # Set of station_ids that have failed API calls
    _warning_lock = None  # Will be initialized on first use for thread safety
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.station_cache_path = self.cache_dir / "snotel_stations_wyoming.geojson"
        self._stations_gdf = None
        
        # Two-level caching:
        # 1. Station data cache: station_id -> DataFrame with date range data
        #    This avoids re-downloading the same station data for multiple locations/dates
        #    We cache by station ID and date range to minimize API calls
        # 2. Request cache: (lat, lon, date) -> final result dict
        #    This provides fast lookup for exact same location/date queries
        self.station_data_cache = {}  # Cache station data: {station_id: DataFrame}
        self.request_cache = {}  # Cache final results: {(lat, lon, date_str): result_dict}
        
        # Cache size limits to prevent unbounded memory growth
        self.MAX_STATION_CACHE_SIZE = 100  # Keep only 100 most recent stations
        self.MAX_REQUEST_CACHE_SIZE = 10000  # Keep only 10k most recent requests
        
        # Initialize lock for thread-safe warning tracking (lazy initialization)
        if AWDBClient._warning_lock is None:
            import threading
            AWDBClient._warning_lock = threading.Lock()
        
        # Load stations (from local cache or AWDB API)
        self._load_stations()
    
    def _trim_caches(self):
        """
        Trim caches to prevent unbounded memory growth.
        Uses LRU-style eviction (removes oldest entries first).
        """
        # Trim station data cache
        if len(self.station_data_cache) > self.MAX_STATION_CACHE_SIZE:
            excess = len(self.station_data_cache) - self.MAX_STATION_CACHE_SIZE
            # Remove oldest entries (simple: remove first N keys)
            keys_to_remove = list(self.station_data_cache.keys())[:excess]
            for key in keys_to_remove:
                del self.station_data_cache[key]
            logger.debug(f"Trimmed station cache: removed {excess} entries, {len(self.station_data_cache)} remaining")
        
        # Trim request cache
        if len(self.request_cache) > self.MAX_REQUEST_CACHE_SIZE:
            excess = len(self.request_cache) - self.MAX_REQUEST_CACHE_SIZE
            # Remove oldest entries (simple: remove first N keys)
            keys_to_remove = list(self.request_cache.keys())[:excess]
            for key in keys_to_remove:
                del self.request_cache[key]
            logger.debug(f"Trimmed request cache: removed {excess} entries, {len(self.request_cache)} remaining")
    
    def _load_stations_from_awdb(self) -> bool:
        """
        Load all Wyoming SNOTEL stations from AWDB API.
        
        Returns:
            True if successful, False otherwise
        """
        import traceback
        import os
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Skip API call in test environment to avoid slow tests
            if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING'):
                logger.debug("Skipping AWDB API call in test environment")
                return False
            
            # Query AWDB API for all Wyoming SNOTEL stations
            params = {
                'stationTriplets': '*:WY:SNTL',
                'returnForecastPointMetadata': 'false',
                'returnReservoirMetadata': 'false',
                'returnStationElements': 'false',
                'activeOnly': 'true',
            }
            
            response = requests.get(f"{self.AWDB_BASE_URL}/stations", params=params, timeout=30)
            response.raise_for_status()
            stations_data = response.json()
            
            if not stations_data:
                logger.warning("No Wyoming SNOTEL stations found in AWDB API")
                return False
            
            # Convert to GeoDataFrame
            features = []
            for station in stations_data:
                station_id = str(station.get('stationId', ''))
                name = station.get('name', 'Unknown')
                lat = station.get('latitude')
                lon = station.get('longitude')
                elevation_ft = station.get('elevation', 0)  # AWDB returns in feet
                triplet = station.get('stationTriplet', f"{station_id}:WY:SNTL")
                
                if lat is None or lon is None:
                    logger.debug(f"Skipping station {name} (missing coordinates)")
                    continue
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "station_id": station_id,
                        "triplet": triplet,
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "elevation_ft": elevation_ft,
                        "state": "WY",
                        "awdb_station_id": station_id,  # AWDB station ID (same as snotelr_site_id)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                }
                features.append(feature)
            
            if not features:
                logger.warning("No valid Wyoming SNOTEL stations found in AWDB API response")
                return False
            
            # Create GeoDataFrame
            self._stations_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            
            # Save to cache
            self._stations_gdf.to_file(self.station_cache_path, driver="GeoJSON")
            logger.info(f"Loaded {len(self._stations_gdf)} Wyoming SNOTEL stations from AWDB API")
            return True
            
        except Exception as e:
            logger.warning(f"Error loading stations from AWDB API: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_stations(self):
        """
        Load SNOTEL station locations from AWDB REST API.
        
        Uses local cache if available and recent (within 24 hours), otherwise
        fetches fresh station data from AWDB API. This ensures we have the latest
        active stations while avoiding redundant API calls during parallel processing.
        """
        import time
        from datetime import datetime, timedelta
        
        # Check if cache file exists and is recent (less than 24 hours old)
        cache_valid = False
        if self.station_cache_path.exists():
            try:
                cache_age = time.time() - self.station_cache_path.stat().st_mtime
                cache_age_hours = cache_age / 3600
                if cache_age_hours < 24:
                    cache_valid = True
                    logger.debug(f"Station cache file is {cache_age_hours:.1f} hours old, using cache")
            except Exception as e:
                logger.debug(f"Could not check cache file age: {e}")
        
        if cache_valid:
            # Load from cache
            try:
                import geopandas as gpd
                logger.debug(f"Loading Wyoming SNOTEL stations from cache ({self.station_cache_path.name})...")
                self._stations_gdf = gpd.read_file(self.station_cache_path)
                logger.debug(f"Loaded {len(self._stations_gdf)} Wyoming SNOTEL stations from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load stations from cache: {e}, fetching from API")
                # Fall through to API fetch
        
        # Cache doesn't exist or is stale, fetch from API
        logger.info("Loading Wyoming SNOTEL stations from AWDB API...")
        if not self._load_stations_from_awdb():
            logger.warning("Failed to load stations from AWDB API, falling back to elevation estimates")
            self._stations_gdf = None
    
    def _find_nearest_station(self, lat: float, lon: float, max_distance_km: float = 100.0):
        """
        Find nearest SNOTEL station to location.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum distance to search (default: 100 km)
        
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
        
        # Prioritize mapped stations (those with awdb_station_id) over unmapped ones
        # Get station ID column (prefer awdb_station_id, fall back to snotelr_site_id for backward compatibility)
        if 'awdb_station_id' in self._stations_gdf.columns:
            station_id_col = 'awdb_station_id'
        elif 'snotelr_site_id' in self._stations_gdf.columns:
            station_id_col = 'snotelr_site_id'
        else:
            # No station ID column, use all stations
            station_id_col = None
        
        if station_id_col:
            # Filter to mapped stations (those with non-null station ID)
            station_ids = self._stations_gdf.iloc[within_range.index][station_id_col]
            mapped_mask = station_ids.notna()
            mapped_stations = within_range[mapped_mask]
        else:
            mapped_stations = within_range
        
        if len(mapped_stations) > 0:
            # Use nearest mapped station
            nearest_mapped = mapped_stations.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest_mapped.name]  # Use original index
            distance_km = nearest_mapped['distance_km']
        else:
            # No mapped stations available, use nearest unmapped station
            nearest = within_range.nsmallest(1, 'distance_km').iloc[0]
            station = self._stations_gdf.iloc[nearest.name]  # Use original index
            distance_km = nearest['distance_km']
            logger.debug(f"Using unmapped station {station['name']} (no mapped stations within {max_distance_km} km)")
        
        # Get station ID (prefer awdb_station_id, fall back to snotelr_site_id for backward compatibility)
        station_id = station.get("awdb_station_id") or station.get("snotelr_site_id")
        
        return {
            "triplet": station.get("triplet", f"{station_id}:WY:SNTL" if station_id else None),
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "elevation_ft": station.get("elevation_ft", 0),
            "distance_km": distance_km,
            "awdb_station_id": station_id,  # AWDB station ID (same format as snotelr_site_id)
            "AWDB_site_id": station_id  # Alias for backward compatibility with tests
        }
    
    def get_snow_data(self, lat: float, lon: float, date: datetime, elevation_ft: Optional[float] = None) -> Dict:
        """
        Get snow data for location and date using AWDB REST API.
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date for snow data
            elevation_ft: Optional elevation in feet (from DEM). If provided, used for
                         better elevation-based estimation when no station is nearby.
        
        Returns:
            Dictionary with snow data: depth, swe, crust, station, station_distance_km
        """
        import pandas as pd
        import traceback
        
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
            self._trim_caches()  # Prevent unbounded growth
            return result
        
        try:
            # Get AWDB station ID
            station_id = station.get("awdb_station_id")
            
            if station_id is None or pd.isna(station_id):
                # Station not mapped to AWDB station ID
                logger.debug(f"Station {station['name']} has no AWDB station ID, using elevation estimate")
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
                self.request_cache[cache_key] = result
                self._trim_caches()  # Prevent unbounded growth
                return result
            
            station_id = str(station_id)
            date_str = date.strftime("%Y-%m-%d")
            
            # Check station data cache first
            # We cache by station ID and date range to minimize API calls
            # For efficiency, we fetch a date range around the requested date
            station_cache_key = station_id
            
            # Determine date range for caching (fetch ±30 days to cache more data)
            begin_date = (date - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = (date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            if station_cache_key in self.station_data_cache:
                # Check if cached data covers the requested date
                df = self.station_data_cache[station_cache_key]
                df_dates = pd.to_datetime(df['date'])
                if df_dates.min() <= pd.Timestamp(date) <= df_dates.max():
                    # Use cached data
                    logger.debug(f"Using cached data for station {station['name']} (station_id {station_id})")
                else:
                    # Cached data doesn't cover this date, fetch new data
                    logger.debug(f"Cached data for station {station['name']} doesn't cover {date_str}, fetching from API")
                    df = self._fetch_station_data_from_awdb(station_id, begin_date, end_date)
                    if df is not None:
                        self.station_data_cache[station_cache_key] = df
                        self._trim_caches()  # Prevent unbounded growth
            else:
                # Need to fetch station data
                logger.debug(f"Fetching data for station {station['name']} (station_id {station_id}) from AWDB API")
                df = self._fetch_station_data_from_awdb(station_id, begin_date, end_date)
                if df is not None:
                    self.station_data_cache[station_cache_key] = df
                    self._trim_caches()  # Prevent unbounded growth
            
            if df is None or len(df) == 0:
                # Only warn once per station to avoid log spam (shared across all instances)
                warning_key = (station_id, 'no_data')
                with AWDBClient._warning_lock:
                    if warning_key not in AWDBClient._warned_stations:
                        logger.warning(f"No AWDB data available for station {station['name']} (station_id {station_id})")
                        AWDBClient._warned_stations.add(warning_key)
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
                self.request_cache[cache_key] = result
                return result
            
            # Filter to the specific date
            df['date'] = pd.to_datetime(df['date'])
            date_data = df[df['date'].dt.date == date.date()]
            
            if len(date_data) == 0:
                # No data for this exact date, try to get closest date within ±7 days
                logger.debug(f"No data for {date_str}, using closest available date")
                date_range = pd.date_range(date - timedelta(days=7), date + timedelta(days=7))
                date_data = df[df['date'].isin(date_range)]
                if len(date_data) > 0:
                    # Use closest date
                    closest_date_label = (date_data['date'] - pd.Timestamp(date)).abs().idxmin()
                    date_data = date_data.loc[[closest_date_label]]
            
            if len(date_data) == 0:
                # Only warn once per station to avoid log spam (shared across all instances)
                warning_key = (station_id, 'no_date_data')
                with AWDBClient._warning_lock:
                    if warning_key not in AWDBClient._warned_stations:
                        logger.warning(f"No AWDB data available for station {station['name']} (station_id {station_id}) near {date_str}")
                        AWDBClient._warned_stations.add(warning_key)
                result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
                self.request_cache[cache_key] = result
                return result
            
            # Extract snow depth and SWE from AWDB response
            # AWDB returns data in inches (not mm!)
            row = date_data.iloc[0]
            
            # AWDB element codes: WTEQ (Snow Water Equivalent), SNWD (Snow Depth)
            swe = row.get('WTEQ', 0)
            if pd.isna(swe):
                swe = 0
            swe = float(swe)  # Already in inches!
            
            # Snow depth (if available)
            snow_depth = row.get('SNWD', None)
            if snow_depth is not None and not pd.isna(snow_depth):
                snow_depth = float(snow_depth)  # Already in inches!
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
            self._trim_caches()  # Prevent unbounded growth
            return result
        
        except Exception as e:
            logger.warning(f"Error retrieving SNOTEL data via AWDB API: {e}, using elevation estimate")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Pass elevation if available for better estimation
            result = self._estimate_snow_from_elevation(lat, lon, date, elevation_ft=elevation_ft)
            self.request_cache[cache_key] = result
            self._trim_caches()  # Prevent unbounded growth
            return result
    
    def _fetch_station_data_from_awdb(self, station_id: str, begin_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch snow data for a station from AWDB API.
        
        Args:
            station_id: AWDB station ID
            begin_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: date, WTEQ, SNWD, or None if error
        """
        import pandas as pd
        import traceback
        import os
        
        # Skip API call in test environment to avoid slow tests
        if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING'):
            logger.debug(f"Skipping AWDB API call in test environment for station {station_id}")
            return None
        
        try:
            # Construct station triplet
            triplet = f"{station_id}:WY:SNTL"
            
            # Query AWDB API with retry logic for 5xx errors
            params = {
                'stationTriplets': triplet,
                'elements': 'WTEQ,SNWD',
                'beginDate': begin_date,
                'endDate': end_date,
                'duration': 'DAILY',
                'returnFlags': 'false',
                'returnOriginalValues': 'false',
                'returnSuspectData': 'false',
            }
            
            # Retry logic with exponential backoff for 5xx errors, timeouts, and connection errors
            max_retries = 3
            base_delay = 1.0  # Start with 1 second
            import time
            import threading
            
            # Rate limiting: ensure minimum 100ms between API requests (shared across all instances)
            if not hasattr(AWDBClient, '_rate_limit_lock'):
                AWDBClient._rate_limit_lock = threading.Lock()
                AWDBClient._last_request_time = 0
                AWDBClient.MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests (10 req/sec max)
            
            for attempt in range(max_retries + 1):
                try:
                    # Rate limiting: ensure minimum interval between requests
                    # CRITICAL: Only hold lock for time check, not during API call!
                    with AWDBClient._rate_limit_lock:
                        elapsed = time.time() - AWDBClient._last_request_time
                        if elapsed < AWDBClient.MIN_REQUEST_INTERVAL:
                            sleep_time = AWDBClient.MIN_REQUEST_INTERVAL - elapsed
                            time.sleep(sleep_time)
                        # Update last request time BEFORE making the call
                        AWDBClient._last_request_time = time.time()
                    
                    # Make API call OUTSIDE the lock to avoid serializing all workers
                    # Reduced timeout from 30s to 10s for faster failure detection
                    response = requests.get(f"{self.AWDB_BASE_URL}/data", params=params, timeout=10)
                    
                    # Check for 5xx errors before raising
                    # Use getattr to safely check status_code (handles mocks in tests)
                    status_code = getattr(response, 'status_code', None)
                    if status_code is not None and isinstance(status_code, int) and status_code >= 500:
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                            logger.debug(f"AWDB API 5xx error ({status_code}) for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        else:
                            # Final attempt failed, raise the error
                            response.raise_for_status()
                    
                    # For non-5xx errors, raise immediately if there's an error
                    response.raise_for_status()
                    # Success - break out of retry loop
                    break
                    
                except requests.exceptions.HTTPError as e:
                    # HTTPError from raise_for_status() - check if it's a 5xx error
                    if hasattr(e, 'response') and e.response is not None:
                        error_status_code = getattr(e.response, 'status_code', None)
                        if error_status_code is not None and isinstance(error_status_code, int) and error_status_code >= 500:
                            if attempt < max_retries:
                                delay = base_delay * (2 ** attempt)
                                logger.debug(f"AWDB API 5xx error ({error_status_code}) for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                                time.sleep(delay)
                                continue
                    # Non-5xx HTTP error or final attempt - re-raise
                    raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    # Retry timeouts and connection errors (network issues are often transient)
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        error_type = "timeout" if isinstance(e, requests.exceptions.Timeout) else "connection error"
                        logger.debug(f"AWDB API {error_type} for station {station_id} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    # Final attempt failed - re-raise
                    raise
                except requests.exceptions.RequestException as e:
                    # Other request exceptions - don't retry
                    raise
            
            # If we get here, we should have a successful response
            data = response.json()
            
            if not data or len(data) == 0:
                logger.debug(f"No data returned from AWDB API for station {station_id}")
                return None
            
            # Parse AWDB response structure
            # Response is: [{"stationTriplet": "...", "data": [{"stationElement": {...}, "values": [...]}]}]
            station_data = data[0]
            elements_data = station_data.get('data', [])
            
            # Build DataFrame from response
            records = []
            wteq_values = {}
            snwd_values = {}
            
            for element_data in elements_data:
                element_code = element_data['stationElement']['elementCode']
                values = element_data.get('values', [])
                
                for value_obj in values:
                    date_str = value_obj['date']
                    value = value_obj.get('value')
                    
                    if element_code == 'WTEQ':
                        wteq_values[date_str] = value
                    elif element_code == 'SNWD':
                        snwd_values[date_str] = value
            
            # Combine all dates
            all_dates = set(wteq_values.keys()) | set(snwd_values.keys())
            
            for date_str in sorted(all_dates):
                records.append({
                    'date': date_str,
                    'WTEQ': wteq_values.get(date_str),
                    'SNWD': snwd_values.get(date_str),
                })
            
            if not records:
                logger.debug(f"No data records parsed from AWDB API response for station {station_id}")
                return None
            
            df = pd.DataFrame(records)
            logger.debug(f"Fetched {len(df)} records from AWDB API for station {station_id} ({begin_date} to {end_date})")
            return df
            
        except requests.exceptions.RequestException as e:
            # Only warn once per station to avoid log spam (shared across all instances)
            with AWDBClient._warning_lock:
                if station_id not in AWDBClient._warned_api_failures:
                    logger.warning(f"AWDB API request failed for station {station_id}: {e}")
                    AWDBClient._warned_api_failures.add(station_id)
            return None
        except Exception as e:
            logger.warning(f"Error parsing AWDB API response for station {station_id}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
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