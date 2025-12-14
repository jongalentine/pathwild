from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, shape
import requests
from functools import lru_cache

class DataContextBuilder:
    """Builds comprehensive context for heuristic calculations"""
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir or (self.data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load static layers
        self._load_static_layers()
        
        # Initialize data loaders
        self.snotel_client = SNOTELClient()
        self.weather_client = WeatherClient()
        self.satellite_client = SatelliteClient()
    
    def _load_static_layers(self):
        """Load data that doesn't change over time"""
        
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
            print(f"  ✓ Water sources loaded: {len(self.water_sources)} features")
        else:
            print(f"  ✗ Water sources not found")
            self.water_sources = None
        
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
        context["elevation"] = self._sample_raster(self.dem, lon, lat, default=8500.0)
        context["slope_degrees"] = self._sample_raster(self.slope, lon, lat, default=15.0)
        context["aspect_degrees"] = self._sample_raster(self.aspect, lon, lat, default=180.0)
        context["canopy_cover_percent"] = self._sample_raster(
            self.canopy, lon, lat, default=30.0
        )
        
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
        
        # Snow data
        snow_data = self.snotel_client.get_snow_data(lat, lon, dt)
        context["snow_depth_inches"] = snow_data.get("depth", 0.0)
        context["snow_water_equiv_inches"] = snow_data.get("swe", 0.0)
        context["snow_crust_detected"] = snow_data.get("crust", False)
        
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
        if raster is None:
            return default
        
        try:
            # Convert lon/lat to raster coordinates
            row, col = raster.index(lon, lat)
            
            # Read value
            window = rasterio.windows.Window(col, row, 1, 1)
            data = raster.read(1, window=window)
            
            value = float(data[0, 0])
            
            # Check for nodata
            if value == raster.nodata or np.isnan(value):
                return default
            
            return value
        except Exception as e:
            return default
    
    def _calculate_water_metrics(self, point: Point) -> Tuple[float, float]:
        """Calculate distance to water and reliability"""
        from shapely.ops import nearest_points
        
        # Find nearest water source
        nearest_geom = nearest_points(point, self.water_sources.unary_union)[1]
        distance_m = point.distance(nearest_geom) * 111139  # degrees to meters (approx)
        distance_mi = distance_m / 1609.34
        
        # Get water source attributes
        nearest_feature = self.water_sources[
            self.water_sources.geometry == nearest_geom
        ].iloc[0]
        
        # Reliability: 1.0 for springs/lakes, 0.7 for streams, 0.4 for ephemeral
        water_type = nearest_feature.get("type", "stream").lower()
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
        """Calculate distance to nearest feature in miles"""
        from shapely.ops import nearest_points
        
        nearest_geom = nearest_points(point, features.unary_union)[1]
        distance_m = point.distance(nearest_geom) * 111139
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
        
        try:
            # Extract slope within buffer
            from rasterio.mask import mask as rio_mask
            
            geom = [buffered.__geo_interface__]
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
    
    def _decode_landcover(self, code: int) -> str:
        """Decode NLCD land cover code to description"""
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
    """Client for accessing SNOTEL snow data"""
    
    def __init__(self):
        self.base_url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
        self.cache = {}
    
    def get_snow_data(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get snow data for location and date"""
        
        # Find nearest SNOTEL station
        station = self._find_nearest_station(lat, lon)
        
        if station is None:
            # No station nearby, use model or defaults
            return self._estimate_snow_from_elevation(lat, lon, date)
        
        # Fetch data from SNOTEL API
        try:
            params = {
                "stationTriplets": station["triplet"],
                "elementCd": "SNWD,WTEQ",  # Snow depth, SWE
                "ordinal": 1,
                "duration": "DAILY",
                "getFlags": False,
                "beginDate": date.strftime("%Y-%m-%d"),
                "endDate": date.strftime("%Y-%m-%d")
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            snow_depth = 0.0
            swe = 0.0
            
            for item in data:
                if item["elementCd"] == "SNWD":
                    snow_depth = item.get("value", 0.0)
                elif item["elementCd"] == "WTEQ":
                    swe = item.get("value", 0.0)
            
            # Detect crusting (SWE high relative to depth)
            density = swe / snow_depth if snow_depth > 0 else 0
            crust_detected = density > 0.35  # High density = crust
            
            return {
                "depth": snow_depth,
                "swe": swe,
                "crust": crust_detected,
                "station": station["name"]
            }
        
        except Exception as e:
            print(f"SNOTEL fetch error: {e}")
            return self._estimate_snow_from_elevation(lat, lon, date)
    
    def _find_nearest_station(self, lat: float, lon: float):
        """Find nearest SNOTEL station"""
        # Simplified - in production, query station database
        # For now, return None to use estimates
        return None
    
    def _estimate_snow_from_elevation(self, lat: float, lon: float, 
                                     date: datetime) -> Dict:
        """Estimate snow based on elevation and date"""
        # Placeholder - simple elevation-based model
        # In production: use snow reanalysis products (SNODAS, etc.)
        
        elevation = 8500  # Default
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
            "crust": False
        }


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