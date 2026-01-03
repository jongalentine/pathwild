# Temporal Data Integration Guide: SNOTEL, Weather, and NDVI

This guide provides step-by-step instructions for integrating real-time and historical data sources for SNOTEL/snow, weather, and NDVI/satellite data into PathWild.

## Overview

Current status:
1. ✅ **SNOTEL/Snow Data** - **COMPLETE** - Real snow depth and SWE from USDA SNOTEL stations via `snotelr` R package
2. ⚠️ **Weather Data** - **PLACEHOLDER** - Currently using placeholder values (Open-Meteo integration pending)
3. ⚠️ **NDVI/Satellite Data** - **PLACEHOLDER** - Currently using placeholder values (Google Earth Engine integration pending)

---

## Prerequisites

Before starting, ensure you have:
- Python 3.11+ environment activated
- Basic packages: `requests`, `pandas`, `numpy`
- Optional but recommended: `earthengine-api` (for NDVI via Google Earth Engine)

---

## Part 1: SNOTEL/Snow Data Integration ✅ COMPLETE

### Status: ✅ Fully Implemented

SNOTEL integration is complete and working. The system uses the `snotelr` R package via `rpy2` to download real SNOTEL station data.

**Key Features:**
- Real SNOTEL data from 36 Wyoming stations
- Two-level caching optimization (station-level and request-level)
- Station ID mapping between USDA triplets and snotelr site IDs
- Elevation-based fallback using actual DEM elevation (not hardcoded)
- Data quality tracking fields (`snow_data_source`, `snow_station_name`, `snow_station_distance_km`)

### Step 1.1: Prerequisites

**Required Packages:**
- `rpy2` - Python interface to R
- `r-snotelr` - R package for SNOTEL data (installed via conda)
- `geopandas` - For spatial operations
- `pandas` - For data manipulation

These are already included in `environment.yml`.

**R Setup:**
See `docs/snotelr_setup.md` for detailed R and snotelr installation instructions.

### Step 1.2: Station Database and ID Mapping

**Current Implementation:**

The SNOTEL station database is already set up:
- **Station file**: `data/cache/snotel_stations_wyoming.geojson`
  - Contains 36 Wyoming SNOTEL stations
  - Includes USDA station triplets, coordinates, and elevations
  - Includes `snotelr_site_id` mapping (where available)

**Station ID Mapping:**

The system uses `scripts/map_snotel_station_ids.py` to map USDA station triplets to `snotelr` site IDs. This mapping is critical because `snotelr` uses its own internal site ID system that doesn't match USDA triplets.

**Mapping Status:**
- 31 out of 36 stations (86%) are successfully mapped
- 5 stations are unmapped (not available in snotelr database)
- Unmapped stations fall back to elevation-based estimates

**To update mappings:**
```bash
python scripts/map_snotel_station_ids.py
```

This uses fuzzy matching and location-based matching to maximize coverage.

See `docs/snotelr_setup.md` for detailed setup instructions.

### Step 1.3: Current SNOTELClient Implementation

The `SNOTELClient` in `src/data/processors.py` is fully implemented. Key features:

**Key Implementation Details:**

1. **Two-Level Caching:**
   - `station_data_cache`: Caches full historical records by station ID (keyed by `site_id` only)
   - `request_cache`: Caches final results by location/date (keyed by `lat,lon,date`)
   - **Performance**: For 100 locations near the same station, only 1 download instead of 100

2. **Data Retrieval:**
   - Uses `snotelr::snotel_download()` via `rpy2` to get full historical records
   - Extracts specific dates from cached station data
   - Falls back to elevation-based estimates when no station is within 100 km

3. **Station Finding:**
   - Prioritizes mapped stations (those with `snotelr_site_id`) within 100 km
   - Falls back to unmapped stations only if no mapped ones available
   - Uses UTM projection for accurate distance calculations

4. **Elevation-Based Fallback:**
   - Uses actual DEM elevation (sampled from `wyoming_dem.tif`)
   - Falls back to 8500 ft only if DEM is unavailable or location outside bounds
   - Seasonal adjustments based on elevation and month
   - Logs warnings when default elevation is used (indicates potential DEM boundary issues)

5. **Data Quality Tracking:**
   - `snow_data_source`: "snotel" or "estimate"
   - `snow_station_name`: Station name (None for estimates)
   - `snow_station_distance_km`: Distance to station (None for estimates)

**Code Location:** `src/data/processors.py` - `SNOTELClient` class (lines ~614-900)

### Step 1.4: Testing SNOTEL Integration

**Unit and Integration Tests:**

Comprehensive tests are available in `tests/test_snotel_integration.py`:
- Unit tests for station finding, caching, and elevation estimation
- Integration tests with mocked `snotelr`
- Data context integration tests
- Station mapping validation tests

**Run tests:**
```bash
pytest tests/test_snotel_integration.py -v
```

**Manual Testing:**

Test script: `scripts/test_snotel_integration.py` (legacy - use pytest instead)

**Performance Optimization:**

The caching system significantly reduces API calls:
- **Without cache**: 1 download per location/date query
- **With cache**: 1 download per unique station (reused for all dates/locations)
- **Example**: 100 locations near SHELL CREEK station → 1 download instead of 100
- **Expected savings**: ~99% reduction in downloads for clustered data

**Monitoring Cache Effectiveness:**

The `analyze_integrated_features.py` script reports SNOTEL data quality:
- Station usage statistics
- Cache effectiveness (unique stations vs. total rows)
- Distance statistics

```bash
python scripts/analyze_integrated_features.py data/processed/combined_north_bighorn_presence_absence_test.csv
```

---

## Part 2: Weather Data Integration ⚠️ PLACEHOLDER

### Status: ⚠️ Not Yet Implemented

Weather data is currently using placeholder values. The `WeatherClient` in `src/data/processors.py` returns default values.

### Step 2.1: Choose Weather Data Source

**Options:**
1. **PRISM** (recommended for historical): Gridded, high-resolution, free, daily data
2. **NOAA CDO** (Climate Data Online): Station-based, requires API key
3. **Open-Meteo** (simple API): Free, no key, but limited historical

**We'll use PRISM for historical** (best quality) and keep forecast as placeholder for now (can add NOAA later).

### Step 2.2: PRISM Data Access

PRISM provides data via:
- **Direct download**: http://prism.oregonstate.edu/recent/
- **API**: No official API, but files are accessible via URLs
- **R package**: `prism` package (but we'll use Python)

**For Wyoming, we'll download PRISM data files and create a lookup system.**

### Step 2.3: Install Required Packages

No additional packages needed for basic PRISM access (just `requests` for downloads).

However, PRISM files are in BIL format - you may want `rasterio` for reading (already in your environment).

### Step 2.4: Create PRISM Data Downloader

**Create script:** `scripts/download_prism_weather.py`

```python
"""
Download PRISM weather data for Wyoming (temperature and precipitation).
Note: PRISM files are large. This script downloads specific dates.
"""
import requests
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import shutil

def download_prism_date(date: datetime, element: str = "tmean", output_dir: Path = None):
    """
    Download PRISM data for a specific date.
    
    Args:
        date: Date to download
        element: "tmean" (temperature), "ppt" (precipitation), "tmin", "tmax"
        output_dir: Where to save files
    """
    if output_dir is None:
        output_dir = Path("data/raw/prism")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PRISM URL pattern
    date_str = date.strftime("%Y%m%d")
    base_url = "http://services.nacse.org/prism/data/public/4km"
    
    # Filename pattern
    filename = f"PRISM_{element}_stable_4kmD2_{date_str}_bil.zip"
    url = f"{base_url}/{element}/{date.strftime('%Y')}/{filename}"
    
    output_path = output_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        print(f"Already exists: {filename}")
        return output_path
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return output_path
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Not available: {filename} (may be too recent)")
        else:
            print(f"Error downloading {filename}: {e}")
        return None

if __name__ == "__main__":
    # Example: Download last 30 days of temperature data
    end_date = datetime.now() - timedelta(days=1)  # Yesterday (PRISM has 1-day delay)
    start_date = end_date - timedelta(days=30)
    
    current_date = start_date
    while current_date <= end_date:
        download_prism_date(current_date, element="tmean")
        download_prism_date(current_date, element="ppt")
        current_date += timedelta(days=1)
```

**Note:** PRISM files are ~50MB each. For production, consider:
- Downloading on-demand and caching
- Using a cloud storage solution
- Using PRISM's "lite" datasets (coarser resolution but smaller)

### Step 2.5: Create PRISM Client (Simplified Approach)

For a simpler approach, we'll use **Open-Meteo Historical Weather API** (free, no key) for historical data, and keep forecasts as placeholder.

**Alternative: Use Open-Meteo (easier than PRISM)**

Create updated `WeatherClient`:

```python
class WeatherClient:
    """Client for weather data using Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.cache = {}
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather for location and date"""
        
        # Check cache
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check if date is in future (forecast) or past (historical)
        today = datetime.now().date()
        target_date = date.date()
        
        if target_date > today:
            result = self._get_forecast(lat, lon, date)
        else:
            result = self._get_historical(lat, lon, date)
        
        self.cache[cache_key] = result
        return result
    
    def _get_historical(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical weather from Open-Meteo"""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date": date.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "America/Denver",
                "temperature_unit": "fahrenheit"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            
            temp_high = temps[0] if temps else 50.0
            temp_low = temp_min[0] if temp_min else 30.0
            temp = (temp_high + temp_low) / 2.0
            precip_day = precip[0] if precip else 0.0
            
            # Get 7-day precipitation (approximate - would need to query range)
            precip_7d = precip_day * 7  # Rough estimate
            
            return {
                "temp": float(temp),
                "temp_high": float(temp_high),
                "temp_low": float(temp_low),
                "precip_7d": float(precip_7d),
                "cloud_cover": 30,  # Not available from Open-Meteo free tier
                "wind_mph": 10  # Not available from Open-Meteo free tier
            }
        except Exception as e:
            logger.warning(f"Open-Meteo API error: {e}, using defaults")
            return self._get_default_weather()
    
    def _get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast (placeholder - can integrate NOAA later)"""
        # For now, use reasonable defaults
        # TODO: Integrate NOAA forecast API
        return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Default weather values"""
        return {
            "temp": 45.0,
            "temp_high": 55.0,
            "temp_low": 35.0,
            "precip_7d": 0.3,
            "cloud_cover": 30,
            "wind_mph": 10
        }
```

### Step 2.6: Test Weather Integration

Create test: `scripts/test_weather_integration.py`

```python
"""Test weather data integration"""
from datetime import datetime
from src.data.processors import WeatherClient

client = WeatherClient()

test_location = (41.8350, -106.4250)  # Wyoming
test_date = datetime(2024, 1, 15)

print(f"Testing weather integration")
print(f"Date: {test_date.strftime('%Y-%m-%d')}")
print()

weather = client.get_weather(test_location[0], test_location[1], test_date)

print("Results:")
print(f"  Temperature: {weather['temp']:.1f}°F")
print(f"  High: {weather['temp_high']:.1f}°F")
print(f"  Low: {weather['temp_low']:.1f}°F")
print(f"  Precip (7d): {weather['precip_7d']:.2f} inches")
```

---

## Part 3: NDVI/Satellite Data Integration ⚠️ PLACEHOLDER

### Status: ⚠️ Not Yet Implemented

NDVI data is currently using placeholder values with seasonal variation. The `SatelliteClient` in `src/data/processors.py` returns default values.

### Step 3.1: Choose NDVI Data Source

**Options:**
1. **Google Earth Engine** (recommended): Free, easy API, Landsat/Sentinel-2
2. **MODIS via AppEEARS**: Free, pre-computed NDVI, 250m resolution
3. **USGS EarthExplorer**: Free but requires manual download

**We'll use Google Earth Engine** for the best balance of ease and quality.

### Step 3.2: Set Up Google Earth Engine

1. **Sign up for Google Earth Engine:**
   - Go to: https://earthengine.google.com/
   - Sign in with Google account
   - Click "Sign up" (requires approval, usually instant for research)

2. **Install Earth Engine Python API:**
   ```bash
   pip install earthengine-api
   ```

3. **Authenticate:**
   ```bash
   earthengine authenticate
   ```
   This will open a browser for authentication.

4. **Initialize (one time):**
   ```bash
   python -c "import ee; ee.Initialize()"
   ```

### Step 3.3: Create Earth Engine NDVI Client

Update `SatelliteClient` in `src/data/processors.py`:

```python
class SatelliteClient:
    """Client for satellite imagery (NDVI) using Google Earth Engine"""
    
    def __init__(self):
        self.cache = {}
        self._ee_initialized = False
        self._init_earth_engine()
    
    def _init_earth_engine(self):
        """Initialize Google Earth Engine"""
        try:
            import ee
            if not self._ee_initialized:
                ee.Initialize()
                self._ee_initialized = True
                logger.info("Google Earth Engine initialized")
            self.ee = ee
        except Exception as e:
            logger.warning(f"Earth Engine not available: {e}")
            logger.info("Install with: pip install earthengine-api")
            logger.info("Authenticate with: earthengine authenticate")
            self.ee = None
            self._ee_initialized = False
    
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get NDVI for location and date using Landsat 8/9"""
        
        if self.ee is None:
            return self._get_placeholder_ndvi(date)
        
        # Check cache
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            import ee
            
            # Create point
            point = ee.Geometry.Point([lon, lat])
            
            # Use Landsat 8 or 9 (Collection 2, Tier 1)
            # Search for images within ±8 days of target date
            start_date = (date - timedelta(days=8)).strftime("%Y-%m-%d")
            end_date = (date + timedelta(days=8)).strftime("%Y-%m-%d")
            
            # Try Landsat 9 first, then Landsat 8
            collection = (
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterDate(start_date, end_date)
                .filterBounds(point)
                .filter(ee.Filter.lt("CLOUD_COVER", 20))  # <20% cloud
                .sort("CLOUD_COVER")
            )
            
            # If no Landsat 9, try Landsat 8
            if collection.size().getInfo() == 0:
                collection = (
                    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                    .filterDate(start_date, end_date)
                    .filterBounds(point)
                    .filter(ee.Filter.lt("CLOUD_COVER", 20))
                    .sort("CLOUD_COVER")
                )
            
            # Get first (least cloudy) image
            image_count = collection.size().getInfo()
            if image_count == 0:
                logger.warning(f"No cloud-free Landsat images found for {date}")
                return self._get_placeholder_ndvi(date)
            
            image = collection.first()
            
            # Calculate NDVI
            # Landsat 8/9: NDVI = (NIR - Red) / (NIR + Red)
            # Bands: SR_B5 (NIR), SR_B4 (Red)
            nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)  # Scale factor
            red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            
            # Sample at point
            ndvi_value = ndvi.sample(point, scale=30).first().get("NDVI").getInfo()
            
            # Get image date
            image_date = ee.Date(image.get("system:time_start")).getInfo()
            image_datetime = datetime.fromtimestamp(image_date["value"] / 1000)
            age_days = abs((image_datetime.date() - date.date()).days)
            
            # Calculate IRG (Instantaneous Rate of Green-up)
            # Get NDVI from 16 days ago for comparison
            irg = self._calculate_irg(lat, lon, date)
            
            result = {
                "ndvi": float(ndvi_value) if ndvi_value is not None else 0.5,
                "age_days": age_days,
                "irg": irg,
                "cloud_free": True
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Earth Engine error: {e}, using placeholder")
            return self._get_placeholder_ndvi(date)
    
    def _calculate_irg(self, lat: float, lon: float, date: datetime) -> float:
        """Calculate Instantaneous Rate of Green-up (change in NDVI)"""
        if self.ee is None:
            return 0.0
        
        try:
            # Get NDVI 16 days ago and now
            date_prev = date - timedelta(days=16)
            ndvi_now = self.get_ndvi(lat, lon, date)
            ndvi_prev = self.get_ndvi(lat, lon, date_prev)
            
            irg = (ndvi_now["ndvi"] - ndvi_prev["ndvi"]) / 16.0
            return irg
        except Exception:
            return 0.0
    
    def get_integrated_ndvi(self, lat: float, lon: float,
                           start_date: datetime, end_date: datetime) -> float:
        """Get integrated NDVI over date range (sum of daily NDVI)"""
        
        if self.ee is None:
            return 60.0  # Placeholder
        
        try:
            import ee
            
            point = ee.Geometry.Point([lon, lat])
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Use MODIS NDVI for integrated calculation (faster, daily)
            collection = (
                ee.ImageCollection("MODIS/061/MOD13Q1")  # MODIS 16-day NDVI
                .filterDate(start_str, end_str)
                .filterBounds(point)
                .select("NDVI")
            )
            
            # Calculate mean NDVI over period, then multiply by days
            mean_ndvi = collection.mean()
            ndvi_value = mean_ndvi.sample(point, scale=250).first().get("NDVI").getInfo()
            
            if ndvi_value is None:
                return 60.0
            
            # Scale MODIS NDVI (0-10000) to 0-1, then integrate
            ndvi_normalized = ndvi_value / 10000.0
            days = (end_date - start_date).days
            integrated = ndvi_normalized * days * 100  # Scale for typical range
            
            return float(integrated)
            
        except Exception as e:
            logger.warning(f"Earth Engine integrated NDVI error: {e}")
            return 60.0
    
    def _get_placeholder_ndvi(self, date: datetime) -> Dict:
        """Placeholder NDVI with seasonal variation"""
        month = date.month
        
        if month in [6, 7, 8]:  # Summer
            ndvi = 0.70
        elif month in [9, 10]:  # Fall
            ndvi = 0.55
        elif month in [11, 12, 1, 2, 3]:  # Winter
            ndvi = 0.30
        else:  # Spring
            ndvi = 0.50
        
        return {
            "ndvi": ndvi,
            "age_days": 8,
            "irg": 0.01 if month in [4, 5] else -0.005 if month in [9, 10] else 0.0,
            "cloud_free": True
        }
```

### Step 3.4: Update environment.yml

Add `earthengine-api` to your dependencies:

```yaml
  # Packages only available via pip
  - pip:
    - boto3              # AWS SDK
    - earthengine-api    # Google Earth Engine Python API
```

### Step 3.5: Test NDVI Integration

Create test: `scripts/test_ndvi_integration.py`

```python
"""Test NDVI data integration"""
from datetime import datetime
from src.data.processors import SatelliteClient

client = SatelliteClient()

test_location = (41.8350, -106.4250)
test_date = datetime(2024, 7, 15)  # Summer (should have high NDVI)

print(f"Testing NDVI integration")
print(f"Date: {test_date.strftime('%Y-%m-%d')}")
print()

ndvi_data = client.get_ndvi(test_location[0], test_location[1], test_date)

print("Results:")
print(f"  NDVI: {ndvi_data['ndvi']:.3f}")
print(f"  IRG: {ndvi_data['irg']:.4f}")
print(f"  Age: {ndvi_data['age_days']} days")
print(f"  Cloud-free: {ndvi_data['cloud_free']}")

# Test integrated NDVI
start_date = datetime(2024, 6, 1)
end_date = datetime(2024, 9, 1)
integrated = client.get_integrated_ndvi(
    test_location[0],
    test_location[1],
    start_date,
    end_date
)
print(f"\nSummer integrated NDVI: {integrated:.1f}")
```

---

## Testing All Integrations Together

Create comprehensive test: `scripts/test_temporal_data_integration.py`

```python
"""Test all temporal data integrations"""
from datetime import datetime
from pathlib import Path
from src.data.processors import DataContextBuilder

data_dir = Path("data")
builder = DataContextBuilder(data_dir)

test_location = {"lat": 41.8350, "lon": -106.4250}
test_date = "2024-01-15"

print("Testing complete temporal data integration")
print(f"Location: {test_location}")
print(f"Date: {test_date}")
print("=" * 60)

context = builder.build_context(test_location, test_date)

print("\nSNOTEL/Snow Data:")
print(f"  Snow depth: {context.get('snow_depth_inches', 0):.1f} inches")
print(f"  SWE: {context.get('snow_water_equiv_inches', 0):.1f} inches")

print("\nWeather Data:")
print(f"  Temperature: {context.get('temperature_f', 0):.1f}°F")
print(f"  Precip (7d): {context.get('precip_last_7_days_inches', 0):.2f} inches")

print("\nNDVI Data:")
print(f"  NDVI: {context.get('ndvi', 0):.3f}")
print(f"  IRG: {context.get('irg', 0):.4f}")
print(f"  Age: {context.get('ndvi_age_days', 0)} days")
```

---

## Summary Checklist

- [x] **SNOTEL Integration:** ✅ **COMPLETE**
  - [x] Station database: `data/cache/snotel_stations_wyoming.geojson` (36 stations)
  - [x] Station ID mapping: `scripts/map_snotel_station_ids.py` (31/36 mapped, 86%)
  - [x] `SNOTELClient` implemented with `snotelr` via `rpy2`
  - [x] Two-level caching optimization (station + request level)
  - [x] Elevation-based fallback using actual DEM elevation
  - [x] Data quality tracking fields
  - [x] Comprehensive unit and integration tests
  - [x] Integrated into `DataContextBuilder`

- [ ] **Weather Integration:** ⚠️ **PLACEHOLDER**
  - [ ] Choose data source (Open-Meteo recommended)
  - [ ] Update `WeatherClient` in `src/data/processors.py`
  - [ ] Implement caching (similar to SNOTEL)
  - [ ] Test with `scripts/test_weather_integration.py`

- [ ] **NDVI Integration:** ⚠️ **PLACEHOLDER**
  - [ ] Sign up for Google Earth Engine
  - [ ] Run `pip install earthengine-api`
  - [ ] Run `earthengine authenticate`
  - [ ] Update `SatelliteClient` in `src/data/processors.py`
  - [ ] Update `environment.yml` to include earthengine-api
  - [ ] Implement caching for NDVI queries
  - [ ] Test with `scripts/test_ndvi_integration.py`

- [ ] **Final Integration Test:**
  - [ ] Run `scripts/test_temporal_data_integration.py`
  - [ ] Verify all data sources work together
  - [ ] Test full pipeline with real weather and NDVI data

---

## Next Steps

### Completed for SNOTEL:
1. ✅ Two-level in-memory caching (station + request level)
2. ✅ Error handling and fallback mechanisms
3. ✅ Logging and data quality tracking
4. ✅ Comprehensive unit and integration tests
5. ✅ DEM-based elevation fallback (not hardcoded)

### Future Enhancements for SNOTEL:
1. File-based persistent caching (survive across script runs)
2. Batch processing for multiple dates
3. Additional station mapping strategies to improve coverage beyond 86%

### For Weather and NDVI Integration:
1. Follow SNOTEL pattern: implement two-level caching
2. Add error handling and retry logic
3. Implement fallback mechanisms for missing data
4. Add data quality tracking fields (similar to `snow_data_source`)
5. Create comprehensive tests following `tests/test_snotel_integration.py` pattern
6. Update `integrate_environmental_features.py` to include new fields
7. Update `analyze_integrated_features.py` to validate new data sources

