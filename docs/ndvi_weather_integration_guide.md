# NDVI/Satellite and Weather/Temperature Data Integration Guide

This guide provides detailed step-by-step instructions for sourcing and integrating real NDVI/satellite data and weather/temperature data into PathWild.

**Last Updated**: 2026-01-04

---

## Table of Contents

1. [Overview](#overview)
2. [NDVI/Satellite Data Integration](#ndvi-satellite-data-integration)
   - [Option 1: Google Earth Engine (Recommended)](#option-1-google-earth-engine-recommended)
   - [Option 2: MODIS via AppEEARS (Free, Easy)](#option-2-modis-via-appeears-free-easy)
   - [Option 3: PRISM NDVI (Free, Gridded)](#option-3-prism-ndvi-free-gridded)
3. [Weather/Temperature Data Integration](#weather-temperature-data-integration)
   - [Option 1: PRISM Climate Data (Recommended)](#option-1-prism-climate-data-recommended)
   - [Option 2: Open-Meteo (Free, Easy)](#option-2-open-meteo-free-easy)
   - [Option 3: NOAA CDO (Free, Station-Based)](#option-3-noaa-cdo-free-station-based)
4. [Single Provider vs. Multiple Providers: Strategic Analysis](#single-provider-vs-multiple-providers-strategic-analysis)
5. [Training vs. Inference Data Requirements: Validation and Analysis](#training-vs-inference-data-requirements-validation-and-analysis)
6. [Implementation Steps](#implementation-steps)
7. [Testing and Validation](#testing-and-validation)
8. [Performance Optimization](#performance-optimization)

---

## Overview

### Current Status

- ⚠️ **NDVI/Satellite**: Placeholder implementation in `SatelliteClient.get_ndvi()` and `get_integrated_ndvi()`
- ⚠️ **Weather/Temperature**: Placeholder implementation in `WeatherClient._get_historical()` and `_get_forecast()`

### Data Requirements

**NDVI/Satellite Data Needed:**
- `ndvi`: Normalized Difference Vegetation Index (0-1 range)
- `irg`: Instantaneous Rate of Green-up (rate of change)
- `ndvi_age_days`: Data recency (for confidence scoring)
- `cloud_free`: Cloud cover flag
- `summer_integrated_ndvi`: Integrated NDVI for June-September (for nutritional condition)

**Weather/Temperature Data Needed:**
- `temp`: Average daily temperature (°F)
- `temp_high`: Daily high temperature (°F)
- `temp_low`: Daily low temperature (°F)
- `precip_7d`: 7-day cumulative precipitation (inches)
- Historical data: Daily temperatures for Nov 1 - Apr 30 (for WSI calculation)

### Integration Points

- **NDVI**: Used by `VegetationQualityHeuristic` and `NutritionalConditionHeuristic`
- **Weather/Temperature**: Used by `WinterSeverityHeuristic` and indirectly by other heuristics

---

## NDVI/Satellite Data Integration

### Option 1: Google Earth Engine (Recommended)

**Best For**: High resolution (30m Landsat), flexible processing, cloud-free composites

**Detailed Pros:**
- **Free with Google account** - No per-request charges, generous free tier
- **High-resolution data** - Landsat 8/9 (30m resolution) and Sentinel-2 (10m resolution)
- **Built-in cloud masking** - QA bands provided for automatic cloud detection
- **Python API** - Well-maintained `earthengine-api` package with good documentation
- **Flexible processing** - Can compute NDVI on-the-fly or use pre-computed indices
- **Multiple satellite sources** - Access to Landsat, Sentinel-2, MODIS in one platform
- **Time series analysis** - Easy to build time series for IRG calculations
- **Large catalog** - Access to historical data back to 1984 (Landsat 5)
- **Active community** - Extensive documentation, tutorials, and Stack Overflow support
- **Cloud-based processing** - Processing happens on Google's servers (no local storage needed)

**Detailed Cons:**
- **Requires Google account** - Must create and authenticate Google account (see setup instructions below)
- **Account approval delay** - Initial signup may take 1-2 days for approval
- **API speed limitations** - Can be slow for individual point queries (better for batch processing)
- **Rate limits** - Free tier has usage limits (but generous for most use cases)
- **Learning curve** - Earth Engine API has unique syntax (requires learning curve)
- **Network dependency** - Requires internet connection for all operations
- **No offline access** - All processing happens in cloud, cannot download raw imagery easily
- **Computation quotas** - Daily computation quotas for complex operations
- **Authentication complexity** - Service account setup needed for production deployments

**Cost**: Free (but requires Google account)

**Data Latency**: Near real-time (Landsat 8/9: ~1-2 days after acquisition, Sentinel-2: ~1 day)

**Best Use Case**: When you need high-resolution NDVI with cloud masking and have time to set up Google account authentication

#### Step 1.1: Install Google Earth Engine API

```bash
# Install the Python package
pip install earthengine-api

# Or add to requirements.txt:
# earthengine-api>=0.1.375
```

#### Step 1.2: Set Up Authentication

**Option A: Create a Dedicated Google Account for PathWild (Recommended for Production)**

Creating a dedicated account keeps your personal Google account separate and makes it easier to manage credentials, especially for production deployments.

**Steps to Create a Dedicated Google Account:**

1. **Create a new Google account:**
   - Go to https://accounts.google.com/signup
   - Use a professional email address format, e.g., `pathwild.project@gmail.com` or `pathwild.ai@gmail.com`
   - Fill in the required information
   - Verify your email address and phone number (if required)

2. **Sign up for Google Earth Engine:**
   - Visit https://earthengine.google.com/
   - Sign in with your new dedicated account
   - Click "Sign up" to request access
   - Fill out the registration form:
     - **Institution/Organization**: List your organization or "Independent Researcher/Developer"
     - **Purpose**: Select "Research" or "Education"
     - **Country**: Select your country
     - **Accept terms**: Read and accept the Terms of Service
   - Submit the request
   - **Wait for approval** (usually approved within 1-2 business days, can take up to 1 week)

3. **Verify approval:**
   - Check your email for approval notification
   - Visit https://code.earthengine.google.com/ to verify access
   - You should see the Earth Engine Code Editor interface

4. **Authenticate your local environment:**
   ```bash
   earthengine authenticate
   ```
   This will:
   - Open a browser window
   - Prompt you to sign in (use your dedicated account)
   - Request authorization to access Earth Engine
   - Save credentials to `~/.config/earthengine/credentials`

5. **Verify authentication:**
   ```python
   import ee
   ee.Initialize()
   print("Earth Engine initialized successfully!")
   ```

**Option B: Use Existing Google Account (Quick Start)**

If you want to test quickly before setting up a dedicated account:

1. Sign in to https://earthengine.google.com/ with your existing Google account
2. Click "Sign up" and request access
3. Wait for approval
4. Run `earthengine authenticate` and sign in when prompted

**Note**: For production deployments, using a dedicated account is strongly recommended for:
- Better credential management
- Separation of personal and project accounts
- Easier team collaboration (can share account if needed)
- Service account setup for automated deployments

**Service Account Setup (For Production/Automated Deployments):**

For production deployments (e.g., AWS Lambda, scheduled scripts), you'll need a service account:

1. Go to https://console.cloud.google.com/ (sign in with your dedicated account)
2. Create a new project (or select existing): "PathWild" or "pathwild-project"
3. Enable Earth Engine API:
   - Go to "APIs & Services" > "Library"
   - Search for "Earth Engine API"
   - Click "Enable"
4. Create service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Name: "pathwild-earthengine"
   - Grant role: "Earth Engine User" (if available) or basic "Editor"
5. Create key:
   - Click on the service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download and save securely (e.g., `pathwild-ee-service-account.json`)
6. Register service account with Earth Engine:
   - Email Google Earth Engine support with the service account email
   - Request Earth Engine access for the service account
7. Use in code:
   ```python
   import ee
   credentials = ee.ServiceAccountCredentials(
       email='your-service-account@project-id.iam.gserviceaccount.com',
       key_file='path/to/pathwild-ee-service-account.json'
   )
   ee.Initialize(credentials)
   ```

**Security Best Practices:**
- Never commit service account JSON files to Git (add to `.gitignore`)
- Store service account keys securely (use environment variables or AWS Secrets Manager)
- Rotate keys periodically
- Use least-privilege access (only grant necessary permissions)

#### Step 1.3: Implement NDVI Retrieval

**Recommended Dataset**: Landsat 8/9 Collection 2 Level-2 (Surface Reflectance)

**Key Features:**
- 30m resolution
- 16-day revisit cycle
- Pre-calculated cloud mask
- Surface reflectance (not raw TOA)

**Implementation Strategy:**

1. **Get latest cloud-free image** within date window (e.g., ±16 days)
2. **Calculate NDVI** from red and near-infrared bands
3. **Calculate IRG** from time series (rate of change)
4. **Cache results** to avoid repeated API calls

**Code Example:**

```python
import ee
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np

class SatelliteClient:
    """Client for satellite imagery (NDVI, etc.)"""
    
    def __init__(self):
        self.cache = {}
        ee.Initialize()  # Initialize Google Earth Engine
    
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get NDVI for location and date using Google Earth Engine"""
        
        # Check cache first
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Date window: ±16 days (Landsat revisit cycle)
        start_date = (date - timedelta(days=16)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=16)).strftime('%Y-%m-%d')
        
        # Use Landsat 9 if available, fallback to Landsat 8
        collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                     .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
                     .filterDate(start_date, end_date)
                     .filterBounds(point)
                     .map(self._mask_clouds)
                     .map(self._add_ndvi)
                     .sort('system:time_start', False))  # Most recent first
        
        # Get the most recent cloud-free image
        try:
            image = ee.Image(collection.first())
            
            # Sample at point
            sample = image.select('NDVI').sample(
                region=point,
                scale=30,
                numPixels=1
            )
            
            # Get the value
            ndvi_value = sample.first().get('NDVI').getInfo()
            
            if ndvi_value is None:
                # Fallback to MODIS or use interpolation
                return self._fallback_ndvi(lat, lon, date)
            
            # Calculate IRG (rate of change) - requires time series
            irg = self._calculate_irg(lat, lon, date)
            
            # Calculate data age (days since image acquisition)
            image_date = image.get('system:time_start').getInfo()
            image_datetime = datetime.fromtimestamp(image_date / 1000)
            age_days = (date - image_datetime.date()).days
            
            result = {
                "ndvi": ndvi_value / 10000.0,  # Scale factor (Landsat scales NDVI by 10000)
                "age_days": abs(age_days),
                "irg": irg,
                "cloud_free": True
            }
            
            # Cache result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get NDVI from Earth Engine: {e}")
            return self._fallback_ndvi(lat, lon, date)
    
    def _mask_clouds(self, image):
        """Mask clouds using Landsat QA band"""
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud bit
        dilated_cloud_mask = qa.bitwiseAnd(1 << 1).eq(0)  # Dilated cloud bit
        return image.updateMask(cloud_mask).updateMask(dilated_cloud_mask)
    
    def _add_ndvi(self, image):
        """Calculate NDVI from Landsat bands"""
        # Surface reflectance bands
        nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)  # NIR band
        red = image.select('SR_B4').multiply(0.0000275).add(-0.2)  # Red band
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = nir.subtract(red).divide(nir.add(red)).multiply(10000)
        return image.addBands(ndvi.rename('NDVI'))
    
    def _calculate_irg(self, lat: float, lon: float, date: datetime) -> float:
        """Calculate Instantaneous Rate of Green-up from time series"""
        # Get NDVI for current date and 16 days ago
        current_ndvi_data = self.get_ndvi(lat, lon, date)
        previous_date = date - timedelta(days=16)
        previous_ndvi_data = self.get_ndvi(lat, lon, previous_date)
        
        current_ndvi = current_ndvi_data.get('ndvi', 0.5)
        previous_ndvi = previous_ndvi_data.get('ndvi', 0.5)
        
        # IRG = (NDVI_current - NDVI_previous) / days
        irg = (current_ndvi - previous_ndvi) / 16.0
        return irg
    
    def get_integrated_ndvi(self, lat: float, lon: float, 
                           start_date: datetime, end_date: datetime) -> float:
        """Get integrated NDVI over date range (summer iNDVI)"""
        # Sample every 16 days (Landsat revisit)
        current_date = start_date
        ndvi_values = []
        
        while current_date <= end_date:
            ndvi_data = self.get_ndvi(lat, lon, current_date)
            if ndvi_data.get('ndvi') is not None:
                ndvi_values.append(ndvi_data['ndvi'])
            current_date += timedelta(days=16)
        
        if not ndvi_values:
            return 60.0  # Fallback default
        
        # Integrated NDVI = sum of NDVI values
        return sum(ndvi_values)
    
    def _fallback_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fallback NDVI (could use MODIS or seasonal average)"""
        # Placeholder fallback
        month = date.month
        if month in [6, 7, 8]:
            ndvi = 0.70
        elif month in [9, 10]:
            ndvi = 0.55
        elif month in [11, 12, 1, 2, 3]:
            ndvi = 0.30
        else:
            ndvi = 0.50
        
        return {
            "ndvi": ndvi,
            "age_days": 16,
            "irg": 0.0,
            "cloud_free": False
        }
```

#### Step 1.4: Add to Requirements

Add to `requirements.txt`:
```
earthengine-api>=0.1.375
```

Or to `environment.yml` under pip:
```yaml
  - pip
  - pip:
    - earthengine-api>=0.1.375
```

---

### Option 2: MODIS via AppEEARS (Free, Easy)

**Best For**: Quick setup, daily composites, automated downloads, regional analysis

**Detailed Pros:**
- **Free, no authentication for downloads** - Simple web interface, no API keys needed
- **Pre-calculated NDVI** - MOD13Q1 provides NDVI ready to use (no calculation needed)
- **Daily 16-day composites** - MOD13Q1 updated every 16 days with best-pixel composites
- **Long historical record** - Data available from 2000 to present
- **Batch processing** - Can download large areas/time periods efficiently
- **Multiple output formats** - GeoTIFF, NetCDF, CSV time series
- **Point extraction** - Can extract time series for specific points directly
- **Quality assurance** - MODIS includes QA bands for data quality assessment
- **NASA support** - Well-maintained by NASA, reliable data source
- **No coding required** - Web interface available for simple downloads

**Detailed Cons:**
- **Lower spatial resolution** - 250m pixels (vs 30m Landsat) may miss fine-scale vegetation patterns
- **Data latency** - ~2-3 weeks behind real-time (composite processing takes time)
- **Coarser temporal resolution** - 16-day composites (vs daily Landsat availability, though cloud-free daily Landsat is rare)
- **API complexity** - Programmatic access requires understanding AppEEARS API
- **File management** - Need to download and manage large raster files locally
- **Storage requirements** - Daily Wyoming coverage = ~100-200 MB per date (adds up over time)
- **No real-time data** - Cannot get same-day or next-day NDVI
- **Composite artifacts** - 16-day composites may blend phenological changes
- **Limited customization** - Cannot adjust cloud masking or composite algorithms

**Cost**: Free (NASA Earthdata account required, but free)

**Data Latency**: ~2-3 weeks (16-day composites plus processing time)

**Best Use Case**: When you need consistent historical NDVI data and can work with lower resolution

#### Step 2.1: Set Up AppEEARS Account

1. Visit https://appeears.earthdatacloud.nasa.gov/
2. Create a free account (NASA Earthdata account)
3. Request access to MODIS data (automatic for registered users)

#### Step 2.2: Download MODIS NDVI Data

**Manual Approach:**
1. Use AppEEARS web interface to download MOD13Q1 (16-day NDVI)
2. Cover Wyoming bounding box
3. Extract time series for specific points using Python

**API Approach (Recommended):**
Use `requests` library to interact with AppEEARS API:

```python
import requests
from datetime import datetime
import numpy as np
import pandas as pd

class MODISNDVIClient:
    """Client for MODIS NDVI via AppEEARS API"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.api_base = "https://appeears.earthdatacloud.nasa.gov/api"
        self.token = self._authenticate()
        self.cache = {}
    
    def _authenticate(self) -> str:
        """Authenticate with AppEEARS API"""
        response = requests.post(
            f"{self.api_base}/login",
            auth=(self.username, self.password)
        )
        response.raise_for_status()
        return response.json()['token']
    
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get MODIS NDVI for location and date"""
        # MODIS data is organized in 16-day composites
        # Find the composite period containing the date
        composite_date = self._get_composite_date(date)
        
        # Use pre-downloaded MODIS data or API
        # Implementation depends on whether you pre-download or use API
        # ...
```

**Recommendation**: For production use, pre-download MODIS data for Wyoming and create a local cache/raster stack.

---

### Option 3: PRISM NDVI (Free, Gridded)

**Best For**: Historical data, consistency with PRISM climate data, regional-scale analysis

**Detailed Pros:**
- **Free, no authentication** - Direct download, no API keys or accounts needed
- **Consistent data source** - Same provider as PRISM temperature/precipitation (good for data consistency)
- **Long historical record** - Available from 1981 to present (longer than MODIS)
- **Daily data** - Daily NDVI values (though derived from longer-period composites)
- **Gridded format** - Easy to work with if you're already using PRISM rasters
- **Simple data structure** - Standard BIL/BIP/FLT raster format, well-documented
- **Spatial consistency** - Same 4km grid as PRISM climate data (perfect alignment)
- **No API complexity** - Simple file download, no API learning curve

**Detailed Cons:**
- **Very coarse resolution** - 4km pixels (vs 250m MODIS, 30m Landsat) - not suitable for point locations
- **Modeled/satellite-blended** - Not pure satellite observations (includes interpolation/modeling)
- **Limited documentation** - Less documentation and examples compared to MODIS/Landsat
- **Less accurate for points** - 4km resolution means single pixel may cover large area with mixed vegetation
- **Not commonly used** - Fewer examples and community support compared to MODIS/Landsat
- **Unknown methodology** - Less transparent about how NDVI is derived/blended
- **File management** - Need to download and manage many raster files
- **No near-real-time** - Same latency as other PRISM products (1-2 days)

**Cost**: Free

**Data Latency**: ~1-2 days (similar to PRISM temperature data)

**Best Use Case**: Only consider if you're already using PRISM for weather data and need consistent data source (not recommended for point-level NDVI due to resolution)

**Note**: PRISM NDVI is less commonly used than MODIS/Landsat. The 4km resolution makes it unsuitable for point-level predictions. Consider this only if you're already using PRISM for weather data and need data source consistency.

---

## Weather/Temperature Data Integration

### Option 1: PRISM Climate Data (Recommended)

**Best For**: Historical data, gridded data, consistency, research applications

**Detailed Pros:**
- **Free, no authentication** - Direct download, no API keys or accounts required
- **High-quality data** - Interpolated from weather stations using sophisticated models (considers elevation, topography)
- **Long historical record** - Daily data from 1981 to present (40+ years)
- **Consistent spatial coverage** - Complete coverage of CONUS (no gaps)
- **Multiple variables** - Temperature (min/max/mean), precipitation, vapor pressure, etc.
- **Gridded format** - 4km resolution, easy to work with raster tools
- **Elevation-aware** - PRISM interpolation accounts for elevation effects (important for Wyoming's mountains)
- **Well-documented** - Extensive documentation and metadata
- **Research-grade** - Widely used in scientific research, peer-reviewed methodology
- **Stable format** - Consistent file naming and structure over time

**Detailed Cons:**
- **Coarse resolution** - 4km pixels (fine for regional analysis, coarse for point locations)
- **Raster file processing** - Requires rasterio/GDAL to extract point values (more complex than API)
- **Data latency** - ~1-2 days behind real-time (not suitable for same-day forecasts)
- **Storage requirements** - Daily files are ~50-100 MB each (adds up over time)
- **File management** - Need to download, organize, and manage many raster files
- **No forecast data** - Only historical data (need separate source for forecasts)
- **Point interpolation** - Single pixel value may not represent exact point location well
- **Download complexity** - Need to handle downloading/processing many files programmatically
- **No API** - Must download files (though recent data available via HTTP)

**Cost**: Free

**Data Latency**: ~1-2 days (data released 1-2 days after observation date)

**Best Use Case**: Historical temperature data for WSI calculations, research applications, when you need long-term consistent data

#### Step 1.1: Understand PRISM Data Structure

PRISM provides daily gridded data:
- **Temperature**: Daily min/max/mean (°C, convert to °F)
- **Precipitation**: Daily total (mm, convert to inches)
- **Format**: BIL/BIP/FLT raster files
- **Coverage**: CONUS (Continental US)

#### Step 1.2: Set Up PRISM Data Access

**Option A: Direct Download (Recommended for Historical Data)**

1. **Download daily temperature data for Wyoming:**
   - Visit https://prism.oregonstate.edu/recent/
   - Download daily temperature files (BIL format)
   - Extract and organize by date

2. **Create download script:**
   ```python
   # scripts/download_prism_data.py
   import requests
   from pathlib import Path
   from datetime import datetime, timedelta
   import zipfile
   
   PRISM_BASE_URL = "https://services.nacse.org/prism/data/public/4km/daily"
   
   def download_prism_daily(variable: str, date: datetime, output_dir: Path):
       """Download PRISM daily data file"""
       # PRISM uses YYYYMMDD format
       date_str = date.strftime('%Y%m%d')
       url = f"{PRISM_BASE_URL}/{variable}/{date_str}"
       
       # Download (may need to handle authentication for some endpoints)
       response = requests.get(url)
       # Save and extract...
   ```

**Option B: PRISM API (For Recent Data)**

PRISM provides a REST API for recent data (last 30 days typically):
- Documentation: https://prism.oregonstate.edu/projects/api.php
- Endpoint: `https://services.nacse.org/prism/data/public/4km/daily/{variable}/{date}`

#### Step 1.3: Implement PRISM Client

```python
import rasterio
from rasterio.mask import mask
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import requests
import zipfile
import tempfile

class PRISMClient:
    """Client for PRISM climate data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/weather/prism")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
    
    def get_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get daily temperature data for location and date"""
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check if file exists locally
        tmax_file = self._get_prism_file_path('tmax', date)
        tmin_file = self._get_prism_file_path('tmin', date)
        
        # Download if not exists
        if not tmax_file.exists():
            self._download_prism_daily('tmax', date)
        if not tmin_file.exists():
            self._download_prism_daily('tmin', date)
        
        # Sample at point
        from shapely.geometry import Point
        point = Point(lon, lat)
        
        try:
            with rasterio.open(tmax_file) as src:
                tmax_data, _ = mask(src, [point], crop=True)
                tmax = float(tmax_data[0, 0, 0])
            
            with rasterio.open(tmin_file) as src:
                tmin_data, _ = mask(src, [point], crop=True)
                tmin = float(tmin_data[0, 0, 0])
            
            # Convert from °C to °F
            tmax_f = (tmax * 9/5) + 32
            tmin_f = (tmin * 9/5) + 32
            temp_avg_f = (tmax_f + tmin_f) / 2
            
            result = {
                "temp": temp_avg_f,
                "temp_high": tmax_f,
                "temp_low": tmin_f
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get PRISM temperature: {e}")
            return self._fallback_temperature(lat, lon, date)
    
    def get_precipitation(self, lat: float, lon: float, 
                         start_date: datetime, end_date: datetime) -> float:
        """Get cumulative precipitation over date range (inches)"""
        total_precip = 0.0
        current_date = start_date
        
        while current_date <= end_date:
            ppt_file = self._get_prism_file_path('ppt', current_date)
            if not ppt_file.exists():
                self._download_prism_daily('ppt', current_date)
            
            from shapely.geometry import Point
            point = Point(lon, lat)
            
            try:
                with rasterio.open(ppt_file) as src:
                    ppt_data, _ = mask(src, [point], crop=True)
                    ppt_mm = float(ppt_data[0, 0, 0])
                    total_precip += ppt_mm / 25.4  # Convert mm to inches
            except Exception as e:
                logger.warning(f"Failed to get PRISM precipitation for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return total_precip
    
    def _get_prism_file_path(self, variable: str, date: datetime) -> Path:
        """Get expected file path for PRISM data"""
        date_str = date.strftime('%Y%m%d')
        filename = f"PRISM_{variable}_stable_4kmD2_{date_str}_bil.bil"
        return self.data_dir / variable / filename
    
    def _download_prism_daily(self, variable: str, date: datetime):
        """Download PRISM daily data file"""
        # Implementation depends on PRISM API structure
        # This is a placeholder - actual implementation will vary
        pass
    
    def _fallback_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fallback temperature (seasonal average)"""
        month = date.month
        if month in [12, 1, 2]:  # Winter
            temp = 20.0
            temp_high = 30.0
            temp_low = 10.0
        elif month in [3, 4, 5]:  # Spring
            temp = 45.0
            temp_high = 55.0
            temp_low = 35.0
        elif month in [6, 7, 8]:  # Summer
            temp = 65.0
            temp_high = 75.0
            temp_low = 55.0
        else:  # Fall
            temp = 50.0
            temp_high = 60.0
            temp_low = 40.0
        
        return {
            "temp": temp,
            "temp_high": temp_high,
            "temp_low": temp_low
        }
```

#### Step 1.4: Update WeatherClient

```python
class WeatherClient:
    """Client for weather data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.prism_client = PRISMClient(data_dir)
        self.cache = {}
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather for location and date"""
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        today = datetime.now().date()
        target_date = date.date()
        
        if target_date > today:
            return self._get_forecast(lat, lon, date)
        else:
            return self._get_historical(lat, lon, date)
    
    def _get_historical(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical weather from PRISM"""
        temp_data = self.prism_client.get_temperature(lat, lon, date)
        
        # Get 7-day precipitation
        start_date = date - timedelta(days=7)
        precip_7d = self.prism_client.get_precipitation(lat, lon, start_date, date)
        
        return {
            "temp": temp_data["temp"],
            "temp_high": temp_data["temp_high"],
            "temp_low": temp_data["temp_low"],
            "precip_7d": precip_7d,
            "cloud_cover": 40,  # PRISM doesn't provide cloud cover
            "wind_mph": 12  # PRISM doesn't provide wind
        }
    
    def _get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast (use Open-Meteo or similar)"""
        # For forecasts, use Open-Meteo (see Option 2)
        # Placeholder for now
        return {
            "temp": 45.0,
            "temp_high": 55.0,
            "temp_low": 35.0,
            "precip_7d": 0.3,
            "cloud_cover": 30,
            "wind_mph": 10
        }
```

---

### Option 2: Open-Meteo (Free, Easy)

**Best For**: Quick setup, forecasts, recent historical data, point locations, rapid prototyping

**Detailed Pros:**
- **Free, no authentication** - No API keys, accounts, or registration required
- **Simple REST API** - Easy to use, well-documented, returns JSON
- **Historical data** - Reanalysis data from 1940 to present (ERA5-based)
- **Forecast data** - Up to 16 days ahead (great for prediction use cases)
- **Point-based queries** - Get values directly for lat/lon (no raster processing needed)
- **Fast implementation** - Can be implemented in hours (vs days for PRISM)
- **Multiple models** - Choice of reanalysis models (ERA5, ERA5-Land, etc.)
- **Good documentation** - Clear API documentation with examples
- **Real-time data** - Recent data available quickly (good latency)
- **Multiple variables** - Temperature, precipitation, wind, humidity, etc.
- **Timezone support** - Can request data in specific timezones
- **No file management** - All data via API (no downloading/managing files)

**Detailed Cons:**
- **Modeled/reanalysis data** - Not direct observations (blend of models and observations)
- **Lower spatial resolution** - ~11km resolution (vs 4km PRISM) - coarser interpolation
- **Rate limits** - Free tier: 10,000 requests/day (may need caching for large batches)
- **Less accurate than PRISM** - PRISM's elevation-aware interpolation is more accurate for complex terrain
- **API dependency** - Requires internet connection, API availability
- **No control over methodology** - Cannot adjust interpolation methods
- **Potential data gaps** - API outages could interrupt service
- **Less research-grade** - Less commonly used in scientific research than PRISM
- **Forecast uncertainty** - Forecast accuracy decreases with time (especially beyond 7 days)

**Cost**: Free (10,000 requests/day), paid tiers available for higher limits

**Data Latency**: Near real-time for recent data, forecasts updated multiple times per day

**Best Use Case**: Quick prototyping, applications needing forecasts, point-based queries, when you want to avoid file management

#### Step 2.1: Understand Open-Meteo API

**Base URL**: `https://archive-api.open-meteo.com/v1/archive` (historical)
**Base URL**: `https://api.open-meteo.com/v1/forecast` (forecast)

**Parameters:**
- `latitude`, `longitude`: Location
- `start_date`, `end_date`: Date range
- `daily`: Variables (temperature_2m_max, temperature_2m_min, precipitation_sum, etc.)
- `temperature_unit`: fahrenheit
- `precipitation_unit`: inch

#### Step 2.2: Implement Open-Meteo Client

```python
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

class OpenMeteoClient:
    """Client for Open-Meteo weather data"""
    
    def __init__(self):
        self.historical_base = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_base = "https://api.open-meteo.com/v1/forecast"
        self.cache = {}
    
    def get_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get daily temperature data"""
        cache_key = f"{lat:.4f},{lon:.4f},{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        today = datetime.now().date()
        target_date = date.date()
        
        if target_date > today:
            # Forecast
            return self._get_forecast_temperature(lat, lon, date)
        else:
            # Historical
            return self._get_historical_temperature(lat, lon, date)
    
    def _get_historical_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical temperature from Open-Meteo"""
        date_str = date.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': date_str,
            'end_date': date_str,
            'daily': 'temperature_2m_max,temperature_2m_min',
            'temperature_unit': 'fahrenheit',
            'timezone': 'America/Denver'
        }
        
        try:
            response = requests.get(self.historical_base, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            if daily.get('temperature_2m_max') and daily.get('temperature_2m_min'):
                temp_high = daily['temperature_2m_max'][0]
                temp_low = daily['temperature_2m_min'][0]
                temp_avg = (temp_high + temp_low) / 2
                
                result = {
                    "temp": temp_avg,
                    "temp_high": temp_high,
                    "temp_low": temp_low
                }
                
                self.cache[f"{lat:.4f},{lon:.4f},{date_str}"] = result
                return result
            else:
                raise ValueError("No temperature data in response")
                
        except Exception as e:
            logger.warning(f"Failed to get Open-Meteo temperature: {e}")
            return self._fallback_temperature(lat, lon, date)
    
    def _get_forecast_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get forecast temperature from Open-Meteo"""
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'temperature_2m_max,temperature_2m_min',
            'temperature_unit': 'fahrenheit',
            'timezone': 'America/Denver',
            'forecast_days': 16
        }
        
        try:
            response = requests.get(self.forecast_base, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            dates = daily.get('time', [])
            temps_max = daily.get('temperature_2m_max', [])
            temps_min = daily.get('temperature_2m_min', [])
            
            # Find the date in forecast
            target_date_str = date.strftime('%Y-%m-%d')
            if target_date_str in dates:
                idx = dates.index(target_date_str)
                temp_high = temps_max[idx]
                temp_low = temps_min[idx]
                temp_avg = (temp_high + temp_low) / 2
                
                return {
                    "temp": temp_avg,
                    "temp_high": temp_high,
                    "temp_low": temp_low
                }
            else:
                raise ValueError(f"Date {target_date_str} not in forecast")
                
        except Exception as e:
            logger.warning(f"Failed to get Open-Meteo forecast: {e}")
            return self._fallback_temperature(lat, lon, date)
    
    def get_precipitation(self, lat: float, lon: float, 
                         start_date: datetime, end_date: datetime) -> float:
        """Get cumulative precipitation over date range (inches)"""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_str,
            'end_date': end_str,
            'daily': 'precipitation_sum',
            'precipitation_unit': 'inch',
            'timezone': 'America/Denver'
        }
        
        try:
            response = requests.get(self.historical_base, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get('daily', {})
            precip_sums = daily.get('precipitation_sum', [])
            return sum(precip_sums) if precip_sums else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get Open-Meteo precipitation: {e}")
            return 0.0
    
    def _fallback_temperature(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fallback temperature"""
        month = date.month
        if month in [12, 1, 2]:
            return {"temp": 20.0, "temp_high": 30.0, "temp_low": 10.0}
        elif month in [3, 4, 5]:
            return {"temp": 45.0, "temp_high": 55.0, "temp_low": 35.0}
        elif month in [6, 7, 8]:
            return {"temp": 65.0, "temp_high": 75.0, "temp_low": 55.0}
        else:
            return {"temp": 50.0, "temp_high": 60.0, "temp_low": 40.0}
```

#### Step 2.3: Update WeatherClient to Use Open-Meteo

```python
class WeatherClient:
    """Client for weather data"""
    
    def __init__(self):
        self.open_meteo = OpenMeteoClient()
        self.cache = {}
    
    def _get_historical(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get historical weather from Open-Meteo"""
        temp_data = self.open_meteo.get_temperature(lat, lon, date)
        
        # Get 7-day precipitation
        start_date = date - timedelta(days=7)
        precip_7d = self.open_meteo.get_precipitation(lat, lon, start_date, date)
        
        return {
            "temp": temp_data["temp"],
            "temp_high": temp_data["temp_high"],
            "temp_low": temp_data["temp_low"],
            "precip_7d": precip_7d,
            "cloud_cover": 40,  # Open-Meteo doesn't provide cloud cover in free tier
            "wind_mph": 12  # Open-Meteo doesn't provide wind in free tier
        }
    
    def _get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast from Open-Meteo"""
        temp_data = self.open_meteo.get_temperature(lat, lon, date)
        
        # Forecast precipitation (7-day window)
        start_date = date - timedelta(days=7)
        precip_7d = self.open_meteo.get_precipitation(lat, lon, start_date, date)
        
        return {
            "temp": temp_data["temp"],
            "temp_high": temp_data["temp_high"],
            "temp_low": temp_data["temp_low"],
            "precip_7d": precip_7d,
            "cloud_cover": 30,
            "wind_mph": 10
        }
```

---

### Option 3: NOAA CDO (Free, Station-Based)

**Best For**: Official weather station data, research applications, validation of other sources

**Detailed Pros:**
- **Official NOAA data** - Gold standard, authoritative source
- **Direct observations** - Real station measurements (not modeled/interpolated)
- **Very long historical records** - Some stations have 100+ years of data
- **Free** - No cost, public data
- **High temporal resolution** - Hourly data available (vs daily for PRISM/Open-Meteo)
- **Quality controlled** - Data goes through quality assurance processes
- **Metadata rich** - Station locations, elevations, instrument types, etc.
- **Research standard** - Widely used in climate research

**Detailed Cons:**
- **Station-based** - Point observations, requires interpolation for locations between stations
- **Sparse coverage** - Stations may be far apart (especially in remote Wyoming areas)
- **Complex API** - Multiple endpoints, complex parameter structures
- **Station selection complexity** - Need to find nearest station(s), handle missing data
- **Data format complexity** - Various formats (CSV, JSON), need to parse correctly
- **Missing data** - Stations may have gaps in record
- **Elevation differences** - Nearest station may be at different elevation (important in mountains)
- **API rate limits** - NOAA API has request limits
- **Token/authentication** - Requires API token (free but requires registration)
- **Interpolation needed** - Must implement spatial interpolation (inverse distance, kriging, etc.)
- **More development time** - Significantly more complex to implement than PRISM/Open-Meteo

**Cost**: Free (but requires API token registration)

**Data Latency**: Varies by station, typically 1-2 days

**Best Use Case**: Research applications requiring station observations, validation of gridded products, when you need hourly data

**Recommendation**: Use PRISM or Open-Meteo unless you specifically need station observations or are doing research requiring official NOAA data.

---

## Single Provider vs. Multiple Providers: Strategic Analysis

### Should You Use One Provider for Both NDVI and Weather?

**Short Answer**: **Generally no** - Use Google Earth Engine for NDVI and Open-Meteo or PRISM for weather/temperature. However, using PRISM for both has some advantages if you can accept lower NDVI resolution.

### Option 1: Different Providers (Recommended)

**Combination**: Google Earth Engine (NDVI) + Open-Meteo (Weather) OR Google Earth Engine (NDVI) + PRISM (Weather)

**Advantages:**
- **Best-of-breed approach** - Use the best provider for each data type
- **Higher NDVI resolution** - Google Earth Engine provides 30m Landsat data (vs 4km PRISM NDVI)
- **Better NDVI accuracy** - Landsat/MODIS are standard for vegetation monitoring
- **Flexibility** - Can switch weather providers without affecting NDVI
- **Optimized for each use case** - Each provider optimized for their specialty

**Disadvantages:**
- **Two authentication systems** - Google account for Earth Engine, potentially API keys for weather
- **Different data formats** - Earth Engine API vs REST API (more code complexity)
- **Potential data inconsistency** - Slightly different spatial/temporal resolutions
- **More dependencies** - Need to maintain two data pipelines

**When to Use**: Recommended for most applications, especially when NDVI resolution matters (point-level predictions)

### Option 2: Single Provider - PRISM for Both

**Combination**: PRISM NDVI + PRISM Temperature/Precipitation

**Advantages:**
- **Single data source** - One provider, one authentication method, one data format
- **Spatial alignment** - Same 4km grid for all variables (perfect spatial consistency)
- **Temporal consistency** - Same processing methodology and temporal resolution
- **Simpler codebase** - One client class, one caching strategy, one error handling pattern
- **Easier maintenance** - Fewer dependencies, simpler deployment
- **Data consistency** - All data from same source reduces systematic biases

**Disadvantages:**
- **Coarse NDVI resolution** - 4km pixels are too coarse for point-level predictions
- **Less accurate NDVI** - PRISM NDVI is less commonly used/validated than MODIS/Landsat
- **Limited NDVI documentation** - Less community support and examples
- **Not suitable for point locations** - 4km resolution means single pixel covers large area
- **Unknown NDVI methodology** - Less transparent about NDVI derivation

**When to Use**: Only if you're doing regional-scale analysis (not point-level), need perfect spatial alignment, and NDVI resolution is not critical

### Option 3: Single Provider - Google Earth Engine for Both

**Combination**: Google Earth Engine NDVI + Google Earth Engine Weather (ERA5, GFS, etc.)

**Advantages:**
- **Single platform** - All data from Earth Engine
- **High-resolution NDVI** - Best available (30m Landsat)
- **Weather reanalysis** - ERA5 reanalysis data available in Earth Engine
- **Unified API** - Same Python API for all data access
- **Cloud processing** - All processing on Google's servers

**Disadvantages:**
- **Weather data limitations** - Earth Engine weather data is less commonly used than PRISM/Open-Meteo
- **Learning curve** - Earth Engine API syntax is unique
- **Computation quotas** - Weather data processing counts toward computation limits
- **Less weather documentation** - Fewer examples for weather data in Earth Engine
- **Google account dependency** - Single point of failure (account issues affect all data)

**When to Use**: If you're already heavily invested in Earth Engine and want unified platform

### Recommendation Matrix

| Use Case | NDVI Provider | Weather Provider | Rationale |
|----------|---------------|------------------|-----------|
| **Point-level predictions (RECOMMENDED)** | Google Earth Engine | Open-Meteo | Best NDVI resolution, easiest weather setup |
| **Historical research, regional analysis** | Google Earth Engine | PRISM | High-res NDVI, research-grade weather |
| **Quick prototyping** | MODIS/AppEEARS | Open-Meteo | Fast setup, adequate resolution |
| **Regional analysis only** | PRISM | PRISM | Single provider, spatial alignment (if 4km OK) |
| **Production, high volume** | Google Earth Engine | PRISM (pre-downloaded) | Best quality, cached weather files |

### Decision Factors

1. **NDVI Resolution Requirements**:
   - Point-level predictions → Need 30m (Google Earth Engine) or 250m (MODIS)
   - Regional analysis → 4km (PRISM) may be acceptable

2. **Development Speed**:
   - Fast setup → Open-Meteo for weather (hours)
   - Best quality → PRISM for weather (days for setup)

3. **Data Consistency Needs**:
   - Perfect alignment → PRISM for both (same grid)
   - Best quality → Different providers (optimize each)

4. **Production Requirements**:
   - High volume → Pre-download PRISM weather files
   - Real-time → Open-Meteo API
   - Forecasts needed → Open-Meteo (PRISM doesn't have forecasts)

**Final Recommendation for PathWild**: Use **Google Earth Engine for NDVI** (30m resolution critical for point predictions) and **Open-Meteo for weather/temperature** (quick setup, forecasts, point queries). This gives you the best NDVI resolution while keeping weather implementation simple. Consider PRISM for weather if you need research-grade historical data and can invest in file management infrastructure.

---

## Training vs. Inference Data Requirements: Validation and Analysis

### Your Assumptions (VALIDATED ✅)

You correctly identified that PathWild has two distinct data requirements:

1. **Training Phase**: Historical NDVI and weather/temperature data for dates when GPS collar data was collected
2. **Inference Phase**: Current/recent NDVI and forecasted weather/temperature for future predictions

Let's validate these assumptions and assess how they affect provider selection.

### Training Phase Requirements

**Data Needs:**
- Historical NDVI data for dates matching your GPS collar observations (e.g., 2018-2023)
- Historical weather/temperature data for the same dates
- Data should align temporally with when elk were actually observed (presence) or not observed (absence)

**Key Considerations:**
- **Data quality matters** - Model learns from this data, so accuracy is important
- **Temporal coverage** - Need data covering all dates in your training dataset
- **Spatial coverage** - Need data for all locations in Wyoming where you have GPS collar data
- **Consistency** - Data should be consistent across all training dates/locations

### Inference Phase Requirements

**Data Needs:**
- **Current/recent NDVI** - For "today" or near-real-time vegetation conditions
- **Forecasted weather/temperature** - For future dates (e.g., "tomorrow", "next week")
- Data should be available on-demand when users make predictions

**Key Considerations:**
- **Forecasts required** - Must support future dates (weather forecasts)
- **Low latency** - Users expect fast responses (API-based better than file downloads)
- **Real-time availability** - Need current/recent data, not just historical
- **Point queries** - Individual location queries, not batch processing

### Provider Assessment: Training vs. Inference

#### NDVI Data Assessment

| Provider | Training (Historical) | Inference (Current/Forecast) | Verdict |
|----------|----------------------|------------------------------|---------|
| **Google Earth Engine** | ✅ Excellent - Historical Landsat data back to 1984 | ✅ Good - Recent data with 1-2 day latency | **WORKS FOR BOTH** |
| **MODIS/AppEEARS** | ✅ Good - Historical data from 2000 | ⚠️ Limited - 2-3 week latency, no forecasts | **TRAINING ONLY** |
| **PRISM NDVI** | ✅ Available - Historical from 1981 | ⚠️ Limited - 1-2 day latency, no forecasts | **TRAINING ONLY** |

**Analysis**: Google Earth Engine is the only provider that works well for **both** training and inference. MODIS and PRISM NDVI have too much latency for real-time inference use cases.

#### Weather/Temperature Data Assessment

| Provider | Training (Historical) | Inference (Forecasts) | Verdict |
|----------|----------------------|----------------------|---------|
| **PRISM** | ✅ Excellent - Research-grade, elevation-aware, 1981-present | ❌ No forecasts - Historical only | **TRAINING ONLY** |
| **Open-Meteo** | ✅ Good - Historical ERA5 reanalysis (1940-present) | ✅ Excellent - Forecasts up to 16 days | **WORKS FOR BOTH** |
| **NOAA CDO** | ✅ Good - Station observations, very long records | ⚠️ Limited - Forecasts require separate NWS API | **COMPLEX SETUP** |

**Analysis**: 
- **Open-Meteo** works for both training and inference (historical + forecasts)
- **PRISM** is excellent for training but cannot provide forecasts (historical only)
- **NOAA CDO** requires additional setup for forecasts

### Revised Recommendation: Single vs. Hybrid Approach

Given your training vs. inference requirements, here are the strategic options:

#### Option A: Single Provider for Both Phases (Simplest)

**Combination**: Google Earth Engine (NDVI) + Open-Meteo (Weather)

**Training:**
- ✅ Google Earth Engine: Historical Landsat NDVI (back to 1984)
- ✅ Open-Meteo: Historical weather (ERA5 reanalysis, 1940-present)

**Inference:**
- ✅ Google Earth Engine: Recent NDVI (1-2 day latency acceptable)
- ✅ Open-Meteo: Weather forecasts (up to 16 days ahead)

**Pros:**
- Single implementation for each data type
- Same code works for both training and inference
- Simple to maintain
- Open-Meteo historical data quality is good (ERA5-based)

**Cons:**
- Open-Meteo historical data is slightly less accurate than PRISM for complex terrain (but still good)
- PRISM would be better for training if forecasts weren't needed

**Verdict**: **RECOMMENDED** - Simplest approach that works well for both phases

#### Option B: Hybrid Approach (Best Quality, More Complex)

**Combination**: Google Earth Engine (NDVI) + PRISM (Training) + Open-Meteo (Inference)

**Training:**
- ✅ Google Earth Engine: Historical Landsat NDVI
- ✅ PRISM: Historical weather (research-grade, elevation-aware)

**Inference:**
- ✅ Google Earth Engine: Recent NDVI
- ✅ Open-Meteo: Weather forecasts

**Pros:**
- Best possible data quality for training (PRISM is more accurate)
- Best for inference (Open-Meteo has forecasts)
- Optimized for each use case

**Cons:**
- More complex implementation (two weather clients)
- Need to switch providers between training/inference
- More code to maintain
- Requires handling both PRISM files and Open-Meteo API

**Verdict**: **OPTIONAL** - Consider if data quality is critical and you can handle added complexity

#### Option C: PRISM for Everything (Not Recommended)

**Combination**: PRISM NDVI + PRISM Weather

**Training:**
- ⚠️ PRISM NDVI: Historical data available but 4km resolution too coarse
- ✅ PRISM Weather: Excellent historical data

**Inference:**
- ❌ PRISM NDVI: 1-2 day latency, no real-time capability
- ❌ PRISM Weather: No forecasts available

**Verdict**: **NOT RECOMMENDED** - Cannot support inference phase requirements (no forecasts, coarse NDVI)

### Final Recommendation Based on Training vs. Inference Analysis

**Recommended Approach**: **Option A - Single Provider (Google Earth Engine + Open-Meteo)**

**Rationale:**

1. **Works for Both Phases**:
   - Google Earth Engine provides historical NDVI for training AND recent NDVI for inference
   - Open-Meteo provides historical weather for training AND forecasts for inference

2. **Simplest Implementation**:
   - Single code path for each data type
   - Same `WeatherClient` and `SatelliteClient` classes work for both training and inference
   - Easier to maintain and debug

3. **Good Data Quality**:
   - Google Earth Engine NDVI: 30m resolution, industry standard
   - Open-Meteo historical data: ERA5 reanalysis (widely used, good quality)
   - Open-Meteo forecasts: Up to 16 days, updated frequently

4. **Production-Ready**:
   - API-based (no file management for Open-Meteo)
   - Fast response times for inference
   - Suitable for AWS Lambda/serverless deployment

**When to Consider Option B (Hybrid Approach):**

- If you find Open-Meteo historical data quality insufficient after validation
- If you're doing research requiring PRISM's elevation-aware interpolation
- If you have resources to maintain dual weather clients
- If training data quality is more critical than implementation simplicity

**Implementation Note:**

Your current `WeatherClient` design already supports this! It checks if the date is in the future or past:

```python
def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
    today = datetime.now().date()
    target_date = date.date()
    
    if target_date > today:
        return self._get_forecast(lat, lon, date)  # Open-Meteo forecast
    else:
        return self._get_historical(lat, lon, date)  # Open-Meteo historical
```

This automatically handles both training (past dates → historical) and inference (future dates → forecast) with a single implementation!

### Summary

✅ **Your assumptions are correct**: Training needs historical data, inference needs current NDVI + weather forecasts.

✅ **Recommended approach remains the same**: Google Earth Engine (NDVI) + Open-Meteo (Weather)

✅ **This approach satisfies both requirements**:
- Training: Historical data available from both providers
- Inference: Recent NDVI + weather forecasts available from both providers

✅ **Consider hybrid approach (PRISM for training, Open-Meteo for inference) only if**:
- You need maximum training data quality (PRISM's elevation-aware interpolation)
- You can handle additional complexity of dual weather clients

---

## Implementation Steps

### Step 1: Choose Your Data Sources

**Recommended Combination:**
- **NDVI**: Google Earth Engine (Landsat 8/9) - Best quality and resolution (30m)
- **Weather/Temperature**: Open-Meteo (quick setup, forecasts) OR PRISM (better for historical data, research-grade)

### Step 2: Install Dependencies

Add to `requirements.txt`:
```
earthengine-api>=0.1.375
requests>=2.31.0
```

Or to `environment.yml`:
```yaml
  - pip
  - pip:
    - earthengine-api>=0.1.375
```

### Step 3: Set Up Authentication/Configuration

**For Google Earth Engine:**
```bash
earthengine authenticate
```

**For Open-Meteo:**
No authentication needed!

**For PRISM:**
No authentication needed, but organize download directory structure.

### Step 4: Update `src/data/processors.py`

Replace the placeholder implementations in:
- `SatelliteClient.get_ndvi()`
- `SatelliteClient.get_integrated_ndvi()`
- `WeatherClient._get_historical()`
- `WeatherClient._get_forecast()`

### Step 5: Update `WinterSeverityHeuristic._get_temperature()`

In `src/scoring/heuristics/winterkill.py`, update `_get_temperature()` to use the real `WeatherClient`:

```python
def _get_temperature(self, location, date, context):
    """Get temperature for location/date (°F)"""
    # Use WeatherClient from context if available
    weather_client = context.get('weather_client')
    if weather_client:
        weather_data = weather_client.get_weather(
            location['lat'], 
            location['lon'], 
            date
        )
        return weather_data.get('temp', 25.0)
    return 25.0  # Fallback
```

### Step 6: Add Caching Strategy

Both NDVI and weather data benefit from caching:
- Cache API responses to avoid repeated calls
- Cache processed data for frequently accessed locations/dates
- Consider disk caching for large historical datasets

### Step 7: Handle Errors Gracefully

- Implement fallback values when APIs fail
- Log warnings for debugging
- Return reasonable defaults for missing data

---

## Testing and Validation

### Step 1: Unit Tests

Create test files:
- `tests/test_satellite_client.py`
- `tests/test_weather_client.py`

**Test Cases:**
1. Valid location/date returns reasonable values
2. NDVI values are in valid range (0-1)
3. Temperature values are reasonable for Wyoming (-40°F to 100°F)
4. Caching works correctly
5. Error handling returns fallback values
6. IRG calculation is correct

### Step 2: Integration Tests

Test with real elk GPS collar data:
1. Run `scripts/integrate_environmental_features.py` with test data
2. Verify NDVI and weather fields are populated
3. Check data quality (no NaN values, reasonable ranges)

### Step 3: Validation

**NDVI Validation:**
- Compare with known seasonal patterns (summer high, winter low)
- Check that IRG is positive in spring, negative in fall
- Verify cloud-free flags are set correctly

**Weather Validation:**
- Compare with nearby SNOTEL station temperatures (if available)
- Check seasonal patterns (winter cold, summer warm)
- Verify precipitation totals are reasonable

---

## Performance Optimization

### Caching Strategies

1. **Memory Cache**: Cache recent requests in memory (LRU cache)
2. **Disk Cache**: Cache historical data to disk (especially PRISM raster files)
3. **Batch Processing**: Group requests when possible (e.g., batch Earth Engine requests)

### Optimization Tips

1. **NDVI**: 
   - Pre-compute cloud-free composites for common dates
   - Cache time series for IRG calculation
   - Use MODIS for large-area analyses (faster than Landsat)

2. **Weather**:
   - Pre-download PRISM data for your date range
   - Use Open-Meteo for point queries (faster than raster processing)
   - Cache frequently accessed dates

3. **Batch Processing**:
   - When processing many locations, batch API calls
   - Use parallel processing for independent requests

---

## Next Steps

1. **Choose data sources** (recommend Google Earth Engine + Open-Meteo to start)
2. **Set up authentication** (Google Earth Engine)
3. **Implement clients** (follow code examples above)
4. **Update processors.py** (replace placeholders)
5. **Test with sample data** (verify values are reasonable)
6. **Run integration tests** (verify with real elk data)
7. **Optimize performance** (add caching, batch processing)

---

## Additional Resources

- **Google Earth Engine**: https://earthengine.google.com/
- **Earth Engine Python API Docs**: https://developers.google.com/earth-engine/guides/python_install
- **Open-Meteo API Docs**: https://open-meteo.com/en/docs
- **PRISM Climate Data**: https://prism.oregonstate.edu/
- **MODIS AppEEARS**: https://appeears.earthdatacloud.nasa.gov/

---

## Questions or Issues?

- Check existing implementations in `src/data/processors.py`
- Review test files in `tests/` directory
- See `docs/dataset_gap_analysis.md` for overall data status

