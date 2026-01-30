# PathWild Data Acquisition Strategy

## Overview

This document details the data acquisition and caching strategy for all environmental data sources used in the PathWild pipeline. The strategy balances performance (via caching) with data freshness (via live API calls for recent/forecast data).

## Data Sources Summary

| Data Source | Provider | Historical Cache | Recent Data | Forecast Data | Cache Type |
|------------|----------|------------------|-------------|---------------|------------|
| NDVI | Google Earth Engine | ✅ Persistent (SQLite) | ✅ Cached | N/A | SQLite DB |
| SNOTEL Snow | AWDB API | ✅ Persistent (SQLite, >30 days) | ❌ Live API (<30 days) | N/A | SQLite DB |
| Weather Historical | PRISM | ✅ File-based | ❌ Open-Meteo fallback | N/A | TIF files |
| Weather Forecast | Open-Meteo | N/A | N/A | ✅ Live API | In-memory only |
| Satellite NDVI | AppEEARS | ❌ No cache | ❌ Live API | N/A | In-memory only |
| Static Layers | Local files | ✅ Permanent | N/A | N/A | GeoJSON/TIF files |

---

## 1. NDVI Data (Google Earth Engine)

### Source
- **Primary**: Google Earth Engine (GEE) via `GEENDVIClient`
- **Fallback**: AppEEARS via `SatelliteClient` (if GEE unavailable)

### Caching Strategy
- **Cache Type**: Persistent SQLite database (`data/cache/ndvi_cache.db`)
- **Cache Key**: Location (lat/lon rounded to 5 decimal places), date, collection, buffer_days, max_cloud_cover
- **Historical Data**: ✅ **Cached permanently** - All historical NDVI data is cached and never expires
- **Recent Data**: ✅ **Cached** - No distinction between historical and recent data; all retrieved data is cached
- **Forecast Data**: N/A - NDVI is not forecasted

### Behavior
- **First Run**: Queries GEE API, stores results in SQLite cache
- **Subsequent Runs**: Checks cache first, only queries GEE if cache miss
- **Cache Persistence**: Permanent (historical satellite imagery never changes)
- **Cache Statistics**: Tracks total entries and database size

### Configuration
- Default: `use_cache=True`
- Cache directory: `data/cache/` (configurable via `cache_dir` parameter)
- Cache is thread-safe and supports concurrent access

---

## 2. SNOTEL Snow Data (AWDB API)

### Source
- **Primary**: AWDB REST API (USDA Natural Resources Conservation Service)
- **Fallback**: Elevation-based estimation (if no station nearby or API fails)

### Caching Strategy
- **Cache Type**: 
  - **Station Locations**: GeoJSON file (`data/cache/snotel_stations_wyoming.geojson`) - permanent
  - **Station Data**: Persistent SQLite database (`data/cache/snotel_data_cache.db`)
- **Cache Key**: Station ID, begin_date, end_date (date range)
- **Historical Data (>30 days old)**: ✅ **Cached permanently** - Data older than 30 days is cached and never expires
- **Recent Data (≤30 days old)**: ❌ **Live API** - Data within 30 days bypasses cache to ensure freshness for inference
- **Forecast Data**: N/A - Snow data is not forecasted

### Behavior
- **Station Locations**: 
  - First run: Fetches from AWDB API, saves to GeoJSON
  - Subsequent runs: Loads from cached GeoJSON (permanent, locations don't change)
  - Retry logic: 3 attempts with exponential backoff for connection errors
  
- **Station Data**:
  - Historical (>30 days): Checks SQLite cache first, queries API only on cache miss
  - Recent (≤30 days): Always queries live API (bypasses cache)
  - Cache persistence: Historical entries are permanent
  - In-memory cache: Also maintains in-memory cache for faster access during same run

### Configuration
- Default: `use_cache=True`, `cache_historical_only=True`
- Cache directory: `data/cache/` (configurable)
- Rate limiting: 100ms minimum between API requests (10 req/sec max)
- Retry logic: 3 attempts with exponential backoff for connection errors and 5xx responses

---

## 3. Weather Data (PRISM + Open-Meteo)

### Source
- **Historical**: PRISM (Parameter-elevation Regressions on Independent Slopes Model)
- **Recent/Forecast**: Open-Meteo API

### Caching Strategy
- **PRISM Historical Data**:
  - **Cache Type**: File-based (TIF raster files in `data/prism/{variable}/`)
  - **Cache Key**: Variable (tmean/tmin/tmax/ppt), date
  - **Historical Data**: ✅ **Cached permanently** - Files are downloaded once and reused
  - **Cache Check**: Checks if file exists before downloading
  - **File Locking**: Thread-safe file locking prevents duplicate downloads
  
- **Open-Meteo Data**:
  - **Cache Type**: In-memory only (per `WeatherClient` instance)
  - **Recent Data**: ❌ **Live API** - Always queries Open-Meteo for recent dates not in PRISM
  - **Forecast Data**: ❌ **Live API** - Always queries Open-Meteo for future dates

### Behavior
- **Historical (PRISM available)**:
  - First run: Downloads PRISM TIF files, stores in `data/prism/`
  - Subsequent runs: Checks if file exists, uses cached file if present
  - File format: COG (Cloud-Optimized GeoTIFF) delivered in ZIP, extracted to TIF
  
- **Recent (PRISM unavailable)**:
  - Falls back to Open-Meteo historical API
  - No persistent caching (in-memory only)
  
- **Forecast (Future dates)**:
  - Always uses Open-Meteo forecast API
  - No persistent caching (in-memory only)

### Configuration
- PRISM data directory: `data/prism/` (configurable)
- PRISM files are organized by variable: `tmean/`, `tmin/`, `tmax/`, `ppt/`
- Open-Meteo: No configuration needed (free public API)

---

## 4. Satellite NDVI (AppEEARS - Fallback)

### Source
- **Provider**: NASA AppEEARS (Application for Extracting and Exploring Analysis Ready Samples)
- **Usage**: Fallback when GEE is unavailable or disabled

### Caching Strategy
- **Cache Type**: In-memory only (per `SatelliteClient` instance)
- **Historical Data**: ❌ **No persistent cache** - Always queries AppEEARS API
- **Recent Data**: ❌ **Live API** - Always queries AppEEARS API
- **Forecast Data**: N/A - NDVI is not forecasted

### Behavior
- **All Requests**: Always queries AppEEARS API (no persistent caching)
- **Batch Processing**: Supports batch requests for efficiency
- **Timeout Handling**: 5 minutes for single requests, 15 minutes for batch requests
- **Fallback**: Uses placeholder seasonal values if API fails

### Configuration
- Requires: `APPEEARS_USERNAME` and `APPEEARS_PASSWORD` environment variables
- Note: AppEEARS uses async API (requires minutes to process), not suitable for real-time inference

---

## 5. Static Environmental Layers

### Sources
- **DEM (Elevation)**: Local TIF file (`data/dem/wyoming_dem.tif`)
- **Slope/Aspect**: Local TIF files (`data/terrain/`)
- **Land Cover**: Local TIF file (`data/landcover/nlcd.tif`)
- **Canopy Cover**: Local TIF file (`data/canopy/canopy_cover.tif`)
- **Water Sources**: Local GeoJSON (`data/hydrology/water_sources.geojson`)
- **Roads/Trails**: Local GeoJSON files (`data/infrastructure/`)
- **Wildlife Data**: Local GeoJSON files (`data/wildlife/`)

### Caching Strategy
- **Cache Type**: File-based (permanent local files)
- **All Data**: ✅ **Permanent** - Loaded once at initialization, no expiration
- **Updates**: Manual (files must be replaced/updated manually)

### Behavior
- **Initialization**: All static layers loaded once when `DataContextBuilder` is created
- **Runtime**: No API calls, all data accessed from local files
- **Memory**: Raster files kept open for efficient sampling

---

## Summary of Caching Behavior

### Historical Data (Training Data)
| Data Source | Cached? | Cache Type | Expiration |
|------------|---------|------------|------------|
| NDVI (GEE) | ✅ Yes | SQLite | Never |
| SNOTEL (>30 days) | ✅ Yes | SQLite | Never |
| Weather (PRISM) | ✅ Yes | TIF files | Never |
| Satellite NDVI | ❌ No | N/A | N/A |

### Recent Data (Inference - Last 30 Days)
| Data Source | Cached? | Source |
|------------|---------|--------|
| NDVI (GEE) | ✅ Yes | SQLite cache (no distinction) |
| SNOTEL (≤30 days) | ❌ No | Live AWDB API |
| Weather (recent) | ❌ No | Live Open-Meteo API |
| Satellite NDVI | ❌ No | Live AppEEARS API |

### Forecast Data (Future Dates)
| Data Source | Cached? | Source |
|------------|---------|--------|
| Weather | ❌ No | Live Open-Meteo API |
| NDVI | N/A | Not forecasted |
| SNOTEL | N/A | Not forecasted |

---

## Cache Locations

All persistent caches are stored in the `data/cache/` directory:

```
data/cache/
├── ndvi_cache.db              # GEE NDVI data (SQLite)
├── snotel_data_cache.db       # SNOTEL station data (SQLite)
└── snotel_stations_wyoming.geojson  # SNOTEL station locations (GeoJSON)
```

PRISM weather data is stored separately:

```
data/prism/
├── tmean/                     # Mean temperature TIF files
├── tmin/                      # Min temperature TIF files
├── tmax/                      # Max temperature TIF files
└── ppt/                       # Precipitation TIF files
```

---

## Performance Implications

### First Pipeline Run
- **Slow**: All historical data must be downloaded from APIs
- **NDVI**: GEE API calls (can be slow for large batches)
- **SNOTEL**: AWDB API calls (rate-limited to 10 req/sec)
- **Weather**: PRISM file downloads (can be large files)

### Subsequent Pipeline Runs
- **Fast**: Historical data served from cache
- **NDVI**: Cache hits are instant (SQLite lookup)
- **SNOTEL**: Historical data from cache, only recent data queries API
- **Weather**: PRISM files loaded from disk (no download)

### Inference (Recent/Forecast Data)
- **NDVI**: Still uses cache (GEE caches all data)
- **SNOTEL**: Live API calls for data ≤30 days old
- **Weather**: Live API calls for recent/forecast data

---

## Recommendations

1. **For Training Data**: 
   - Run pipeline once to populate caches
   - Subsequent runs will be much faster
   - Caches persist between runs

2. **For Inference**:
   - Recent SNOTEL data (≤30 days) always uses live API
   - Weather forecasts always use live API
   - Consider implementing short-term caching (e.g., 1-hour TTL) for inference workloads

3. **Cache Management**:
   - SQLite caches grow over time but remain efficient
   - PRISM files are large but only downloaded once per date
   - Consider cache cleanup scripts if disk space is limited

4. **Error Handling**:
   - All clients have retry logic for transient errors
   - Fallback mechanisms in place (e.g., elevation-based snow estimates)
   - Connection errors are retried with exponential backoff

---

## Notes

- **Thread Safety**: All SQLite caches use locks for thread-safe concurrent access
- **Rate Limiting**: SNOTEL client enforces 100ms minimum between requests
- **Data Freshness**: Recent data (≤30 days) always uses live APIs to ensure accuracy for inference
- **Historical Accuracy**: Historical data is cached permanently since it never changes


