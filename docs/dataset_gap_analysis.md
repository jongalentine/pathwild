# Dataset Gap Analysis for PathWild Scoring Heuristics

This document analyzes which datasets are required for each scoring heuristic and identifies gaps in the current data pipeline.

## Executive Summary

**Status**: ❌ **Not all datasets are available**

You have **6 of 10** required datasets fully integrated. Four datasets are **missing**, and three data sources are using **placeholder implementations** that need real API integration.

---

## Heuristic-by-Heuristic Analysis

### ✅ 1. Elevation Heuristic (`elevation.py`)
**Required Data:**
- `elevation` (from DEM)
- `dem_grid` (for fallback sampling)

**Status**: ✅ **Available**
- Dataset: `data/dem/wyoming_dem.tif`
- Integration: Fully integrated in `DataContextBuilder._load_static_layers()` and `build_context()`

---

### ⚠️ 2. Snow Conditions Heuristic (`snow.py`)
**Required Data:**
- `snow_depth_inches`
- `snow_crust_detected` (optional)
- `elevation` (for context)

**Status**: ⚠️ **Placeholder Implementation**
- Current: `SNOTELClient` exists but uses placeholder `_estimate_snow_from_elevation()`
- Real Implementation Needed:
  - SNOTEL API integration (USDA Natural Resources Conservation Service)
  - Or: SNODAS (Snow Data Assimilation System) gridded data
- **Where to get it:**
  - **SNOTEL API**: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data
  - **SNODAS**: https://nsidc.org/data/g02158 (NOAA National Operational Hydrologic Remote Sensing Center)
- **Integration Steps:**
  1. Implement real SNOTEL station lookup and data fetching
  2. Optionally add SNODAS gridded snow depth as fallback
  3. Update `SNOTELClient.get_snow_data()` to use real API

---

### ✅ 3. Water Distance Heuristic (`water.py`)
**Required Data:**
- `water_distance_miles`
- `water_reliability` (0-1 scale)
- `water_sources` (GeoDataFrame)

**Status**: ✅ **Available**
- Dataset: `data/hydrology/water_sources.geojson`
- Integration: Fully integrated via `_calculate_water_metrics()`
- Source: Processed from NHD (National Hydrography Dataset)

---

### ⚠️ 4. Vegetation Quality Heuristic (`vegetation.py`)
**Required Data:**
- `ndvi` (Normalized Difference Vegetation Index)
- `irg` (Instantaneous Rate of Green-up, optional)
- `precip_last_7_days_inches` (optional)
- `land_cover_type` (from NLCD)
- `cloud_cover_percent` (optional)
- `ndvi_age_days` (data recency)

**Status**: ⚠️ **Placeholder + Partial**
- `land_cover_type`: ✅ Available (from NLCD)
- `ndvi`: ⚠️ Placeholder in `SatelliteClient.get_ndvi()`
- Real Implementation Needed:
  - Landsat 8/9 or Sentinel-2 NDVI data
  - Or: MODIS NDVI (250m resolution, daily)
- **Where to get it:**
  - **Google Earth Engine**: https://earthengine.google.com/ (free, requires account)
  - **USGS EarthExplorer**: https://earthexplorer.usgs.gov/ (free, Landsat)
  - **MODIS via AppEEARS**: https://appeears.earthdatacloud.nasa.gov/ (free, daily NDVI)
  - **Sentinel Hub**: https://www.sentinel-hub.com/ (paid, high resolution)
- **Integration Steps:**
  1. Choose data source (recommend Google Earth Engine for ease of use)
  2. Implement NDVI calculation from Landsat or use pre-computed MODIS
  3. Calculate IRG (rate of change) from time series
  4. Add cloud masking and data quality flags
  5. Update `SatelliteClient.get_ndvi()` and `get_integrated_ndvi()`

---

### ❌ 5. Hunting Pressure Heuristic (`access.py`)
**Required Data:**
- `road_distance_miles`
- `trail_distance_miles`
- `security_habitat_percent` (for context)

**Status**: ❌ **Missing Infrastructure Data**
- Dataset Expected: `data/infrastructure/roads.geojson`
- Dataset Expected: `data/infrastructure/trails.geojson`
- Current: Code expects these files but they don't exist in data directory
- **Where to get it:**
  - **Roads (TIGER/Line):** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
    - Download "Roads" shapefile for Wyoming
    - Filter to relevant road types (primary, secondary, tertiary)
  - **Trails (USGS National Map):** https://apps.nationalmap.gov/downloader/
    - Select "Trails" layer
    - Download for Wyoming extent
  - **OpenStreetMap (alternative):** https://www.openstreetmap.org/
    - Extract roads/trails via Overpass API or Geofabrik downloads
    - More complete trail coverage, but requires processing
- **Integration Steps:**
  1. Download TIGER/Line roads for Wyoming
  2. Download USGS National Map trails for Wyoming
  3. Clip to Wyoming boundary
  4. Convert to GeoJSON and save to `data/infrastructure/roads.geojson` and `trails.geojson`
  5. Code already handles loading - just need files!

---

### ⚠️ 6. Security Habitat Heuristic (`security.py`)
**Required Data:**
- `slope_degrees`
- `canopy_cover_percent`
- `road_distance_miles`
- `trail_distance_miles`
- `security_habitat_percent` (calculated from above)

**Status**: ⚠️ **Partial**
- `slope_degrees`: ✅ Available (`data/terrain/slope.tif`)
- `canopy_cover_percent`: ✅ Available (`data/canopy/canopy_cover.tif`)
- `road_distance_miles`: ❌ Missing (see Hunting Pressure above)
- `trail_distance_miles`: ❌ Missing (see Hunting Pressure above)
- **Integration Steps:**
  - Add roads/trails datasets (same as Hunting Pressure heuristic)
  - Code already calculates `security_habitat_percent` from slope/canopy, but needs roads/trails for complete calculation

---

### ❌ 7. Predation Risk Heuristic (`predat  ion.py`)
**Required Data:**
- `wolves_per_1000_elk`
- `bear_activity_distance_miles`
- `snow_depth_inches` (for wolf hunting efficiency)
- `security_habitat_percent` (for context)
- `wolf_data_quality` (metadata)
- `bear_data_quality` (metadata)

**Status**: ❌ **Missing Wildlife Data**
- Dataset Expected: `data/wildlife/wolf_packs.geojson`
- Dataset Expected: `data/wildlife/bear_activity.geojson`
- Current: Code expects these files but they don't exist
- **Where to get it:**
  - **Wolf Pack Territories:**
    - **Wyoming Game & Fish Department (WGFD):** Contact GIS department or check annual wolf reports
      - Annual Wolf Report: https://wgfd.wyo.gov/Wildlife-in-Wyoming/More-Wildlife/Wolf-Management
      - May need to request GIS data directly
    - **USFWS Northern Rocky Mountain Wolf Recovery Program:** Historical data available
    - **Research Publications:** Many wolf studies publish pack territory maps
    - **Note:** This data may require manual compilation from annual reports or direct agency request
  - **Bear Activity:**
    - **Wyoming Game & Fish Department:** Request grizzly bear activity/conflict data
    - **USGS Grizzly Bear Recovery Program:** https://www.usgs.gov/centers/norock/science/grizzly-bear-recovery-program
    - **Research Publications:** Look for spatial conflict/activity data
    - **Note:** May need to use conflict incident locations as proxy for activity centers
- **Integration Steps:**
  1. Obtain wolf pack territory polygons (or centroids with pack size)
  2. Obtain bear activity/conflict locations (point data)
  3. Convert to GeoJSON format with attributes:
     - Wolf packs: `pack_size`, `elk_count` (optional), `territory_area`
     - Bears: `activity_type`, `season`, `year` (optional)
  4. Save to `data/wildlife/wolf_packs.geojson` and `bear_activity.geojson`
  5. Code already handles loading - just need files!

---

### ⚠️ 8. Nutritional Condition Heuristic (`nutrition.py`)
**Required Data:**
- `summer_integrated_ndvi` (June-September integrated NDVI)
- `predation_score` (from PredationRiskHeuristic)
- `pregnancy_rate` (defaulted to 0.90)

**Status**: ⚠️ **Depends on Other Data**
- `summer_integrated_ndvi`: ⚠️ Depends on NDVI data (see Vegetation Quality)
- `predation_score`: ❌ Depends on predator data (see Predation Risk)
- `pregnancy_rate`: ✅ Defaulted (could use research values if available)
- **Integration Steps:**
  - Resolve NDVI and predator data dependencies first
  - Nutritional condition will work once those are available

---

### ⚠️ 9. Winter Severity Heuristic (`winterkill.py`)
**Required Data:**
- `snow_depth` (daily, Nov 1 - Apr 30)
- `temperature` (daily, °F)
- Historical data for cumulative WSI calculation

**Status**: ⚠️ **Placeholder Implementation**
- Current: `_get_snow_depth()` and `_get_temperature()` are placeholders
- Real Implementation Needed:
  - Historical snow depth (SNOTEL or SNODAS)
  - Historical temperature (NOAA weather stations or gridded data)
- **Where to get it:**
  - **Snow:** Same as Snow Conditions heuristic (SNOTEL API or SNODAS)
  - **Temperature:** 
    - **NOAA Climate Data Online (CDO):** https://www.ncdc.noaa.gov/cdo-web/
    - **PRISM Climate Data:** https://prism.oregonstate.edu/ (gridded, high resolution)
    - **GHCN-Daily:** Global Historical Climatology Network
- **Integration Steps:**
  1. Implement historical SNOTEL data retrieval (or SNODAS)
  2. Implement historical temperature data retrieval (PRISM or NOAA CDO)
  3. Update `WinterSeverityHeuristic._get_snow_depth()` and `_get_temperature()`
  4. Cache historical data to avoid repeated API calls

---

## Summary: Missing Datasets

### Critical (Required for Heuristics to Function)
1. ❌ **Roads** (`data/infrastructure/roads.geojson`)
2. ❌ **Trails** (`data/infrastructure/trails.geojson`)
3. ❌ **Wolf Pack Territories** (`data/wildlife/wolf_packs.geojson`)
4. ❌ **Bear Activity** (`data/wildlife/bear_activity.geojson`)

### Partial (Using Placeholders)
5. ⚠️ **SNOTEL/Snow Data** (real API integration needed)
6. ⚠️ **Weather Data** (real API integration needed)
7. ⚠️ **Satellite/NDVI Data** (real data source needed)

---

## Integration Priority

### Priority 1: Infrastructure Data (Roads & Trails)
**Why**: Required by Hunting Pressure and Security Habitat heuristics  
**Effort**: Low (download and convert)  
**Impact**: High (enables 2 heuristics)

**Quick Start:**
```bash
# Download TIGER/Line roads for Wyoming (2023)
# URL: https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_56_roads.zip
# Extract and convert to GeoJSON

# Download USGS trails for Wyoming
# Use National Map Downloader: https://apps.nationalmap.gov/downloader/
# Select Wyoming, layer "Trails"
```

### Priority 2: Wildlife Data (Wolves & Bears)
**Why**: Required by Predation Risk heuristic  
**Effort**: Medium (may require manual compilation or agency request)  
**Impact**: High (enables predation modeling)

**Quick Start:**
1. Check WGFD website for annual wolf reports (may contain maps)
2. Contact WGFD GIS department for shapefiles
3. Compile from research publications if needed
4. For bears, use conflict incident locations as activity proxies

### Priority 3: Real-Time Data Sources (Snow, Weather, NDVI)
**Why**: Improves accuracy of multiple heuristics  
**Effort**: High (API integration, data processing)  
**Impact**: High (better predictions)

**Recommended Approach:**
1. Start with Google Earth Engine for NDVI (easiest)
2. Use SNOTEL API for snow (already have client structure)
3. Use PRISM for historical temperature (gridded, reliable)

---

## Recommended Integration Scripts

You may want to create scripts similar to your existing data processing scripts:

1. `scripts/download_tiger_roads.py` - Download and process TIGER/Line roads
2. `scripts/download_usgs_trails.py` - Download and process USGS trails
3. `scripts/compile_wolf_data.py` - Compile wolf pack territories from WGFD/reports
4. `scripts/compile_bear_data.py` - Compile bear activity from WGFD/conflict data
5. `scripts/integrate_snotel_real.py` - Replace SNOTEL placeholder with real API
6. `scripts/integrate_ndvi_earthengine.py` - Integrate Google Earth Engine NDVI
7. `scripts/integrate_weather_prism.py` - Integrate PRISM temperature data

---

## Testing Recommendations

Once datasets are integrated:

1. **Unit Tests**: Test each heuristic with real data
2. **Integration Tests**: Test `DataContextBuilder.build_context()` with all datasets
3. **Validation**: Compare heuristic outputs with known elk GPS collar locations
4. **Edge Cases**: Test with missing data, boundary conditions, etc.

---

## Notes

- The code structure is already well-designed to handle these datasets - most work is in data acquisition and preparation
- Many datasets may need to be updated periodically (e.g., wolf pack territories change year-to-year)
- Consider caching strategies for expensive API calls (SNOTEL, NDVI)
- For production, consider pre-processing and storing temporal data (snow, weather, NDVI) in a time-series database

