# Dataset Gap Analysis for PathWild Scoring Heuristics

This document analyzes which datasets are required for each scoring heuristic and identifies gaps in the current data pipeline.

## Executive Summary

**Status**: ⚠️ **Partially Complete - 9 of 10 datasets integrated**

You have **9 of 10** required datasets fully integrated. One dataset is **missing**, and two data sources are using **placeholder implementations** that need real API integration.

**Last Updated**: 2026-01-04

### Current Status Breakdown
- ✅ **Fully Integrated (9)**: Elevation, Snow (SNOTEL), Water, Slope, Aspect, Canopy, Land Cover, Roads, Trails
- ❌ **Missing (1)**: Wolf/Bear Wildlife Data
- ⚠️ **Using Placeholders (2)**: NDVI/Satellite (intentionally deferred), Weather/Temperature (real data providers integrated, use placeholders if credentials unavailable)

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

### ✅ 2. Snow Conditions Heuristic (`snow.py`)
**Required Data:**
- `snow_depth_inches`
- `snow_crust_detected` (optional)
- `elevation` (for context)

**Status**: ✅ **Fully Implemented**
- Implementation: `AWDBClient` class uses real USDA AWDB REST API
- API Endpoint: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1
- Features:
  - Real-time SNOTEL station lookup and data fetching
  - Automatic fallback to elevation-based estimates when no station nearby
  - Two-level caching (station data + request cache) for performance
  - Data quality tracking (`snow_data_source` field: "snotel" vs "estimate")
- Station Coverage: 36 Wyoming SNOTEL stations loaded from AWDB API
- Fallback: Elevation-based estimation when stations >100km away or unavailable
- **Note**: See `docs/snotel_station_mapping_status.md` for station mapping details

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

**Status**: ⚠️ **Using Placeholders (Intentionally Deferred)**
- `land_cover_type`: ✅ Available (from NLCD)
- `ndvi`: ⚠️ **Currently using placeholder values** (seasonal variation)
- **Decision (2026-01-04)**: Using placeholders for model training. Real NDVI integration deferred.

**Why Placeholders for Now:**
- **Training**: Placeholder values (with seasonal variation) are sufficient for initial model development
- **Inference Requirements**: Real-time inference needs near-instantaneous responses (< few seconds)
- **AppEEARS Limitation**: AppEEARS async API requires minutes to process requests - not suitable for inference
- **Future Solution Needed**: Pre-downloaded raster files (similar to DEM/landcover pattern) for training, and cloud-based solutions for inference

**Future Implementation (Deferred):**
- **Training**: Pre-download MODIS NDVI raster tiles (16-day composites) covering Wyoming extent (2006-2024)
  - Store as geoTIFF files in `data/satellite/ndvi/`
  - Sample values using rasterio (same pattern as DEM/landcover)
  - Spatial: Wyoming-wide mosaics (1-2GB per composite)
  - Temporal: 16-day composites (~414 files for 18-year period)
  - Total storage: ~200-400GB
- **Inference**: Cloud-based solutions (Google Earth Engine API, or pre-computed NDVI time series database)
- **Where to get data:**
  - **MODIS via AppEEARS**: https://appeears.earthdatacloud.nasa.gov/ (for bulk downloads)
  - **Google Earth Engine**: https://earthengine.google.com/ (for API-based inference)
  - **USGS EarthExplorer**: https://earthexplorer.usgs.gov/ (Landsat, for higher resolution)
- **📖 Detailed Integration Guide**: See `docs/ndvi_weather_integration_guide.md` for step-by-step instructions

---

### ✅ 5. Hunting Pressure Heuristic (`access.py`)
**Required Data:**
- `road_distance_miles`
- `trail_distance_miles`
- `security_habitat_percent` (for context)

**Status**: ✅ **Available**
- **Roads Dataset**: `data/infrastructure/roads.geojson`
  - Source: TIGER/Line 2025 Primary/Secondary Roads (`tl_2025_56_prisecroads`)
  - Features: 1,341 road segments
  - Total length: 13,086.7 miles
  - Integration: Fully integrated in `DataContextBuilder._load_static_layers()` and `build_context()`
- **Trails Dataset**: `data/infrastructure/trails.geojson`
  - Source: USGS National Transportation Dataset (`Trans_TrailSegment.shp`)
  - Features: 5,976 trail segments
  - Total length: 8,299.6 miles
  - Integration: Fully integrated in `DataContextBuilder._load_static_layers()` and `build_context()`
- **Verification**: Tested with sample data - distances calculated correctly for all test locations
- **Note**: See `docs/infrastructure_data_verification.md` for complete verification results

---

### ✅ 6. Security Habitat Heuristic (`security.py`)
**Required Data:**
- `slope_degrees`
- `canopy_cover_percent`
- `road_distance_miles`
- `trail_distance_miles`
- `security_habitat_percent` (calculated from above)

**Status**: ✅ **Available**
- `slope_degrees`: ✅ Available (`data/terrain/slope.tif`)
- `canopy_cover_percent`: ✅ Available (`data/canopy/canopy_cover.tif`)
- `road_distance_miles`: ✅ Available (see Hunting Pressure above)
- `trail_distance_miles`: ✅ Available (see Hunting Pressure above)
- `security_habitat_percent`: ✅ Calculated from all above factors
- **Integration**: All required data is available and integrated

---

### ❌ 7. Predation Risk Heuristic (`predation.py`)
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

**Status**: ⚠️ **Partial Implementation**
- `snow_depth`: ✅ Available via `AWDBClient` (can fetch historical data)
- `temperature`: ⚠️ Placeholder in `WeatherClient._get_historical()`
- Real Implementation Needed:
  - Historical temperature data (NOAA weather stations or gridded data)
- **Where to get it:**
  - **Temperature:** 
    - **PRISM Climate Data:** https://prism.oregonstate.edu/ (gridded, high resolution) - **RECOMMENDED**
    - **Open-Meteo:** https://open-meteo.com/ (free, easy API, good for quick setup)
    - **NOAA Climate Data Online (CDO):** https://www.ncdc.noaa.gov/cdo-web/ (station-based)
    - **GHCN-Daily:** Global Historical Climatology Network
- **📖 Detailed Integration Guide**: See `docs/ndvi_weather_integration_guide.md` for step-by-step instructions
- **Integration Steps:**
  1. Implement historical temperature data retrieval (PRISM or Open-Meteo recommended)
  2. Update `WeatherClient._get_historical()` to use real API
  3. Add historical data caching to avoid repeated API calls
  4. Update `WinterSeverityHeuristic._get_temperature()` to use real weather data

---

## Summary: Missing Datasets

### Critical (Required for Heuristics to Function)
1. ❌ **Wolf Pack Territories** (`data/wildlife/wolf_packs.geojson`)
2. ❌ **Bear Activity** (`data/wildlife/bear_activity.geojson`)

### Partial (Using Placeholders)
3. ⚠️ **Weather/Temperature Data** (real API integration needed for historical temperature)
4. ⚠️ **Satellite/NDVI Data** (real data source needed)

---

## Integration Priority

### ✅ Priority 1: Infrastructure Data (Roads & Trails) - **COMPLETE**
**Status**: ✅ Completed 2026-01-04  
**Result**: Both roads and trails datasets successfully integrated and verified

**Completed:**
- ✅ TIGER/Line 2025 roads processed (`data/infrastructure/roads.geojson`)
- ✅ USGS National Transportation Dataset trails processed (`data/infrastructure/trails.geojson`)
- ✅ Integration tested and verified with sample data
- ✅ Both datasets load successfully in DataContextBuilder

**See**: `docs/infrastructure_data_verification.md` for complete verification results

### Priority 1: Wildlife Data (Wolves & Bears)
**Why**: Required by Predation Risk heuristic  
**Effort**: Medium (may require manual compilation or agency request)  
**Impact**: High (enables predation modeling)

**Quick Start:**
1. Check WGFD website for annual wolf reports (may contain maps)
2. Contact WGFD GIS department for shapefiles
3. Compile from research publications if needed
4. For bears, use conflict incident locations as activity proxies

### Priority 3: Real-Time Data Sources (Weather, NDVI)
**Why**: Improves accuracy of multiple heuristics  
**Effort**: High (API integration, data processing)  
**Impact**: High (better predictions)

**Recommended Approach:**
1. Start with Google Earth Engine for NDVI (easiest)
2. Use PRISM for historical temperature (gridded, reliable)
3. Note: SNOTEL is already implemented ✅

---

## Recommended Integration Scripts

You may want to create scripts similar to your existing data processing scripts:

1. `scripts/download_tiger_roads.py` - Download and process TIGER/Line roads
2. `scripts/download_usgs_trails.py` - Download and process USGS trails
3. `scripts/compile_wolf_data.py` - Compile wolf pack territories from WGFD/reports
4. `scripts/compile_bear_data.py` - Compile bear activity from WGFD/conflict data
5. `scripts/integrate_ndvi_earthengine.py` - Integrate Google Earth Engine NDVI
6. `scripts/integrate_weather_prism.py` - Integrate PRISM temperature data

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
- Consider caching strategies for expensive API calls (NDVI, weather)
- For production, consider pre-processing and storing temporal data (weather, NDVI) in a time-series database
- SNOTEL data is already cached efficiently with two-level caching (station data + request cache)

---

## TODO: Remaining Work

### Critical Missing Datasets (Priority 1)

#### ✅ TODO 1: Download and Process TIGER/Line Roads - **COMPLETE**
- [x] Create `scripts/download_tiger_roads.py`
- [x] Download TIGER/Line 2025 roads for Wyoming (FIPS code 56)
  - Used: `tl_2025_56_prisecroads` (primary/secondary roads)
- [x] Filter to relevant road types (primary, secondary)
- [x] Clip to Wyoming boundary
- [x] Convert to GeoJSON format
- [x] Save to `data/infrastructure/roads.geojson` (1,341 segments, 13,086.7 miles)
- [x] Verify file loads correctly in `DataContextBuilder._load_static_layers()`
- [x] Test with `scripts/integrate_environmental_features.py` on sample data

#### ✅ TODO 2: Download and Process USGS Trails - **COMPLETE**
- [x] Download USGS National Transportation Dataset for Wyoming
  - Source: USGS National Map Downloader
  - Used: `Trans_TrailSegment.shp` from TRAN_Wyoming_State_Shape
- [x] Process with `scripts/download_usgs_trails.py`
- [x] Clip to Wyoming boundary
- [x] Convert to GeoJSON format
- [x] Save to `data/infrastructure/trails.geojson` (5,976 segments, 8,299.6 miles)
- [x] Verify file loads correctly in `DataContextBuilder._load_static_layers()`
- [x] Test with `scripts/integrate_environmental_features.py` on sample data

**Completion Date**: 2026-01-04  
**Verification**: See `docs/infrastructure_data_verification.md`

#### TODO 3: Compile Wolf Pack Territory Data
- [ ] Create `scripts/compile_wolf_data.py`
- [ ] Research data sources:
  - [ ] Check WGFD annual wolf reports for maps/data
  - [ ] Contact WGFD GIS department for shapefiles
  - [ ] Search research publications for pack territory polygons
  - [ ] Check USFWS Northern Rocky Mountain Wolf Recovery Program data
- [ ] Compile pack territories with required attributes:
  - `pack_size` (number of wolves)
  - `elk_count` (optional, estimated elk in territory)
  - `territory_area` (optional, calculated from geometry)
  - `year` (optional, for temporal tracking)
- [ ] Convert to GeoJSON format with proper CRS (EPSG:4326)
- [ ] Save to `data/wildlife/wolf_packs.geojson`
- [ ] Verify file loads correctly in `DataContextBuilder._load_static_layers()`
- [ ] Test `_calculate_wolf_density()` with real data

#### TODO 4: Compile Bear Activity Data
- [ ] Create `scripts/compile_bear_data.py`
- [ ] Research data sources:
  - [ ] Request grizzly bear activity/conflict data from WGFD
  - [ ] Check USGS Grizzly Bear Recovery Program data
  - [ ] Search research publications for spatial conflict/activity data
  - [ ] Use conflict incident locations as proxy if activity centers unavailable
- [ ] Compile bear activity locations with required attributes:
  - `activity_type` (conflict, sighting, etc.)
  - `season` (optional, for temporal filtering)
  - `year` (optional, for temporal tracking)
- [ ] Convert to GeoJSON format (point data) with proper CRS (EPSG:4326)
- [ ] Save to `data/wildlife/bear_activity.geojson`
- [ ] Verify file loads correctly in `DataContextBuilder._load_static_layers()`
- [ ] Test `_calculate_distance_to_nearest()` with real data

### Placeholder Implementations (Priority 2)

#### TODO 5: Integrate Real NDVI/Satellite Data - **DEFERRED (Using Placeholders)**
**Status**: ⚠️ **Deferred to focus on model training**  
**Current**: Using placeholder values with seasonal variation  
**Priority**: Low (placeholders sufficient for initial training)

**Decision Rationale:**
- Placeholder values (with seasonal variation) are sufficient for initial model development
- Real-time inference requires near-instantaneous responses (< few seconds)
- AppEEARS async API not suitable for inference (requires minutes to process)
- Pre-downloaded raster approach needed for production training pipeline
- Cloud-based solutions needed for inference

**Future Implementation (When Ready):**

**For Training (Pre-downloaded Rasters):**
- [ ] Pre-download MODIS NDVI raster tiles covering Wyoming extent
  - [ ] Product: MODIS MOD13Q1 (250m resolution, 16-day composites)
  - [ ] Spatial: Wyoming-wide mosaics (4 MODIS tiles: h09v05, h10v05, h09v04, h10v04)
  - [ ] Temporal: 16-day composites for 2006-2024 (18 years × ~23 composites = ~414 files)
  - [ ] Storage: ~200-400GB total (1-2GB per composite mosaicked)
  - [ ] File naming: `wyoming_ndvi_YYYYMMDD.tif` (e.g., `wyoming_ndvi_20240615.tif`)
  - [ ] Storage path: `data/satellite/ndvi/`
- [ ] Create download script: `scripts/download_modis_ndvi.py`
  - [ ] Download MODIS tiles via AppEEARS or NASA Earthdata
  - [ ] Mosaic tiles to Wyoming extent
  - [ ] Clip to Wyoming boundary
  - [ ] Save as geoTIFF files
- [ ] Update `SatelliteClient` to load and sample from raster files (same pattern as DEM/landcover)
  - [ ] Load raster files per time period in `DataContextBuilder._load_static_layers()`
  - [ ] Sample values using rasterio (milliseconds per point, fast and scalable)
  - [ ] Handle temporal lookup (find nearest 16-day composite for given date)
  - [ ] Calculate IRG from time series (neighboring composites)
- [ ] **📖 See `docs/ndvi_weather_integration_guide.md` for additional details**

**For Inference (Cloud-based):**
- [ ] Research cloud-based solutions for real-time NDVI access
  - [ ] Google Earth Engine API (fast, requires account)
  - [ ] Pre-computed NDVI time series database (fastest option)
  - [ ] CDN-hosted NDVI tiles with temporal indexing
- [ ] Implement inference-optimized NDVI client
  - [ ] Near-instantaneous response times (< 1 second)
  - [ ] Handle current/forecast dates
  - [ ] Fallback to cached/computed values if needed

**Benefits of Raster Approach:**
- ✅ Fast: Local raster sampling (milliseconds per point)
- ✅ Scalable: Handles thousands of points efficiently
- ✅ Consistent: Matches pattern used for DEM/landcover/canopy
- ✅ Reliable: No API timeouts or rate limits
- ✅ Cacheable: Files can be cached locally or in cloud storage

#### TODO 6: Integrate Real Weather/Temperature Data
- [ ] **📖 See `docs/ndvi_weather_integration_guide.md` for detailed step-by-step instructions**
- [ ] Choose data source (recommend PRISM or Open-Meteo)
- [ ] Create `scripts/integrate_weather_prism.py` or similar
- [ ] Implement historical temperature retrieval:
  - [ ] PRISM Climate Data (daily, 4km resolution, 1981-present) - **RECOMMENDED**
  - [ ] Or: Open-Meteo (free, easy API, good for quick setup)
  - [ ] Or: NOAA CDO (station data, requires interpolation)
  - [ ] Or: GHCN-Daily (Global Historical Climatology Network)
- [ ] Update `WeatherClient._get_historical()` to use real API
- [ ] Implement forecast data (for future dates):
  - [ ] Open-Meteo API (free, up to 16 days)
  - [ ] Or: NOAA NWS API (weather.gov)
  - [ ] Or: OpenWeatherMap API (requires API key)
- [ ] Update `WeatherClient._get_forecast()` to use real API
- [ ] Add caching for expensive API calls
- [ ] Update `WinterSeverityHeuristic._get_temperature()` to use real weather data
- [ ] Test with historical dates from elk GPS collar data
- [ ] Verify temperature values are reasonable for Wyoming (-40°F to 100°F range)

### Testing and Validation (Priority 3)

#### TODO 7: Test Infrastructure Data Integration
- [ ] Run `scripts/integrate_environmental_features.py` with roads/trails data
- [ ] Verify `road_distance_miles` and `trail_distance_miles` are calculated correctly
- [ ] Check that distances are reasonable (0-50 miles typical range)
- [ ] Test edge cases (locations far from roads/trails)
- [ ] Validate that `HuntingPressureHeuristic` works with real data
- [ ] Validate that `SecurityHabitatHeuristic` works with real data

#### TODO 8: Test Wildlife Data Integration
- [ ] Run `scripts/integrate_environmental_features.py` with wolf/bear data
- [ ] Verify `wolves_per_1000_elk` is calculated correctly
- [ ] Verify `bear_activity_distance_miles` is calculated correctly
- [ ] Check that values are reasonable:
  - Wolf density: 0-10 wolves per 1000 elk typical
  - Bear distance: 0-50 miles typical
- [ ] Test edge cases (locations outside pack territories, far from bear activity)
- [ ] Validate that `PredationRiskHeuristic` works with real data

#### TODO 9: Test NDVI Integration
- [ ] Run `scripts/integrate_environmental_features.py` with real NDVI data
- [ ] Verify NDVI values are in valid range (0-1)
- [ ] Check that IRG (rate of change) is calculated correctly
- [ ] Verify `summer_integrated_ndvi` is calculated correctly for nutritional condition
- [ ] Test with various dates (summer high NDVI, winter low NDVI)
- [ ] Validate that `VegetationQualityHeuristic` works with real data
- [ ] Validate that `NutritionalConditionHeuristic` works with real data

#### TODO 10: Test Weather Integration
- [ ] Run `scripts/integrate_environmental_features.py` with real weather data
- [ ] Verify temperature values are reasonable for Wyoming (-40°F to 100°F)
- [ ] Check that `precip_last_7_days_inches` is calculated correctly
- [ ] Test with historical dates from elk GPS collar data
- [ ] Test with future dates (forecast data)
- [ ] Validate that `WinterSeverityHeuristic` works with real temperature data

#### TODO 11: End-to-End Integration Testing
- [ ] Run full data pipeline with all datasets integrated
- [ ] Verify `DataContextBuilder.build_context()` works with all datasets
- [ ] Test all heuristics with complete context data
- [ ] Compare heuristic outputs with known elk GPS collar locations
- [ ] Validate model training with complete feature set
- [ ] Check for data quality issues (missing values, outliers, etc.)

### Documentation and Maintenance

#### TODO 12: Update Documentation
- [ ] Update `README.md` with new dataset requirements
- [ ] Document data sources and update procedures
- [ ] Create data acquisition guide for new datasets
- [ ] Update `docs/environmental_data_prerequisites.md` with new datasets
- [ ] Document API keys and authentication setup (if needed)

#### TODO 13: Data Maintenance Procedures
- [ ] Create update scripts for time-sensitive data:
  - [ ] Annual wolf pack territory updates
  - [ ] Seasonal bear activity updates
  - [ ] Road/trail updates (if needed)
- [ ] Document data refresh schedule
- [ ] Set up monitoring for data quality issues
- [ ] Create backup procedures for critical datasets

### Performance Optimization (Future)

#### TODO 14: Optimize Data Access
- [ ] Implement spatial indexing for large vector datasets (roads, trails)
- [ ] Pre-compute distance matrices for common locations
- [ ] Cache expensive API calls (NDVI, weather) more aggressively
- [ ] Consider pre-processing temporal data into time-series database
- [ ] Optimize raster sampling for large-scale batch processing

---

## Progress Tracking

**Last Updated**: 2026-01-04

**Completion Status**:
- ✅ Completed: 9/10 datasets (90%)
- ⚠️ In Progress: 0/10 datasets (0%)
- ❌ Not Started: 1/10 datasets (10%)

**Recent Completions** (2026-01-04):
- ✅ Roads dataset integrated (TIGER/Line 2025)
- ✅ Trails dataset integrated (USGS NTD)

**Next Steps**:
1. Research and compile wolf pack data (TODO 3)
2. Research and compile bear activity data (TODO 4)
3. Integrate real NDVI/satellite data (TODO 5)
4. Integrate real weather/temperature data (TODO 6)

