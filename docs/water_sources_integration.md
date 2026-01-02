# Water Sources Data Integration Guide

This guide documents the workflow for downloading water sources data from the National Hydrography Dataset (NHD), processing it, and integrating it into your PathWild project.

## Overview

Water sources are critical for elk habitat modeling. In October, elk will be within 1 mile of permanent water 90% of the time. The PathWild system uses water source data to calculate:
- Distance to nearest water source (in miles)
- Water reliability (permanent vs ephemeral sources)
- Water type classification (springs, lakes, streams, etc.)

**Key Points:**
- NHD (National Hydrography Dataset) is the recommended source for accuracy
- OpenStreetMap is an alternative but less accurate
- Data is stored as GeoJSON vector format
- Multiple water feature types are included (streams, rivers, lakes, springs)
- Wyoming is covered by 147 HU8 regions (102 northern + 45 southern)

**Data Characteristics:**
- **Format:** GeoJSON (vector)
- **CRS:** WGS84 (EPSG:4326)
- **Feature Types:** Streams, rivers, lakes, ponds, springs, seeps
- **Reliability:** High for NHD (0.9), variable for OSM (0.6-0.8)

## ✅ Successfully Completed Workflow

**Status:** ✅ **Fully Integrated and Verified**

The following workflow was successfully tested and achieved **100% Wyoming coverage** and **successful integration** into the dataset:

### Data Collection and Processing

1. **Downloaded 102 northern Wyoming files** (HUC4 1002-1012) using automated URL generation
2. **Downloaded 45 southern Wyoming files** (HUC4 1013, 1014, 1015, 1018, 1019, 1401, 1404, 1405) using manual URL discovery from National Map Downloader
3. **Extracted all shapefiles** using `scripts/extract_all_nhd_shapefiles.py`
4. **Processed into GeoJSON** using `scripts/process_nhd_water_sources.py`
5. **Verified coverage** using `scripts/verify_wyoming_coverage.py`

**Processing Result:**
- ✅ 100% Wyoming coverage
- ✅ 958,440 water features
- ✅ 1.1 GB GeoJSON file
- ✅ Balanced distribution across all quadrants

### Integration and Verification

7. **Integrated water features** using `scripts/integrate_environmental_features.py`
8. **Verified integration** using comprehensive analysis

**Integration Result:**
- ✅ Successfully integrated into 16,745-row dataset
- ✅ 100% of rows have real water distance values (no placeholders)
- ✅ 11,669 unique water distances calculated
- ✅ 95.2% of points within 0.5 miles of water (realistic for elk habitat)
- ✅ Processing speed: ~35-40 points/second using spatial indexing
- ✅ Water reliability correctly assigned (0.7 for streams, 1.0 for lakes/springs)

**Performance Optimization:**
- Uses spatial indexing (R-tree) for efficient nearest-neighbor queries
- Water sources converted to UTM Zone 12N (EPSG:32612) at load time
- Eliminates GeoPandas warnings about geographic CRS
- Provides accurate distance calculations in meters

See [Water Coverage Analysis](./water_coverage_analysis.md) for detailed coverage analysis.

## Step 1: Download NHD Water Sources Data

### Complete Workflow (Successfully Tested)

This workflow was successfully used to download and process all Wyoming water sources data, achieving 100% coverage.

#### Step 1.1: Download Northern Wyoming (Initial)

**Option A: Generate URLs automatically**
```bash
# Generate URLs for northern Wyoming HU8 regions (HUC4 1002-1012)
python scripts/generate_nhd_urls.py --output data/raw/nhd/nhd_download_urls.txt

# Validate URLs
python scripts/validate_nhd_urls.py \
    --urls-file data/raw/nhd/nhd_download_urls.txt \
    --output data/raw/nhd/nhd_download_urls_validated.txt

# Download files
python scripts/download_nhd_water_sources.py \
    --urls-file data/raw/nhd/nhd_download_urls_validated.txt \
    --output-dir data/raw/nhd
```

**Result:** 102 files downloaded covering northern Wyoming (HUC4 1002-1012)

#### Step 1.2: Download Southern Wyoming (Complete Coverage)

**Manual URL Discovery (Recommended for Southern Wyoming)**

1. **Get URLs from National Map Downloader:**
   - Go to: https://apps.nationalmap.gov/downloader/
   - Search for 'NHD High Resolution' product
   - Draw bounding box: West: -111.0°, South: 41.0°, East: -104.0°, North: 42.5°
   - Select all HU8 Shape files that appear
   - Copy download URLs to `data/raw/nhd/southern_wyoming_urls.txt`

2. **Download files:**
   ```bash
   python scripts/download_nhd_water_sources.py \
       --urls-file data/raw/nhd/southern_wyoming_urls.txt \
       --output-dir data/raw/nhd
   ```
   
   **Note:** The original file (with duplicates) can be used - duplicates will be overwritten.

**Result:** 152 files downloaded (including duplicates and northern Wyoming files already present)

#### Step 1.3: Extract and Process All Files

```bash
# Extract all shapefiles from zip archives
python scripts/extract_all_nhd_shapefiles.py

# Process all shapefiles into single GeoJSON
python scripts/process_nhd_water_sources.py \
    --input-dir data/raw/nhd/shapefiles \
    --output data/hydrology/water_sources.geojson
```

#### Step 1.4: Verify Complete Coverage

```bash
python scripts/verify_wyoming_coverage.py data/hydrology/water_sources.geojson
```

**Expected Output:**
- ✅ Coverage: 100.0% of Wyoming
- ✅ Total features: ~958,440
- ✅ Balanced distribution across all quadrants
- ✅ File size: ~1.1 GB

### Alternative: Automated Download (For Initial Setup)

**Use the provided script to automate downloading:**

```bash
# Step 1: Generate URLs for northern Wyoming
python scripts/generate_nhd_urls.py --output data/raw/nhd/nhd_download_urls.txt

# Step 2: Validate URLs
python scripts/validate_nhd_urls.py \
    --urls-file data/raw/nhd/nhd_download_urls.txt \
    --output data/raw/nhd/nhd_download_urls_validated.txt

# Step 3: Download all files automatically
python scripts/download_nhd_water_sources.py \
    --urls-file data/raw/nhd/nhd_download_urls_validated.txt \
    --output-dir data/raw/nhd \
    --workers 4
```

**What the script does:**
- Downloads all NHD files in parallel (4 workers by default)
- Shows progress for each download
- Handles retries and errors gracefully
- Skips already-downloaded files

**Note:** Wyoming has ~147 HU8 (8-digit Hydrologic Unit Code) regions total (102 northern + 45 southern), which is why there are so many files. The automated download script will handle all of them efficiently.

### Option B: Manual Download via USGS National Map Downloader

If you prefer to download manually:

1. **Go to:** https://apps.nationalmap.gov/downloader/
2. **Select data type:**
   - Click **"Data"** tab
   - Select **"National Hydrography Dataset (NHD)"**
   - Choose **"NHD High Resolution"** (not Medium Resolution)
3. **Select area:**
   - Click **"Use Map"** or **"Draw Rectangle"**
   - Draw rectangle around Wyoming or enter coordinates:
     - North: 45.0, South: 41.0
     - East: -104.0, West: -111.0
4. **Select products:**
   - Check **"NHDFlowline"** (streams and rivers)
   - Check **"NHDWaterbody"** (lakes and ponds)
   - Check **"NHDArea"** (large water areas)
   - Optionally check **"NHDSpring"** (springs and seeps)
5. **Download:**
   - Click **"Find Products"**
   - Select all files for Wyoming (typically 20 subregions)
   - Click **"Download"** for each file
   - Files will be zip archives

**Method 2: NHDPlus HR (Alternative - More Complete)**

1. **Go to:** https://www.usgs.gov/national-hydrography/nhdplus-high-resolution
2. **Find Wyoming subregions:**
   - Wyoming is covered by multiple NHDPlus HR subregions
   - Look for subregions covering Wyoming (e.g., 14, 15, 16, 17)
3. **Download:**
   - Download the complete subregion files
   - Extract the NHD components

**Method 3: Direct FTP Access (Advanced)**

1. **FTP Server:** ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Hydrography/NHD/HighResolution/GPKG/
2. **Navigate to Wyoming subregions**
3. **Download:** GPKG or Shapefile format

**What to download:**
- **NHDFlowline:** Streams and rivers (most important)
- **NHDWaterbody:** Lakes and ponds
- **NHDArea:** Large water areas
- **NHDSpring:** Springs and seeps (optional but useful)

**File sizes:**
- NHDFlowline: ~50-200 MB per subregion
- NHDWaterbody: ~10-50 MB per subregion
- Total for Wyoming: ~500 MB - 1 GB (compressed)

### Option B: OpenStreetMap (Easier, Less Accurate)

OpenStreetMap provides water features but may be less complete, especially in remote areas.

**Using Python (osmnx):**

```python
import osmnx as ox
import geopandas as gpd

# Wyoming bounding box
wyoming_bbox = (41.0, -111.0, 45.0, -104.0)  # (south, west, north, east)

# Download water features
water = ox.features_from_bbox(
    bbox=wyoming_bbox,
    tags={
        'waterway': True,
        'natural': ['water', 'spring'],
        'water': True
    }
)

# Convert to GeoDataFrame
water_gdf = gpd.GeoDataFrame(water, crs='EPSG:4326')
water_gdf.to_file('data/hydrology/water_sources.geojson', driver='GeoJSON')
```

**Note:** OSM data may be incomplete in remote Wyoming areas. NHD is recommended for production use.

## Step 2: Create Hydrology Directory

Create the directory for water sources data:

```bash
mkdir -p data/hydrology
```

## Step 3: Process NHD Data

Use the provided script to process NHD data and convert to GeoJSON:

```bash
python scripts/process_nhd_water_sources.py \
    --nhd-flowline path/to/NHDFlowline.shp \
    --nhd-waterbody path/to/NHDWaterbody.shp \
    --output data/hydrology/water_sources.geojson
```

**If you have multiple subregions:**

```bash
# Process each subregion, then merge
python scripts/process_nhd_water_sources.py \
    --nhd-flowline data/raw/nhd_subregion_14/NHDFlowline.shp \
    --nhd-waterbody data/raw/nhd_subregion_14/NHDWaterbody.shp \
    --output data/hydrology/water_sources_subregion_14.geojson

python scripts/process_nhd_water_sources.py \
    --nhd-flowline data/raw/nhd_subregion_15/NHDFlowline.shp \
    --nhd-waterbody data/raw/nhd_subregion_15/NHDWaterbody.shp \
    --output data/hydrology/water_sources_subregion_15.geojson

# Merge all subregions
python scripts/merge_water_sources.py \
    --input-dir data/hydrology \
    --output data/hydrology/water_sources.geojson
```

**What the script does:**
1. Loads NHD shapefiles (NHDFlowline, NHDWaterbody, etc.)
2. Filters for relevant water types:
   - StreamRiver (FType 460)
   - LakePond (FType 390)
   - SpringSeep (FType 388)
   - Other permanent water features
3. Clips to Wyoming bounds (if needed)
4. Converts to WGS84 (EPSG:4326)
5. Adds water type and reliability attributes
6. Saves as GeoJSON

**Expected output:**
- File: `data/hydrology/water_sources.geojson`
- Size: ~1.1 GB (for complete Wyoming coverage)
- Format: GeoJSON
- CRS: WGS84 (EPSG:4326)
- Features: ~958,440 water features (streams, rivers, lakes, ponds, springs)
- Coverage: 100% of Wyoming

## Step 4: Verify the Processed File

Verify the processed water sources file:

```bash
python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson
```

**What to look for:**

✅ **Good signs:**
- Thousands of features (10,000+ for Wyoming)
- Multiple water types: streams, rivers, lakes, springs
- Geographic coverage: "✓ File covers Wyoming"
- Realistic distribution:
  - Most features: streams/rivers (linear features)
  - Some features: lakes/ponds (polygon features)
  - Few features: springs (point features)
- CRS: EPSG:4326 (WGS84)
- Attributes include: `type`, `reliability`, `name` (if available)

❌ **Warning signs:**
- Very few features (< 1,000) → Incomplete data or wrong area
- All features same type → Filtering issue
- No geographic coverage → Wrong area or CRS issue
- Missing attributes → Processing issue

**Expected output:**
```
File: data/hydrology/water_sources.geojson
File size: 1.1 GB
Features: 958,440

--- METADATA ---
CRS: EPSG:4326 (WGS84)
Bounds: [-111.12, 40.45, -102.03, 45.65]
Geometry types: LineString, Polygon, Point

--- FEATURE TYPES ---
Stream/River: ~875,000 (91.3%)
Lake/Pond: ~83,000 (8.7%)
Spring/Seep: ~500 (0.0%)

--- RELIABILITY ---
High (0.9-1.0): ~83,500 (8.7%)  # Lakes, springs
Medium (0.7-0.9): ~875,000 (91.3%)  # Streams, rivers
Low (0.4-0.7): 0 (0.0%)

--- COVERAGE ---
✓ Coverage: 100.0% of Wyoming
✓ All borders covered
✓ Balanced distribution across quadrants
✓ Multiple feature types
✓ Realistic distribution
```

## Step 5: Integrate Environmental Features

Once the water sources file is in place, integrate environmental features into your dataset:

```bash
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What the script does:**
1. Loads your dataset
2. Initializes `DataContextBuilder` which:
   - Loads water sources GeoJSON (958,440 features)
   - Converts to projected CRS (UTM Zone 12N) for accurate distance calculations
   - Builds spatial index for efficient nearest-neighbor queries
3. For each point, calculates:
   - Distance to nearest water source (in miles) using spatial indexing
   - Water reliability (0.7 for streams, 1.0 for lakes/springs)
   - Water type (spring, lake, stream, etc.)
4. Updates the dataset with real values (replaces placeholder 0.5 miles)
5. Saves the updated dataset

**Performance:**
- Uses spatial indexing (R-tree) for fast nearest-neighbor queries
- Processes ~35-40 points/second
- Expected runtime: ~7-10 minutes for 16,745 points
- Progress is saved every 1000 rows

**Note:** The `DataContextBuilder` automatically loads water sources from `data/hydrology/water_sources.geojson` if it exists. The water sources are converted to UTM Zone 12N (EPSG:32612) at load time for accurate distance calculations and to avoid GeoPandas warnings.

## Step 6: Verify Integration

Analyze the integrated dataset to verify everything worked:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**Actual Integration Results (from successful run):**

The integration was successfully completed with the following results:

- **Total rows processed:** 16,745
- **Water features integrated:** 100% (no placeholder values)
- **Unique water distances:** 11,669
- **Water distance range:** 0.000 to 41.919 miles
- **Mean water distance:** 0.526 miles
- **Median water distance:** 0.104 miles

**Water Distance Distribution:**
- < 0.01 miles: 1,068 (6.4%) — on or very close to water
- 0.01-0.1 miles: 7,027 (42.0%) — very close
- 0.1-0.5 miles: 7,846 (46.9%) — close
- 0.5-1.0 miles: 231 (1.4%)
- 1.0-2.0 miles: 72 (0.4%)
- 2.0-5.0 miles: 130 (0.8%)
- 5.0-10.0 miles: 121 (0.7%)
- > 10.0 miles: 250 (1.5%) — far from water

**Water Reliability Distribution:**
- 0.70 (streams): 15,551 (92.9%)
- 1.00 (lakes/springs): 1,194 (7.1%)

**Geographic Coverage:**
- Full Wyoming coverage: 41.00° to 45.00°N, -111.00° to -104.06°W
- Northern Wyoming: mean 0.122 miles to water
- Central Wyoming: mean 1.543 miles to water
- Southern Wyoming: mean 0.412 miles to water

**What to look for:**

✅ **Success indicators:**
- **Water distance:** Good diversity (0.0 to 5+ miles)
- **Water distance distribution:** Realistic for Wyoming:
  - Most points: 0.0-1.0 miles (elk near water)
  - Some points: 1.0-2.0 miles (acceptable)
  - Few points: 2.0+ miles (marginal)
- **Water reliability:** Range 0.4-1.0
- **Water types:** Multiple types (streams, lakes, springs)
- **Elevation:** Good diversity (thousands of unique values)
- **Slope:** Good diversity, reasonable range (0-60°)
- **Aspect:** Full 0-360° range
- **Land cover:** Multiple codes

❌ **Issues to watch for:**
- Water distance: Only one value (0.5) → Placeholder values not replaced, check file path
- Water distance: All values very high (> 5 miles) → Incomplete water data or wrong area
- Water distance: All values 0.0 → Data issue or all points on water
- Water reliability: All same value → Attribute processing issue

**Example good output:**
```
--- WATER DISTANCE ANALYSIS ---
Valid values: 16,745 (100.0%)
Range: 0.01 to 4.87 miles
Mean: 0.68 miles
Median: 0.42 miles
  ✓ Good diversity: 1,234 unique values
  ✓ Values are realistic for elk habitat
  ✓ Distribution matches expected pattern

  Water distance distribution:
    Very close (0-0.25 mi)    :   4,123 (24.6%)
    Close (0.25-0.5 mi)       :   5,234 (31.2%)
    Acceptable (0.5-1.0 mi)   :   4,567 (27.3%)
    Marginal (1.0-2.0 mi)     :   2,456 (14.7%)
    Far (2.0+ mi)             :     365 ( 2.2%)

--- WATER RELIABILITY ANALYSIS ---
Range: 0.4 to 1.0
Mean: 0.78
  ✓ Good diversity of reliability values
  ✓ Most sources are permanent (high reliability)
```

## Understanding Water Distance Values

Water distance values represent the distance (in miles) from each elk observation point to the nearest water source:

- **0.0-0.25 miles:** Very close to water (optimal)
- **0.25-0.5 miles:** Close to water (good)
- **0.5-1.0 miles:** Acceptable distance
- **1.0-2.0 miles:** Marginal (elk may still use area)
- **2.0+ miles:** Far from water (less suitable)

**For elk in October:**
- 90% of observations should be within 1 mile of water
- Most observations should be within 0.5 miles
- Very few observations should be > 2 miles from water

## Understanding Water Reliability

Water reliability indicates how permanent/reliable a water source is:

- **1.0:** Permanent sources (springs, large lakes)
- **0.9:** Very reliable (permanent streams, large ponds)
- **0.7:** Reliable (perennial streams, rivers)
- **0.4:** Ephemeral (seasonal streams, temporary water)

**For elk habitat:**
- Elk prefer permanent water sources
- Ephemeral sources reduce habitat suitability
- Reliability affects the water distance heuristic score

## Technical Implementation Details

### Spatial Indexing for Performance

The integration uses spatial indexing (R-tree) for efficient nearest-neighbor queries:

- **Initial Implementation:** Used `unary_union()` which was extremely slow (processing only a few rows per minute)
- **Optimized Implementation:** Uses `gpd.sjoin_nearest()` with spatial indexing
- **Performance Improvement:** ~35-40 points/second (1000x faster)
- **Memory Efficiency:** Spatial index built once and reused for all queries

### Coordinate Reference System (CRS) Transformation

To ensure accurate distance calculations and eliminate GeoPandas warnings:

- **Source CRS:** WGS84 (EPSG:4326) - geographic coordinates
- **Projected CRS:** UTM Zone 12N (EPSG:32612) - projected coordinates for Wyoming
- **Transformation:** Water sources are converted to UTM at load time and cached
- **Benefits:**
  - Accurate distance calculations in meters
  - Eliminates warnings about geographic CRS
  - Single transformation per dataset (not per query)

**Implementation in `DataContextBuilder`:**
```python
# Water sources are converted to UTM Zone 12N once during initialization
self.water_sources_proj = self.water_sources.to_crs('EPSG:32612')

# Nearest-neighbor queries use the projected CRS
nearest = gpd.sjoin_nearest(point_gdf, self.water_sources_proj, ...)
```

### Water Reliability Assignment

Water reliability is assigned based on NHD feature types:

- **1.0 (High):** Lakes, ponds, springs, seeps (permanent water sources)
- **0.7 (Medium):** Streams, rivers (perennial water sources)

This classification is based on NHD `FType` codes:
- `FType 390` (LakePond) → 1.0
- `FType 388` (SpringSeep) → 1.0
- `FType 460` (StreamRiver) → 0.7

## Troubleshooting

### "Water distance values are all 0.5 (placeholder)"

**Cause:** Water sources file not found or not loading

**Fix:**
1. Check file exists: `ls -lh data/hydrology/water_sources.geojson`
2. Verify file is readable: `python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson`
3. Check `DataContextBuilder` logs for "✓ Water sources loaded" message
4. Re-run integration script: `python scripts/integrate_environmental_features.py`

### "Water distance values are all very high (> 5 miles)"

**Cause:** Incomplete water data or wrong geographic area

**Fix:**
1. Verify water sources cover Wyoming: `python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson`
2. Check if you need to download additional NHD subregions
3. Verify CRS is correct (should be WGS84)

### "UserWarning: Geometry is in a geographic CRS. Results from 'sjoin_nearest' are likely incorrect."

**Cause:** Using geographic CRS (EPSG:4326) for distance calculations

**Fix:**
- This warning has been fixed in the current implementation
- Water sources are automatically converted to UTM Zone 12N (EPSG:32612) at load time
- If you see this warning, ensure you're using the latest version of `src/data/processors.py`
- The `DataContextBuilder` now caches projected GeoDataFrames (`water_sources_proj`, `roads_proj`, `trails_proj`)

### "Integration is extremely slow (only a few rows per minute)"

**Cause:** Using `unary_union()` instead of spatial indexing

**Fix:**
- This performance issue has been fixed in the current implementation
- The code now uses `gpd.sjoin_nearest()` with spatial indexing
- If you experience slowness, ensure you're using the latest version of `src/data/processors.py`

### "Water sources file is too large (> 500 MB)"

**Cause:** Too much detail or unnecessary features included

**Fix:**
1. Filter to only essential water types (streams, lakes, springs)
2. Simplify geometries (reduce vertex count)
3. Remove very small features (e.g., streams < 100m)
4. Use the `--simplify` option in the processing script

### "Processing script fails with 'FType' error"

**Cause:** NHD file structure different than expected

**Fix:**
1. Check NHD file version (High Resolution vs Medium Resolution)
2. Verify you downloaded the correct NHD products
3. Check the script handles your NHD version
4. Update the script if needed to handle different FType codes

### "Water sources not loading in DataContextBuilder"

**Cause:** File path or format issue

**Fix:**
1. Verify file is at: `data/hydrology/water_sources.geojson`
2. Check file is valid GeoJSON: `python -c "import geopandas as gpd; gpd.read_file('data/hydrology/water_sources.geojson')"`
3. Verify CRS is WGS84 (EPSG:4326)
4. Check file permissions

## File Structure

After successful integration, you should have:

```
data/
├── hydrology/
│   └── water_sources.geojson          # Processed water sources (~50-200 MB)
├── dem/
│   └── wyoming_dem.tif                 # Digital elevation model
├── terrain/
│   ├── slope.tif                        # Slope in degrees
│   └── aspect.tif                      # Aspect in degrees
├── landcover/
│   └── nlcd.tif                         # NLCD land cover
├── canopy/
│   └── canopy_cover.tif                # Canopy cover
├── processed/
│   └── combined_north_bighorn_presence_absence_cleaned.csv  # Updated dataset
└── raw/
    └── nhd/                             # Optional: Original NHD files
        ├── subregion_14/
        └── subregion_15/
```

## Quick Reference

**Complete workflow (Automated):**
```bash
# 1. Create directory
mkdir -p data/hydrology data/raw/nhd

# 2. Get download URLs (guided process)
python scripts/download_nhd_water_sources.py --extract-urls

# 3. Add URLs to data/raw/nhd/nhd_download_urls.txt (one per line)

# 4. Download all NHD files automatically
python scripts/download_nhd_water_sources.py \
    --urls-file data/raw/nhd/nhd_download_urls.txt \
    --output-dir data/raw/nhd \
    --extract-shapefiles \
    --shapefiles-dir data/raw/nhd/shapefiles \
    --workers 4

# 5. Process NHD data (combine all subregions)
python scripts/process_nhd_water_sources.py \
    --nhd-flowline data/raw/nhd/shapefiles/NHDFlowline.shp \
    --nhd-waterbody data/raw/nhd/shapefiles/NHDWaterbody.shp \
    --output data/hydrology/water_sources.geojson

# 6. Verify
python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson

# 7. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 8. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

**Complete workflow (Manual):**
```bash
# 1. Create directory
mkdir -p data/hydrology

# 2. Download NHD data from USGS National Map (manual)
#    - Go to: https://apps.nationalmap.gov/downloader/
#    - Select: NHD High Resolution
#    - Select area: Wyoming
#    - Download: NHDFlowline, NHDWaterbody, NHDArea, NHDSpring
#    - Save to: data/raw/nhd/

# 3. Extract shapefiles from zip files (if needed)
#    Unzip all downloaded files to data/raw/nhd/shapefiles/

# 4. Process NHD data
python scripts/process_nhd_water_sources.py \
    --nhd-flowline data/raw/nhd/shapefiles/NHDFlowline.shp \
    --nhd-waterbody data/raw/nhd/shapefiles/NHDWaterbody.shp \
    --output data/hydrology/water_sources.geojson

# 5. Verify
python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson

# 6. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 7. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

## Data Sources and Alternatives

### Primary Source: NHD High Resolution (Recommended)

- **Source:** USGS National Hydrography Dataset
- **Website:** https://apps.nationalmap.gov/downloader/
- **Resolution:** High resolution (detailed)
- **Coverage:** Complete United States
- **Pros:** Most accurate, comprehensive, official USGS data
- **Cons:** Large file sizes, multiple subregions for Wyoming
- **Format:** Shapefile or GeoPackage

### Alternative: NHD Medium Resolution

- **Source:** USGS National Hydrography Dataset
- **Resolution:** Medium resolution (less detailed)
- **Pros:** Smaller file sizes, easier to download
- **Cons:** Less detail, may miss small streams
- **Use case:** Quick testing or if high-res is too large

### Alternative: OpenStreetMap

- **Source:** OpenStreetMap
- **Website:** https://www.openstreetmap.org/
- **Pros:** Easy to download, global coverage
- **Cons:** May be incomplete in remote areas, variable quality
- **Use case:** Quick prototyping or if NHD unavailable

**Recommendation:** Use NHD High Resolution for production accuracy.

## Next Steps

After successful integration:
1. ✅ Dataset is ready for model training
2. ✅ Water distance feature is populated with real values
3. ✅ Water reliability is calculated for each point
4. ✅ Values are realistic for elk habitat (most within 1 mile)
5. ✅ Can be used for water distance heuristic scoring

## Integration Status

**Status:** ⚠️ Ready for Integration

Once you download and process the NHD data following this guide, the water sources will be integrated into your PathWild dataset. The system is already configured to use water sources data when available.

**Expected Results:**
- **Coverage:** 100% of points will have water distance data
- **Value Range:** 0.0 to 5+ miles (realistic for Wyoming)
- **Distribution:** Most points within 1 mile (matching elk behavior)
- **Reliability:** Range 0.4-1.0 (permanent to ephemeral sources)
- **Data Quality:** High accuracy from NHD data

The dataset will be ready for machine learning model training with water distance as a key environmental feature!

