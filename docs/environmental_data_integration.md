# Environmental Data Integration Guide

This guide provides an overview of how to integrate environmental datasets (DEM, water sources, land cover, etc.) into PathWild to replace placeholder values with real environmental features.

**📘 For detailed prerequisite setup instructions, see [Environmental Data Prerequisites Guide](./environmental_data_prerequisites.md)** - This guide includes step-by-step instructions for generating all required environmental data files, including how to adapt the process for other states and geographies.

## Overview

PathWild's `DataContextBuilder` and absence generators expect environmental data in specific directory structures. Once integrated, these datasets will automatically populate environmental features for both presence and absence points.

**Current Status:**
- ✅ **DEM, Slope, Aspect:** Fully integrated - see detailed guide below
- ✅ **Land Cover (NLCD):** Fully integrated - see detailed guide below
- ✅ **Canopy Cover:** Fully integrated - see detailed guide below
- ✅ **Water Sources:** Fully integrated - see detailed guide below
- ⚠️ **Roads/Trails:** Placeholder values (data not yet downloaded)
- ⚠️ **Wildlife Data:** Not yet integrated

## Quick Links to Detailed Guides

For step-by-step instructions on specific datasets:

1. **📘 [Elevation, Slope, and Aspect Integration](./elevation_slope_aspect_integration.md)**
   - Downloading DEM data from USGS
   - Generating slope and aspect rasters
   - Handling placeholder/inappropriate elevations
   - Complete workflow with verification steps

2. **📘 [NLCD Land Cover Integration](./nlcd_landcover_integration.md)**
   - Downloading NLCD land cover data
   - Clipping CONUS dataset to Wyoming
   - Fixing CRS transformation issues
   - Complete workflow with verification steps

3. **📘 [Canopy Cover Integration](./canopy_cover_integration.md)**
   - Downloading NLCD Tree Canopy Cover data
   - Clipping CONUS dataset to Wyoming
   - Verifying percentage values (0-100)
   - Complete workflow with verification steps

4. **📘 [Water Sources Integration](./water_sources_integration.md)**
   - Downloading NHD High Resolution water data
   - Processing NHD shapefiles to GeoJSON
   - Filtering water feature types
   - Complete workflow with verification steps

## Directory Structure

Create the following directory structure under `data/`:

```
data/
├── dem/                    # Digital Elevation Model (raster)
│   └── wyoming_dem.tif
├── terrain/                # Derived terrain rasters
│   ├── slope.tif          # Slope in degrees
│   └── aspect.tif         # Aspect in degrees
├── landcover/              # Land cover classification
│   └── nlcd.tif           # NLCD land cover codes
├── canopy/                 # Vegetation canopy cover
│   └── canopy_cover.tif   # Canopy cover percentage
├── hydrology/              # Water sources (vector)
│   └── water_sources.geojson
├── infrastructure/         # Roads and trails (vector)
│   ├── roads.geojson
│   └── trails.geojson
└── wildlife/              # Predator data (vector)
    ├── wolf_packs.geojson
    └── bear_activity.geojson
```

## Data Sources

### 1. Digital Elevation Model (DEM), Slope, and Aspect

**Status:** ✅ Fully integrated

**Detailed Guide:** See [Elevation, Slope, and Aspect Integration](./elevation_slope_aspect_integration.md)

**Quick Summary:**
- **DEM Source:** USGS 3D Elevation Program (3DEP) 1 arc-second
- **Download:** Automated script downloads from AWS S3
- **Processing:** Generate slope and aspect using Python script
- **Files:**
  - `data/dem/wyoming_dem.tif` (~1-3 GB)
  - `data/terrain/slope.tif` (~40-50 MB)
  - `data/terrain/aspect.tif` (~40-50 MB)

**Quick Start:**
```bash
# Download DEM
python scripts/download_wyoming_dem.py --mosaic

# Generate slope and aspect
python scripts/generate_slope_aspect.py

# Verify
python scripts/analyze_dem.py data/dem/wyoming_dem.tif
```

**Important:** The guide includes instructions for handling placeholder/inappropriate elevations that may appear in your dataset.

### 2. Land Cover (NLCD)

**Status:** ✅ Fully integrated

**Detailed Guide:** See [NLCD Land Cover Integration](./nlcd_landcover_integration.md)

**Quick Summary:**
- **Source:** National Land Cover Database (NLCD) 2021
- **Download:** USGS EarthExplorer (Wyoming too large for MRLC viewer)
- **Processing:** Clip CONUS dataset to Wyoming
- **File:** `data/landcover/nlcd.tif` (~25-50 MB)

**Quick Start:**
```bash
# Download from USGS EarthExplorer (manual)
# Then clip to Wyoming
python scripts/clip_conus_nlcd_to_wyoming.py --zip-file path/to/nlcd.zip

# Verify
python scripts/analyze_nlcd.py data/landcover/nlcd.tif
```

**Important:** The guide includes instructions for fixing CRS transformation issues and verifying the correct product type.

### 3. Canopy Cover

**Status:** ✅ Fully integrated

**Detailed Guide:** See [Canopy Cover Integration](./canopy_cover_integration.md)

**Quick Summary:**
- **Source:** National Land Cover Database (NLCD) Tree Canopy Cover
- **Download:** MRLC Direct Download (https://www.mrlc.gov/data) - CONUS dataset
- **Processing:** Clip CONUS dataset to Wyoming
- **File:** `data/canopy/canopy_cover.tif` (~50-60 MB)

**Quick Start:**
```bash
# 1. Download from MRLC (https://www.mrlc.gov/data) - find "Tree Canopy Cover"
#    Download CONUS dataset (Wyoming is too large for viewer)

# 2. Clip to Wyoming
python scripts/clip_conus_canopy_to_wyoming.py --zip-file path/to/nlcd_tree_canopy.zip

# 3. Verify
python scripts/analyze_canopy.py data/canopy/canopy_cover.tif

# 4. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 5. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

**Integration Results:**
- ✅ Successfully integrated with realistic canopy cover percentages (0-100%)
- ✅ Good diversity: 78 unique values across dataset
- ✅ Distribution matches Wyoming's sparse tree cover (mean ~16.7%, median 0%)
- ✅ Values automatically clamped to valid range (0-100%) during sampling

**Important:** The guide includes instructions for verifying the correct product type (Tree Canopy Cover, not Land Cover) and ensuring values are percentages (0-100).

### 5. Water Sources

**Status:** ✅ Fully integrated

**Detailed Guide:** See [Water Sources Integration](./water_sources_integration.md)

**Quick Summary:**
- **Source:** National Hydrography Dataset (NHD) High Resolution
- **Download:** USGS National Map Downloader (https://apps.nationalmap.gov/downloader/)
- **Processing:** Automated scripts for processing multiple HU8 regions
- **File:** `data/hydrology/water_sources.geojson` (1.1 GB, 958,440 features)
- **Coverage:** 100% of Wyoming

**Quick Start:**
```bash
# 1. Download NHD data from USGS National Map Downloader
#    - Go to: https://apps.nationalmap.gov/downloader/
#    - Search for 'NHD High Resolution'
#    - Draw bounding box covering Wyoming
#    - Copy all HU8 Shape file URLs to data/raw/nhd/southern_wyoming_urls.txt

# 2. Download files
python scripts/download_nhd_water_sources.py \
    --urls-file data/raw/nhd/southern_wyoming_urls.txt \
    --output-dir data/raw/nhd

# 4. Extract and process
python scripts/extract_all_nhd_shapefiles.py
python scripts/process_nhd_water_sources.py \
    --input-dir data/raw/nhd/shapefiles \
    --output data/hydrology/water_sources.geojson

# 5. Verify coverage
python scripts/verify_wyoming_coverage.py data/hydrology/water_sources.geojson

# 6. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 7. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

**Integration Results:**
- ✅ Complete Wyoming coverage (100%)
- ✅ 958,440 water features (streams, rivers, lakes, springs)
- ✅ Balanced distribution across all quadrants
- ✅ File size: 1.1 GB GeoJSON

**Actual Integration Performance:**
- ✅ Successfully integrated into 16,745-row dataset
- ✅ 100% of rows have real water distance values (no placeholders)
- ✅ 11,669 unique water distances calculated
- ✅ 95.2% of points within 0.5 miles of water (realistic for elk habitat)
- ✅ Processing speed: ~35-40 points/second using spatial indexing
- ✅ Water reliability correctly assigned (0.7 for streams, 1.0 for lakes/springs)

**Performance Optimization:**
The integration uses spatial indexing (R-tree) for efficient nearest-neighbor queries:
- Water sources are converted to UTM Zone 12N (EPSG:32612) at load time
- Spatial index built once and reused for all queries
- Eliminates GeoPandas warnings about geographic CRS
- Provides accurate distance calculations in meters

**Important:** The guide includes detailed instructions for downloading NHD data, cleaning URLs, processing multiple HU8 regions, verifying complete coverage, and integrating into datasets.

### 6. Roads and Trails

**Source:** OpenStreetMap or USGS National Transportation Dataset

**OpenStreetMap (Easier):**
```python
import osmnx as ox
import geopandas as gpd

wyoming_bbox = (41.0, -111.0, 45.0, -104.0)

# Download roads
roads = ox.features_from_bbox(
    bbox=wyoming_bbox,
    tags={'highway': True}
)

# Filter for major roads (optional)
major_roads = roads[roads['highway'].isin(['primary', 'secondary', 'tertiary', 'trunk'])]

# Convert to GeoJSON
roads_gdf = gpd.GeoDataFrame(major_roads, crs='EPSG:4326')
roads_gdf.to_file('data/infrastructure/roads.geojson', driver='GeoJSON')

# Download trails
trails = ox.features_from_bbox(
    bbox=wyoming_bbox,
    tags={'highway': ['path', 'track', 'footway']}
)

trails_gdf = gpd.GeoDataFrame(trails, crs='EPSG:4326')
trails_gdf.to_file('data/infrastructure/trails.geojson', driver='GeoJSON')
```

### 7. Wildlife Data (Wolves, Bears)

**Source:** State wildlife agencies or research datasets

**Wolf Pack Territories:**
- Contact Wyoming Game & Fish Department
- Or use published research datasets
- Format: GeoJSON with polygon territories

**Bear Activity:**
- Similar sources
- Can use point locations or activity zones
- Format: GeoJSON

**Example structure:**
```python
import geopandas as gpd
from shapely.geometry import Point

# Example: Create from point data
wolf_data = {
    'pack_name': ['Pack1', 'Pack2'],
    'geometry': [Point(-107.5, 43.4), Point(-108.0, 44.0)]
}
wolf_gdf = gpd.GeoDataFrame(wolf_data, crs='EPSG:4326')
wolf_gdf.to_file('data/wildlife/wolf_packs.geojson', driver='GeoJSON')
```

## Integration Steps

### Step 1: Create Directory Structure

```bash
mkdir -p data/{dem,terrain,landcover,canopy,hydrology,infrastructure,wildlife}
```

### Step 2: Download and Prepare Core Datasets

Follow the detailed guides for each dataset:

1. **DEM, Slope, and Aspect:** See [Elevation, Slope, and Aspect Integration](./elevation_slope_aspect_integration.md)
   - Download DEM using automated script
   - Generate slope and aspect rasters
   - Verify files are correct

2. **Land Cover:** See [NLCD Land Cover Integration](./nlcd_landcover_integration.md)
   - Download NLCD from USGS EarthExplorer
   - Clip CONUS dataset to Wyoming
   - Verify land cover codes are correct

### Step 3: Verify Data Loading

Test that data loads correctly:

```python
from pathlib import Path
from src.data.processors import DataContextBuilder

# Initialize with data directory
data_dir = Path("data")
builder = DataContextBuilder(data_dir)

# Check what loaded
print(f"DEM loaded: {builder.dem is not None}")
print(f"Slope loaded: {builder.slope is not None}")
print(f"Aspect loaded: {builder.aspect is not None}")
print(f"Land cover loaded: {builder.landcover is not None}")
print(f"Water sources loaded: {builder.water_sources is not None}")
print(f"Roads loaded: {builder.roads is not None}")
```

### Step 4: Integrate Environmental Features

Use the provided script to update your dataset with real environmental features:

```bash
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What the script does:**
1. Loads your dataset
2. For each point, samples environmental data from rasters/vectors
3. Updates environmental columns with real values
4. Replaces placeholder values (e.g., 8500.0 for elevation, 0 for land cover)
5. Saves the updated dataset

**Expected runtime:**
- ~1-2 minutes for 10,000-20,000 points
- Progress saved every 1000 rows

### Step 5: Verify Integration

Analyze the integrated dataset to verify everything worked:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What to look for:**
- ✅ Elevation: Good diversity, realistic range for Wyoming
- ✅ Slope: Good diversity, reasonable range (0-60°)
- ✅ Aspect: Full 0-360° range
- ✅ Land cover: Multiple codes (11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, etc.)
- ⚠️ Water distance: May still be placeholder if water sources data not available

**If you see placeholder elevations:** See the [Handling Placeholder/Inappropriate Elevations](./elevation_slope_aspect_integration.md#handling-placeholderinappropriate-elevations) section in the elevation guide.

## Priority Order

If you can't get all datasets at once, prioritize:

1. **✅ DEM, Slope, Aspect** (highest priority) - **COMPLETED**
   - Needed for elevation, slope, aspect
   - See [Elevation, Slope, and Aspect Integration](./elevation_slope_aspect_integration.md)

2. **✅ Land Cover (NLCD)** (high priority) - **COMPLETED**
   - Important for habitat type classification
   - See [NLCD Land Cover Integration](./nlcd_landcover_integration.md)

3. **⚠️ Water sources** (high priority) - **READY FOR INTEGRATION**
   - Critical for habitat suitability
   - Currently using placeholder values (0.5 miles)
   - See [Water Sources Integration](./water_sources_integration.md) for detailed instructions

4. **⚠️ Roads/Trails** (medium priority) - **NOT YET INTEGRATED**
   - Useful for access patterns
   - Currently using placeholder values
   - See "Roads and Trails" section below

5. **✅ Canopy cover** (medium priority) - **COMPLETED**
   - Important for habitat modeling and security habitat calculations
   - Successfully integrated with real canopy cover percentages
   - See [Canopy Cover Integration](./canopy_cover_integration.md) for details

6. **⚠️ Wildlife data** (optional) - **NOT YET INTEGRATED**
   - Optional, can add later
   - See "Wildlife Data" section below

## Troubleshooting

### Raster files not loading
- Check file paths match expected locations
- Verify file format (should be GeoTIFF)
- Check coordinate system (should be WGS84 or UTM)
- **For DEM issues:** See [Elevation, Slope, and Aspect Integration](./elevation_slope_aspect_integration.md#troubleshooting)
- **For NLCD issues:** See [NLCD Land Cover Integration](./nlcd_landcover_integration.md#troubleshooting)

### Vector files not loading
- Verify GeoJSON format
- Check coordinate system (should be WGS84)
- Ensure geometry is valid

### Placeholder values not being replaced
- **Elevation placeholders:** See [Handling Placeholder/Inappropriate Elevations](./elevation_slope_aspect_integration.md#handling-placeholderinappropriate-elevations)
- **Land cover placeholders:** Check CRS transformation is working (see NLCD guide)
- Verify `DataContextBuilder` is loading files correctly
- Re-run integration script: `python scripts/integrate_environmental_features.py`

### CRS transformation issues
- **Land cover sampling fails:** Ensure `_sample_raster` method handles CRS transformation
- See [NLCD Land Cover Integration](./nlcd_landcover_integration.md#step-4-ensure-crs-transformation-is-fixed) for fix

### Performance issues
- Large rasters can be slow - consider clipping to study area
- Use spatial indexing for vector data
- Consider caching frequently accessed data

### Missing data
- Code handles missing data gracefully with defaults
- Check logs for warnings about missing files
- Verify file permissions

## Next Steps

### Completed ✅
1. ✅ **DEM, Slope, Aspect** - Downloaded, generated, and integrated
2. ✅ **Land Cover (NLCD)** - Downloaded, clipped, and integrated
3. ✅ **Canopy Cover** - Downloaded, clipped, and integrated
4. ✅ **Environmental Feature Integration** - Script created and tested
5. ✅ **Dataset Updated** - Combined dataset has real elevation, slope, aspect, land cover, and canopy cover

### In Progress / Next
1. **Download Water Sources** - Critical for habitat modeling
   - See "Water Sources" section above for download instructions
   - Once downloaded, re-run integration script

2. **Download Roads/Trails** - Useful for access patterns
   - See "Roads and Trails" section above
   - Once downloaded, re-run integration script

3. **Validate Final Dataset** - Ensure all features are realistic
   ```bash
   python scripts/analyze_integrated_features.py \
       data/processed/combined_north_bighorn_presence_absence_cleaned.csv
   ```

4. **Model Training** - Dataset is ready for ML model training with current features

## Resources

- **USGS EarthExplorer:** https://earthexplorer.usgs.gov/
- **NLCD Data:** https://www.mrlc.gov/data
- **NHD Data:** https://www.usgs.gov/national-hydrography/national-hydrography-dataset
- **OpenStreetMap:** https://www.openstreetmap.org/
- **GDAL Documentation:** https://gdal.org/

