# NLCD Tree Canopy Cover Data Integration Guide

This guide documents the workflow for downloading NLCD Tree Canopy Cover data, clipping it to Wyoming, and integrating it into your processed datasets.

## Overview

The National Land Cover Database (NLCD) Tree Canopy Cover dataset provides percent tree canopy cover at 30-meter resolution for the conterminous United States. For PathWild, we need this data covering Wyoming to provide canopy cover percentages for habitat modeling.

**Key Points:**
- Wyoming is too large (253,348 sq km) for MRLC viewer's 250,000 sq km limit
- Download the complete CONUS dataset instead
- Clip to Wyoming using the provided script
- Values are percentages (0-100), not categorical codes
- Ensure CRS transformation is handled correctly

**Data Characteristics:**
- **Resolution:** 30 meters
- **Format:** GeoTIFF
- **Values:** Percent tree canopy cover (0-100)
- **CRS:** Albers Equal Area (AEA) projection
- **NoData:** Areas with no tree canopy or outside coverage

## Step 1: Download NLCD Tree Canopy Cover Data

### Option A: MRLC Direct Download (Recommended)

**Note:** Wyoming is too large (253,348 sq km) for the MRLC viewer's 250,000 sq km limit, so you need to download the complete CONUS dataset.

1. **Go to:** https://www.mrlc.gov/data
2. **Find:** "Tree Canopy Cover" product
   - Look for "NLCD Tree Canopy Cover" or "Percent Tree Canopy"
   - May be listed under "Science Products" or "Tree Canopy" section
3. **Select year:** Choose the latest available year (e.g., 2021, 2019)
4. **Download:** Complete CONUS dataset
   - Look for "CONUS" or "Contiguous United States" download option
   - Files are typically named like: `NLCD_2019_TreeCanopy_20210604.zip` or `NLCD_2021_TreeCanopy_*.zip`
5. **Save to:** `data/canopy/` directory (create if needed)

**Direct link (if available):** https://www.mrlc.gov/data/type/tree-canopy

### Option B: MRLC NLCD Viewer (For smaller areas)

If you only need a smaller area (under 250,000 sq km):

1. **Go to:** https://www.mrlc.gov/viewer/
2. **Draw area:** Use the rectangle or polygon tool to select your area of interest
3. **Select product:** Choose "Tree Canopy Cover" and select the year
4. **Enter email:** Provide your email address
5. **Download:** You'll receive an email with download instructions

**Note:** Wyoming exceeds the viewer's size limit, so use Option A (direct download) instead.

### Option C: USGS EarthExplorer (Alternative)

NLCD Tree Canopy Cover may be available through EarthExplorer, but it can be harder to locate:

1. **Go to:** https://earthexplorer.usgs.gov/
2. **Login:** Create a free account if needed
3. **Set search area:**
   - Click **"Use Map"** button
   - Zoom to Wyoming or enter coordinates:
     - North: 45.0, South: 41.0
     - East: -104.0, West: -111.0
4. **Select dataset:**
   - Click **"Data Sets"** tab
   - Navigate to: `Land Cover` → `National Land Cover Database (NLCD)`
   - After searching, use filters to find "Tree Canopy Cover" product
   - Select the desired year (e.g., 2021, 2019)
5. **Download:**
   - Select CONUS dataset tile(s)
   - Click **"Download Options"** → **"GeoTIFF"**
   - Click **"Download"**

**Important:** If you don't see Tree Canopy Cover options in EarthExplorer, use Option A (MRLC Direct Download) instead, which is the primary source for this data.

**Important:** Make sure you download the **Tree Canopy Cover** product, not:
- ❌ NLCD Land Cover (categorical codes, not percentages)
- ❌ Fractional Impervious (FctImp)
- ❌ Percent Developed

The filename should contain "TreeCanopy", "Tree_Canopy", or "Canopy", and the values should be 0-100 (percentages), not categorical codes like 11, 21, 42, etc.

**Recommended approach:** Use **Option A (MRLC Direct Download)** as it's the most straightforward method for downloading the complete CONUS dataset needed for Wyoming.

## Step 2: Create Canopy Directory

Create the directory for canopy data:

```bash
mkdir -p data/canopy
```

Move or copy the downloaded zip file to this directory:

```bash
# If downloaded to Downloads or another location
mv ~/Downloads/NLCD_2021_TreeCanopy_*.zip data/canopy/
```

## Step 3: Clip to Wyoming

Once you have the downloaded NLCD Tree Canopy file (usually a zip file), extract and clip it to Wyoming:

```bash
python scripts/clip_conus_canopy_to_wyoming.py --zip-file path/to/nlcd_tree_canopy.zip
```

**Example:**
```bash
# If downloaded to data/canopy/
python scripts/clip_conus_canopy_to_wyoming.py \
    --zip-file data/canopy/NLCD_2021_TreeCanopy_20210604.zip
```

**What the script does:**
1. Extracts the `.tif` file from the zip
2. Clips to Wyoming bounds (41°N to 45°N, 111°W to 104°W)
3. Saves to `data/canopy/canopy_cover.tif`
4. Compresses the output to save space
5. Handles CRS transformation automatically

**Expected output:**
- File: `data/canopy/canopy_cover.tif`
- Size: ~50-60 MB (clipped from full CONUS)
- Format: GeoTIFF
- CRS: Albers Equal Area (AEA)
- Values: 0-100 (percent tree canopy cover)

## Step 4: Verify the Clipped File

Verify the clipped file is correct:

```bash
python scripts/analyze_canopy.py data/canopy/canopy_cover.tif
```

**What to look for:**

✅ **Good signs:**
- Value range: 0-100 (percentages)
- Mean canopy cover: ~5-15% for Wyoming (relatively sparse tree cover)
- Realistic distribution:
  - Most pixels: 0-20% (grasslands, shrublands, open areas)
  - Some pixels: 20-60% (forests, woodlands)
  - Few pixels: 60-100% (dense forests)
- Geographic coverage: "✓ File covers all of Wyoming"
- Resolution: ~30m x 30m pixels
- NoData handled correctly (areas without trees)

❌ **Warning signs:**
- Values outside 0-100 range → Wrong product or corrupted data
- All values are 0 → Wrong area or product issue
- Values look like categorical codes (11, 21, 42, etc.) → Wrong product (downloaded Land Cover instead of Tree Canopy)
- File name contains "LandCover", "FctImp", or "Impervious" → Wrong product

**Expected output:**
```
File: data/canopy/canopy_cover.tif
File size: 42.3 MB

--- METADATA ---
Dimensions: 14,400 x 12,000 pixels
CRS: EPSG:5070 (Albers Equal Area)
Bounds: [clipped to Wyoming]
Data type: uint8
No data value: None (or 255)

--- VALUE STATISTICS ---
Value range: 0 to 100
Mean: 8.5%
Median: 0%
Percentiles:
  25th: 0%
  50th: 0%
  75th: 5%
  95th: 45%

✓ Values are percentages (0-100) as expected
✓ File covers all of Wyoming
✓ Resolution looks correct for NLCD (30m)
```

## Step 5: Ensure CRS Transformation is Handled

The `DataContextBuilder` needs to handle CRS transformation when sampling raster data. The NLCD Tree Canopy file is in Albers Equal Area projection, but coordinates are provided in WGS84 (lat/lon).

**Check:** The `_sample_raster` method in `src/data/processors.py` should handle CRS transformation automatically (same as for NLCD Land Cover). This was fixed in the NLCD land cover integration.

The method should:
1. Detect if raster is in projected CRS (not geographic)
2. Transform lat/lon coordinates to the raster's CRS
3. Sample the raster value
4. Return the percentage value (0-100)

If you encounter issues with canopy cover sampling, verify the `_sample_raster` method handles CRS transformation correctly (see NLCD Land Cover Integration guide, Step 4).

## Step 6: Integrate Environmental Features

Once the canopy cover file is in place, integrate environmental features into your dataset:

```bash
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What the script does:**
1. Loads your dataset
2. For each point, samples environmental data:
   - Elevation from DEM
   - Slope and aspect from terrain rasters
   - Land cover from NLCD
   - **Canopy cover from NLCD Tree Canopy** (new!)
   - Water distance from water sources (if available)
   - Road/trail distances (if available)
3. Updates the dataset with real values (replaces placeholders)
4. Saves the updated dataset

**Expected runtime:**
- ~1-2 minutes for 10,000-20,000 points
- Progress is saved every 1000 rows (configurable with `--batch-size`)

## Step 7: Verify Integration

Analyze the integrated dataset to verify everything worked:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What to look for:**

✅ **Success indicators:**
- **Canopy cover:** Range 0-100% (percentages)
- **Canopy cover distribution:** Realistic for Wyoming:
  - Most values: 0-20% (open areas, grasslands, shrublands)
  - Some values: 20-60% (forests, woodlands)
  - Few values: 60-100% (dense forests)
  - Mean: ~5-15% (Wyoming has relatively sparse tree cover)
- **Elevation:** Good diversity (thousands of unique values)
- **Slope:** Good diversity, reasonable range (0-60°)
- **Aspect:** Full 0-360° range
- **Land cover:** Multiple codes (11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, etc.)

❌ **Issues to watch for:**
- Canopy cover: Only one value (e.g., 30.0) → Placeholder values not replaced, check file path
- Canopy cover: Values outside 0-100 → Wrong product or data corruption
- Canopy cover: Values look like categorical codes → Wrong product (downloaded Land Cover instead)
- Canopy cover: All values are 0 → Sampling not working or wrong area

**Example good output (from actual integration):**
```
--- CANOPY COVER ANALYSIS ---
Valid values: 16,745 (100.0%)
Range: 0.0% to 78.0%
Mean: 16.67%
Median: 0.00%
  ✓ Good diversity: 78 unique values
  ✓ Values are in expected range (0-100%)
  ✓ Distribution reasonable for Wyoming

  Canopy cover distribution:
    No canopy (0%)           :   8,537 (50.98%)
    Sparse (1-20%)           :   2,890 (17.26%)
    Moderate-High (21-60%)   :   4,651 (27.78%)
    Very Dense (61-100%)     :     667 ( 3.98%)
```

## Troubleshooting

### "Canopy cover values are all the same (30.0)"

**Cause:** Placeholder values not replaced - canopy cover file not found or not loading

**Fix:**
1. Check file exists: `ls -lh data/canopy/canopy_cover.tif`
2. Verify file is readable: `python scripts/analyze_canopy.py data/canopy/canopy_cover.tif`
3. Check `DataContextBuilder` logs for "✓ Canopy cover loaded" message
4. Re-run integration script: `python scripts/integrate_environmental_features.py`

### "Canopy cover values are outside 0-100"

**Cause:** Wrong product downloaded or data corruption

**Fix:**
1. Verify you downloaded "NLCD Tree Canopy Cover" not "NLCD Land Cover"
2. Check file with: `python scripts/analyze_canopy.py data/canopy/canopy_cover.tif`
3. Re-download correct product from MRLC (https://www.mrlc.gov/data)

### "Canopy cover values look like categorical codes (11, 21, 42, etc.)"

**Cause:** Downloaded NLCD Land Cover instead of NLCD Tree Canopy Cover

**Fix:**
1. Download the correct product: "NLCD Tree Canopy Cover" (not "NLCD Land Cover")
2. Check filename contains "TreeCanopy" or "Canopy" (not "LandCover")
3. Re-clip using the correct file

### "File doesn't cover Wyoming"

**Cause:** Clipping failed or wrong area selected

**Fix:**
1. Verify the input file covers Wyoming
2. Re-run the clip script
3. Check the output with `analyze_canopy.py`

### "Canopy cover sampling fails (CRS errors)"

**Cause:** CRS transformation not working

**Fix:**
1. Ensure `_sample_raster` method in `src/data/processors.py` handles CRS transformation (see Step 5)
2. This should already be fixed from NLCD land cover integration
3. If not, update `_sample_raster` method to handle projected CRS

### "Water distance is still placeholder"

**Cause:** Water sources data (`data/hydrology/water_sources.geojson`) not available

**Status:** This is expected if water sources data hasn't been downloaded yet. It's not related to canopy cover.

## File Structure

After successful integration, you should have:

```
data/
├── canopy/
│   └── canopy_cover.tif                    # Clipped NLCD tree canopy (~50-60 MB)
├── landcover/
│   └── nlcd.tif                            # NLCD land cover
├── dem/
│   └── wyoming_dem.tif                     # Digital elevation model
├── terrain/
│   ├── slope.tif                           # Slope in degrees
│   └── aspect.tif                          # Aspect in degrees
├── processed/
│   └── combined_north_bighorn_presence_absence_cleaned.csv  # Updated dataset
└── hydrology/
    └── water_sources.geojson               # Optional: water sources data
```

## Quick Reference

**Complete workflow:**
```bash
# 1. Create directory
mkdir -p data/canopy

# 2. Download NLCD Tree Canopy from MRLC (manual)
#    - Go to: https://www.mrlc.gov/data
#    - Find: "Tree Canopy Cover" or "NLCD Tree Canopy Cover"
#    - Download: CONUS dataset (Wyoming is too large for viewer)
#    - Save zip file to: data/canopy/

# 3. Clip to Wyoming
python scripts/clip_conus_canopy_to_wyoming.py \
    --zip-file data/canopy/NLCD_2021_TreeCanopy_*.zip

# 4. Verify
python scripts/analyze_canopy.py data/canopy/canopy_cover.tif

# 5. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 6. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

## Understanding Canopy Cover Values

NLCD Tree Canopy Cover values represent the **percentage of tree canopy** within each 30m x 30m pixel:

- **0%:** No tree canopy (grassland, shrubland, open areas, water, developed)
- **1-20%:** Sparse tree cover (open woodlands, savannas, scattered trees)
- **21-60%:** Moderate tree cover (forests, woodlands)
- **61-100%:** Dense tree cover (closed-canopy forests)

**For Wyoming:**
- Most areas: 0-20% (grasslands, sagebrush, open areas)
- Forest areas: 20-80% (montane forests, especially in mountains)
- Dense forests: 60-100% (coniferous forests in high elevations)

## Data Sources and Alternatives

### Primary Source: NLCD Tree Canopy Cover (Recommended)

- **Source:** Multi-Resolution Land Characteristics (MRLC) Consortium
- **Website:** https://www.mrlc.gov/data (Direct download) or https://www.mrlc.gov/viewer/ (Interactive viewer)
- **Direct Link:** https://www.mrlc.gov/data/type/tree-canopy
- **Resolution:** 30 meters
- **Coverage:** CONUS
- **Updates:** Updated periodically (2019, 2021, etc.)
- **Pros:** Consistent with NLCD Land Cover, standardized, well-documented, easy to download
- **Cons:** Lower resolution than LiDAR-derived products
- **Note:** MRLC Direct Download is the recommended method for Wyoming (exceeds viewer size limit)

### Alternative: USDA Forest Service Tree Canopy Cover

- **Source:** USDA Forest Service
- **Website:** https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/
- **Resolution:** 30 meters
- **Coverage:** CONUS, Alaska, Hawaii, Puerto Rico
- **Pros:** Similar to NLCD, annual updates
- **Cons:** Slightly different methodology

### Alternative: LiDAR-derived Canopy Cover (Higher Resolution)

- **Resolution:** 1 meter
- **Coverage:** Limited areas (often state-specific)
- **Pros:** Very high resolution, more accurate
- **Cons:** Not available for all areas, larger file sizes, more complex processing

**Recommendation:** Use NLCD Tree Canopy Cover for consistency with NLCD Land Cover and broader coverage.

## Next Steps

After successful integration:
1. ✅ Dataset is ready for model training
2. ✅ Canopy cover feature is populated with real values
3. ✅ Values are realistic percentages (0-100%) for Wyoming
4. ✅ Can be used for habitat modeling and security habitat calculations
5. ✅ Values are automatically clamped to valid range (0-100%) when sampled

## Integration Status

**Status:** ✅ Successfully Integrated

The canopy cover data has been successfully integrated into the PathWild dataset. Integration results:

- **Coverage:** 100% of points have canopy cover data
- **Value Range:** 0.0% to 78.0% (all values within expected 0-100% range)
- **Diversity:** 78 unique values (good diversity, not placeholder)
- **Distribution:** Realistic for Wyoming:
  - 50.98% no canopy (0%)
  - 17.26% sparse (1-20%)
  - 27.78% moderate-high (21-60%)
  - 3.98% very dense (61-100%)
- **Statistics:**
  - Mean: ~16.7% (realistic for Wyoming's sparse tree cover)
  - Median: 0.0% (most areas have no canopy)
- **Data Quality:** Values automatically clamped to 0-100% range during sampling

The dataset now includes canopy cover as an environmental feature for machine learning model training!

