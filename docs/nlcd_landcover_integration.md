# NLCD Land Cover Data Integration Guide

This guide documents the successful workflow for downloading NLCD land cover data, clipping it to Wyoming, and integrating it into your processed datasets.

## Overview

The National Land Cover Database (NLCD) provides land cover classification data for the United States. For PathWild, we need NLCD data covering Wyoming to classify habitat types (forest, grassland, shrubland, etc.).

**Key Points:**
- Wyoming is too large (253,348 sq km) for MRLC viewer's 250,000 sq km limit
- Download the complete CONUS dataset instead
- Clip to Wyoming using the provided script
- Ensure CRS transformation is handled correctly

## Step 1: Download NLCD Land Cover Data

### Option A: USGS EarthExplorer (Recommended)

1. **Go to:** https://earthexplorer.usgs.gov/
2. **Login:** Create a free account if needed (top right corner)
3. **Set search area:**
   - Click **"Use Map"** button (in Search Criteria panel)
   - Zoom to Wyoming (use zoom controls or search for "Wyoming")
   - Draw rectangle around Wyoming
   - OR enter coordinates manually:
     - North: 45.0, South: 41.0
     - East: -104.0, West: -111.0
4. **Select dataset:**
   - Click **"Data Sets"** tab (left sidebar)
   - Expand: `Land Cover` → `NLCD` → `NLCD Land Cover`
   - **Important:** Select **"NLCD 2021 Land Cover"** (or latest available)
   - **NOT** "Percent Impervious" or "Percent Tree Canopy"
   - Check the box ✓
5. **Search:**
   - Click **"Results"** button
   - Wait for search to complete
6. **Download:**
   - Select tiles covering Wyoming (check boxes)
   - Click **"Download Options"**
   - Select **"GeoTIFF"** format
   - Click **"Download"**

### Option B: MRLC Direct Download

1. **Go to:** https://www.mrlc.gov/data
2. **Find:** "NLCD Land Cover" product (not "Fractional Impervious")
3. **Download:** Complete CONUS dataset
4. **Save to:** `data/landcover/` directory

**Important:** Make sure you download the **Land Cover** product, not:
- ❌ Fractional Impervious (FctImp)
- ❌ Percent Tree Canopy
- ❌ Percent Developed

The filename should contain "Land_Cover" or "LandCover", not "FctImp" or "Impervious".

## Step 2: Clip to Wyoming

Once you have the downloaded NLCD file (usually a zip file), extract and clip it to Wyoming:

```bash
python scripts/clip_conus_nlcd_to_wyoming.py --zip-file path/to/nlcd_land_cover.zip
```

**Example:**
```bash
# If downloaded to data/landcover/
python scripts/clip_conus_nlcd_to_wyoming.py \
    --zip-file data/landcover/Annual_NLCD_Land_Cover_2021_CU_C1V1.zip
```

**What the script does:**
1. Extracts the `.tif` file from the zip
2. Clips to Wyoming bounds (41°N to 45°N, 111°W to 104°W)
3. Saves to `data/landcover/nlcd.tif`
4. Compresses the output to save space

**Expected output:**
- File: `data/landcover/nlcd.tif`
- Size: ~25-50 MB (clipped from full CONUS)
- Format: GeoTIFF
- CRS: Albers Equal Area (AEA)

## Step 3: Verify the Clipped File

Verify the clipped file is correct:

```bash
python scripts/analyze_nlcd.py data/landcover/nlcd.tif
```

**What to look for:**

✅ **Good signs:**
- Land cover classes: 11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95
- Wyoming classes with realistic percentages:
  - Shrub/Scrub (52): ~50-65%
  - Grassland (71): ~15-25%
  - Evergreen Forest (42): ~10-15%
  - Other classes in expected proportions
- Geographic coverage: "✓ File covers all of Wyoming"
- Resolution: ~30m x 30m pixels

❌ **Warning signs:**
- "Values appear to be percentages (0-100), not land cover codes!" → Wrong product
- All classes show 0% → Wrong area or product
- File name contains "FctImp", "Impervious", or "Canopy" → Wrong product

## Step 4: Ensure CRS Transformation is Fixed

The `DataContextBuilder` needs to handle CRS transformation when sampling raster data. The NLCD file is in Albers Equal Area projection, but coordinates are provided in WGS84 (lat/lon).

**Check:** The `_sample_raster` method in `src/data/processors.py` should handle CRS transformation:

```python
def _sample_raster(self, raster, lon: float, lat: float, 
                  default: float = 0.0) -> float:
    """Sample value from raster at point"""
    if raster is None or not RASTERIO_AVAILABLE:
        return default
    
    try:
        # Handle CRS transformation if needed
        if raster.crs and not raster.crs.is_geographic:
            # Raster is in projected CRS, need to transform lat/lon
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
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
```

If this fix isn't already in place, update `src/data/processors.py` with the corrected `_sample_raster` method.

## Step 5: Integrate Environmental Features

Once the NLCD file is in place and CRS transformation is fixed, integrate environmental features into your dataset:

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
   - Water distance from water sources (if available)
   - Road/trail distances (if available)
3. Updates the dataset with real values (replaces placeholders)
4. Saves the updated dataset

**Expected runtime:**
- ~1-2 minutes for 10,000-20,000 points
- Progress is saved every 1000 rows (configurable with `--batch-size`)

## Step 6: Verify Integration

Analyze the integrated dataset to verify everything worked:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What to look for:**

✅ **Success indicators:**
- **Land cover:** Multiple codes (11, 21, 22, 31, 41, 42, 43, 52, 71, 81, 82, etc.)
- **Land cover distribution:** Realistic for Wyoming:
  - Shrub/Scrub (52): ~50-65%
  - Grassland (71): ~15-25%
  - Forest classes (41, 42, 43): ~10-20% combined
- **Elevation:** Good diversity (thousands of unique values)
- **Slope:** Good diversity, reasonable range (0-60°)
- **Aspect:** Full 0-360° range

❌ **Issues to watch for:**
- Land cover: Only one code (0) → CRS transformation not working
- Land cover: All values 0-100 → Wrong product (percentages, not classes)
- Water distance: Only one value → Water sources data not available (expected)

## Troubleshooting

### "Land cover codes are all 0"

**Cause:** CRS transformation not working

**Fix:** Ensure `_sample_raster` method in `src/data/processors.py` handles CRS transformation (see Step 4).

### "Values appear to be percentages (0-100)"

**Cause:** Downloaded wrong product (Fractional Impervious, not Land Cover)

**Fix:** Download the correct **Land Cover** product from USGS EarthExplorer (see Step 1).

### "File doesn't cover Wyoming"

**Cause:** Clipping failed or wrong area selected

**Fix:** 
1. Verify the input file covers Wyoming
2. Re-run the clip script
3. Check the output with `analyze_nlcd.py`

### "Water distance is still placeholder"

**Cause:** Water sources data (`data/hydrology/water_sources.geojson`) not available

**Status:** This is expected if water sources data hasn't been downloaded yet. It's not critical for model training.

## File Structure

After successful integration, you should have:

```
data/
├── landcover/
│   └── nlcd.tif                    # Clipped NLCD land cover (25-50 MB)
├── dem/
│   └── wyoming_dem.tif             # Digital elevation model
├── terrain/
│   ├── slope.tif                    # Slope in degrees
│   └── aspect.tif                  # Aspect in degrees
├── processed/
│   └── combined_north_bighorn_presence_absence_cleaned.csv  # Updated dataset
└── hydrology/
    └── water_sources.geojson        # Optional: water sources data
```

## Quick Reference

**Complete workflow:**
```bash
# 1. Download NLCD from USGS EarthExplorer (manual)
# 2. Clip to Wyoming
python scripts/clip_conus_nlcd_to_wyoming.py --zip-file path/to/nlcd.zip

# 3. Verify
python scripts/analyze_nlcd.py data/landcover/nlcd.tif

# 4. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 5. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

## Next Steps

After successful integration:
1. ✅ Dataset is ready for model training
2. ✅ All critical environmental features are populated
3. ✅ Land cover codes are realistic for Wyoming
4. Optional: Download water sources data if needed

The dataset is now ready for machine learning model training!

