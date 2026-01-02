# Elevation, Slope, and Aspect Data Integration Guide

This guide documents the successful workflow for downloading Digital Elevation Model (DEM) data, generating slope and aspect rasters, and integrating them into your processed datasets.

## Overview

Terrain data is essential for wildlife habitat modeling. This workflow covers:
1. **DEM (Digital Elevation Model)**: Elevation in meters
2. **Slope**: Steepness of terrain in degrees (0° = flat, 90° = vertical)
3. **Aspect**: Direction slope faces in degrees (0° = North, 90° = East, 180° = South, 270° = West)

**Data Source:** USGS 3DEP (3D Elevation Program) 1 arc-second DEM
- Resolution: ~30 meters
- Coverage: Contiguous United States
- Format: GeoTIFF
- Access: AWS S3 public bucket

## Step 1: Download DEM Data

### Automated Download (Recommended)

Use the provided script to download all Wyoming DEM tiles:

```bash
python scripts/download_wyoming_dem.py --mosaic
```

**What the script does:**
1. Downloads 40 DEM tiles covering Wyoming (1°x1° each)
2. Saves tiles to `data/dem/tiles/`
3. Optionally mosaics tiles into single file: `data/dem/wyoming_dem.tif`

**Options:**
- `--output-dir DIR`: Specify output directory (default: `data/dem`)
- `--mosaic`: Automatically mosaic tiles into single file after download
- `--no-mosaic`: Download tiles only (mosaic manually later)

**Expected output:**
- Individual tiles: `data/dem/tiles/USGS_1_nXXwYYY.tif` (40 files, ~50-100 MB each)
- Mosaicked file: `data/dem/wyoming_dem.tif` (~1-3 GB)
- Total download: ~2-4 GB

**Download time:**
- ~10-30 minutes depending on connection speed
- Progress bars show download status for each tile

### Manual Download (Alternative)

If automated download fails, download manually from:
- **USGS National Map:** https://apps.nationalmap.gov/downloader/
- **USGS EarthExplorer:** https://earthexplorer.usgs.gov/

**Search criteria:**
- Dataset: "3D Elevation Program (3DEP)"
- Product: "1 arc-second DEM"
- Area: Wyoming (41°N-45°N, 111°W-104°W)

## Step 2: Verify DEM File

Verify the downloaded DEM is correct:

```bash
python scripts/analyze_dem.py data/dem/wyoming_dem.tif
```

**What to look for:**

✅ **Good signs:**
- **Geographic bounds:** Covers Wyoming (41°N-45°N, 111°W-104°W)
- **Elevation range:** ~3,100-13,800 ft (~945-4,207 m)
  - Minimum: ~3,100-4,000 ft (lowest valleys)
  - Maximum: ~13,000-13,800 ft (highest peaks like Gannett Peak)
  - Mean: ~6,000-7,000 ft (typical Wyoming elevation)
- **Resolution:** ~30 meters (1 arc-second)
- **File size:** ~1-3 GB (reasonable for Wyoming)
- **Data coverage:** >95% valid pixels

❌ **Warning signs:**
- File size <100 MB (might be incomplete)
- Bounds don't cover Wyoming
- Elevation values outside 2,000-15,000 ft range
- Many NoData pixels (>20%)

**Expected output:**
```
File: data/dem/wyoming_dem.tif
File size: 1.9 GB

--- METADATA ---
Dimensions: 14,400 x 12,000 pixels
CRS: EPSG:4326 (WGS84)
Bounds: -111.0° to -104.0° (lon), 41.0° to 45.0° (lat)

--- ELEVATION STATISTICS ---
Range: 945.0 to 4,207.0 meters
Mean: 1,829.0 meters
Median: 1,768.0 meters

--- RESOLUTION ---
Pixel size: ~30m x ~30m
✓ Resolution looks correct for 1 arc-second DEM
```

## Step 3: Generate Slope and Aspect

Generate slope and aspect rasters from the DEM:

```bash
python scripts/generate_slope_aspect.py
```

**What the script does:**
1. Reads the DEM file
2. Calculates slope using Horn's method (similar to GDAL)
3. Calculates aspect (direction slope faces)
4. Saves to `data/terrain/slope.tif` and `data/terrain/aspect.tif`

**Options:**
- `--dem-file PATH`: Specify DEM file (default: `data/dem/wyoming_dem.tif`)
- `--output-dir DIR`: Specify output directory (default: `data/terrain`)

**Expected output:**
```
GENERATING SLOPE AND ASPECT FROM DEM
============================================================

Input DEM: data/dem/wyoming_dem.tif
Output directory: data/terrain

Opening DEM...
  Dimensions: 14,400 x 12,000
  CRS: EPSG:4326
  Pixel size: ~30m x ~30m

Calculating slope...
  Slope range: 0.0° to 65.7°
  Mean slope: 10.2°

Calculating aspect...
  Aspect range: 0° to 360°

Saving slope raster...
  ✓ Saved: data/terrain/slope.tif (45.2 MB)

Saving aspect raster...
  ✓ Saved: data/terrain/aspect.tif (45.1 MB)

✓ Success! Slope and aspect rasters generated.
```

**What to verify:**

✅ **Slope:**
- Range: 0° to ~60-70° (reasonable for Wyoming terrain)
- Mean: ~8-15° (typical for mountainous regions)
- No negative values
- No values >90° (physically impossible)

✅ **Aspect:**
- Range: 0° to 360° (full circle)
- Flat areas should have aspect = 0°

**Important:** The script uses Python/rasterio (no GDAL command-line tools required). The slope calculation correctly divides by pixel size (not multiplies) to get proper slope values.

## Step 4: Verify Slope and Aspect

Check the generated files are correct:

```bash
# Check file sizes
ls -lh data/terrain/*.tif

# Quick verification with Python
python -c "
import rasterio
with rasterio.open('data/terrain/slope.tif') as src:
    data = src.read(1)
    print(f'Slope range: {data.min():.1f}° to {data.max():.1f}°')
    print(f'Mean slope: {data.mean():.1f}°')
"
```

**Expected results:**
- **Slope:** 0° to ~65°, mean ~10°
- **Aspect:** 0° to 360°, full coverage
- **File sizes:** ~40-50 MB each (compressed)

## Step 5: Integrate Environmental Features

Once DEM, slope, and aspect are in place, integrate them into your dataset:

```bash
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What the script does:**
1. Loads your dataset
2. For each point, samples:
   - Elevation from DEM
   - Slope from slope raster
   - Aspect from aspect raster
3. Updates the dataset with real values (replaces placeholders)
4. Saves the updated dataset

**Expected runtime:**
- ~1-2 minutes for 10,000-20,000 points
- Progress saved every 1000 rows

## Step 6: Verify Integration

Analyze the integrated dataset to verify terrain features:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence_cleaned.csv
```

**What to look for:**

✅ **Elevation:**
- Good diversity (thousands of unique values)
- Range: ~1,000-3,700 m (~3,300-12,000 ft) for Wyoming
- Mean: ~1,800-2,200 m (~6,000-7,200 ft)
- No placeholder values (e.g., 8500.0)

✅ **Slope:**
- Good diversity (thousands of unique values)
- Range: 0° to ~60-70°
- Mean: ~8-15°
- No placeholder values (e.g., 15.0 for all points)

✅ **Aspect:**
- Good diversity (thousands of unique values)
- Range: 0° to 360°
- No placeholder values (e.g., 180.0 for all points)

**Example good output:**
```
--- ELEVATION ANALYSIS ---
Valid values: 16,745 (100.0%)
Range: 1004.2 to 3662.8
Mean: 2160.5
  ✓ Good diversity: 11,668 unique values
  ✓ Elevation range looks reasonable for Wyoming

--- SLOPE ANALYSIS ---
Valid values: 16,745 (100.0%)
Range: 0.0° to 65.7°
Mean: 10.2°
  ✓ Good diversity: 11,668 unique values
  ✓ Slope range looks reasonable

--- ASPECT ANALYSIS ---
Valid values: 16,745 (100.0%)
Range: 0° to 360°
Mean: 173°
  ✓ Good diversity: 11,437 unique values
  ✓ Aspect range looks reasonable (0-360°)
```

## Handling Placeholder/Inappropriate Elevations

Sometimes after integration, you may find placeholder or inappropriate elevation values in your dataset. This can happen when:
- Points are outside DEM coverage
- DEM sampling failed (CRS issues, out of bounds)
- Initial data generation used placeholder values

### Step 1: Identify Problematic Elevations

**Signs of placeholder/inappropriate elevations:**
- Many points have the exact same elevation (e.g., 8500.0 m)
- Elevations above 13,800 ft (4,207 m) - Wyoming's highest peak is Gannett Peak at 13,804 ft
- Elevations that don't match the geographic location

**Investigate the issue:**

```bash
python scripts/investigate_high_elevations.py \
    data/processed/your_dataset.csv \
    --threshold-feet 13800
```

This will show:
- How many points have problematic elevations
- Geographic distribution (may be outside Wyoming)
- Presence vs absence balance
- Whether they're exact placeholder values

**Example output:**
```
Investigating elevations above 13,800 feet...
Found 155 points with elevations >= 13,800 feet

Placeholder values detected:
  - 8500.0 m: 155 points

Geographic distribution:
  - Inside Wyoming: 120 points
  - Outside Wyoming: 35 points

Presence vs Absence:
  - Presence: 78 points
  - Absence: 77 points
```

### Step 2: Remove Problematic Points

**Recommended approach:** Remove points with placeholder or inappropriate elevations:

```bash
python scripts/remove_placeholder_elevations.py \
    data/processed/your_dataset.csv \
    --threshold-feet 13800 \
    --output-file data/processed/your_dataset_cleaned.csv \
    --removed-file data/processed/removed_points_for_review.csv
```

**Options:**
- `--threshold-feet N`: Remove elevations above N feet (default: 13800)
- `--placeholder-value N`: Remove exact placeholder value (e.g., 8500.0 for meters)
- `--output-file PATH`: Output file for cleaned dataset
- `--removed-file PATH`: Save removed points for manual review (recommended)

**What the script does:**
1. Identifies points with placeholder or inappropriate elevations
2. Removes them from the dataset
3. Saves cleaned dataset
4. Optionally saves removed points for review

**Example:**
```bash
# Remove elevations above 13,800 feet AND exact placeholder value 8500.0
python scripts/remove_placeholder_elevations.py \
    data/processed/combined_north_bighorn_presence_absence.csv \
    --threshold-feet 13800 \
    --placeholder-value 8500.0 \
    --output-file data/processed/combined_north_bighorn_presence_absence_cleaned.csv \
    --removed-file data/processed/removed_high_elevations.csv
```

**Output:**
- Cleaned dataset with realistic elevations only
- Removed points file for manual review (if needed)

### Step 3: Verify Cleaned Dataset

After removal, verify the cleaned dataset:

```bash
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset_cleaned.csv
```

**Expected results:**
- Elevation range: ~1,000-3,700 m (~3,300-12,000 ft)
- No placeholder values (8500.0)
- No values above 13,800 ft
- Good diversity (thousands of unique values)

### Step 4: Review Removed Points (Optional)

If you saved removed points, review them:

```bash
# Check how many were removed
wc -l data/processed/removed_high_elevations.csv

# View sample of removed points
head -20 data/processed/removed_high_elevations.csv
```

**Decide:**
- If points are outside study area → Removal was correct
- If points are valid but had bad elevations → May need manual fixing or different approach
- If many presence points removed → Check if this affects your analysis

### Alternative: Try to Fix (Not Recommended)

If you want to attempt fixing placeholder elevations by re-sampling from DEM:

```bash
python scripts/fix_placeholder_elevations.py \
    data/processed/your_dataset.csv \
    --placeholder-value 8500.0
```

**Note:** This only works if:
- Points are within DEM coverage
- DEM has valid data at those locations
- CRS transformation works correctly

**If points are outside DEM coverage or sampling fails, removal is the better approach.**

### Prevention

To avoid placeholder elevations in the future:

1. **Check study area bounds:**
   - Ensure all points are within Wyoming bounds
   - Verify DEM covers your entire study area

2. **Verify DEM coverage:**
   ```bash
   python scripts/analyze_dem.py data/dem/wyoming_dem.tif
   ```
   - Check bounds cover your study area
   - Verify no large NoData regions

3. **Ensure CRS transformation works:**
   - Verify `_sample_raster` method handles CRS transformation
   - Check `DataContextBuilder` loads DEM correctly

4. **Re-run integration after DEM updates:**
   - If DEM is updated or extended
   - If study area changes

## Troubleshooting

### "DEM file not found"

**Cause:** Download didn't complete or file in wrong location

**Fix:**
1. Check `data/dem/wyoming_dem.tif` exists
2. Re-run download script
3. Verify file size (~1-3 GB)

### "Slope values are all the same (e.g., 15.0°)"

**Cause:** Slope calculation bug or placeholder values not replaced

**Fix:**
1. Regenerate slope: `python scripts/generate_slope_aspect.py`
2. Verify slope file has diversity: `python -c "import rasterio; import numpy as np; r=rasterio.open('data/terrain/slope.tif'); d=r.read(1); print(f'Unique values: {np.unique(d).size}')"`
3. Re-run integration script

### "Elevation values are placeholders (8500.0)" or "Elevations are too high (>13,800 ft)"

**Cause:** Placeholder values from initial data generation or points outside DEM coverage

**Symptoms:**
- Many points have exact same elevation (e.g., 8500.0 m)
- Elevations above 13,800 ft (4,207 m) - Wyoming's highest peak is Gannett Peak at 13,804 ft
- Points may be outside Wyoming bounds or in areas with no DEM coverage

**Investigation:**

First, investigate the problematic elevations:

```bash
python scripts/investigate_high_elevations.py \
    data/processed/your_dataset.csv \
    --threshold-feet 13800
```

This will show:
- How many points have placeholder/inappropriate elevations
- Geographic distribution of these points
- Whether they're presence or absence points
- If they're outside Wyoming bounds

**Solution: Remove Placeholder Elevations**

Remove points with placeholder or inappropriate elevations:

```bash
python scripts/remove_placeholder_elevations.py \
    data/processed/your_dataset.csv \
    --threshold-feet 13800 \
    --output-file data/processed/your_dataset_cleaned.csv
```

**Options:**
- `--threshold-feet N`: Remove elevations above N feet (default: 13800)
- `--placeholder-value N`: Remove exact placeholder value (e.g., 8500.0)
- `--output-file PATH`: Output file for cleaned dataset
- `--removed-file PATH`: Save removed points for review (optional)

**What the script does:**
1. Identifies points with placeholder or inappropriate elevations
2. Removes them from the dataset
3. Saves cleaned dataset to output file
4. Optionally saves removed points to separate file for review

**Example:**
```bash
# Remove elevations above 13,800 feet
python scripts/remove_placeholder_elevations.py \
    data/processed/combined_north_bighorn_presence_absence.csv \
    --threshold-feet 13800 \
    --output-file data/processed/combined_north_bighorn_presence_absence_cleaned.csv \
    --removed-file data/processed/removed_high_elevations.csv

# This will:
# - Remove ~150-200 points with inappropriate elevations
# - Save cleaned dataset
# - Save removed points for manual review
```

**After removal:**
1. Verify the cleaned dataset:
   ```bash
   python scripts/analyze_integrated_features.py \
       data/processed/your_dataset_cleaned.csv
   ```

2. Check elevation range is now reasonable:
   - Should be ~1,000-3,700 m (~3,300-12,000 ft)
   - No placeholder values (8500.0)
   - No values above 13,800 ft

3. Review removed points (if saved):
   - Check if they're valid points that need manual fixing
   - Or if they're truly outside study area and should be removed

**Alternative: Try to Fix First (Optional)**

If you want to try fixing placeholder elevations by re-sampling from DEM:

```bash
# Note: This may not work if points are outside DEM coverage
python scripts/fix_placeholder_elevations.py \
    data/processed/your_dataset.csv \
    --placeholder-value 8500.0
```

However, if points are outside Wyoming bounds or DEM coverage, removal is the better approach.

**Prevention:**

To avoid placeholder elevations in the future:
1. Ensure DEM covers your entire study area
2. Verify `DataContextBuilder` handles CRS transformation correctly
3. Check for points outside study area bounds before integration
4. Re-run integration script if DEM is updated or extended

### "Slope values are too high (>80°)"

**Cause:** Bug in slope calculation (multiplying instead of dividing by pixel size)

**Fix:** This was fixed in `generate_slope_aspect.py`. If you see this, regenerate slope:
```bash
python scripts/generate_slope_aspect.py
```

### "Aspect values are all 180°"

**Cause:** Placeholder values not replaced or aspect calculation issue

**Fix:**
1. Regenerate aspect: `python scripts/generate_slope_aspect.py`
2. Verify aspect file: Should have 0-360° range
3. Re-run integration script

## File Structure

After successful integration, you should have:

```
data/
├── dem/
│   ├── wyoming_dem.tif              # Mosaicked DEM (~1-3 GB)
│   └── tiles/                       # Individual tiles (optional)
│       ├── USGS_1_n41w111.tif
│       ├── USGS_1_n41w110.tif
│       └── ... (40 tiles total)
├── terrain/
│   ├── slope.tif                    # Slope in degrees (~40-50 MB)
│   └── aspect.tif                   # Aspect in degrees (~40-50 MB)
└── processed/
    └── combined_north_bighorn_presence_absence_cleaned.csv  # Updated dataset
```

## Quick Reference

**Complete workflow:**
```bash
# 1. Download DEM
python scripts/download_wyoming_dem.py --mosaic

# 2. Verify DEM
python scripts/analyze_dem.py data/dem/wyoming_dem.tif

# 3. Generate slope and aspect
python scripts/generate_slope_aspect.py

# 4. Integrate features
python scripts/integrate_environmental_features.py \
    data/processed/your_dataset.csv

# 5. Verify integration
python scripts/analyze_integrated_features.py \
    data/processed/your_dataset.csv
```

## Technical Details

### Slope Calculation

The script uses **Horn's method** (similar to GDAL's slope calculation):

1. Calculate gradients using `np.gradient()` (meters per pixel)
2. Convert to meters per meter by **dividing** by pixel size
3. Calculate slope: `atan(sqrt((dz/dx)² + (dz/dy)²))`
4. Convert to degrees

**Important:** The gradient must be divided by pixel size (not multiplied) to get correct slope values.

### Aspect Calculation

Aspect is calculated as:
1. Calculate gradients (same as slope)
2. Calculate aspect: `atan2(-dz/dx, dz/dy) * 180/π`
3. Convert from -180° to 180° range to 0° to 360° range
4. Set flat areas (slope < 0.01) to aspect = 0°

### CRS Handling

- **DEM:** Usually WGS84 (EPSG:4326) or UTM
- **Slope/Aspect:** Inherit CRS from DEM
- **Sampling:** `DataContextBuilder` handles CRS transformation automatically when sampling raster data

## Next Steps

After successful integration:
1. ✅ Terrain features are ready for model training
2. ✅ Elevation, slope, and aspect are populated with real values
3. ✅ Values are realistic for Wyoming terrain
4. ✅ Ready to combine with other environmental features (land cover, water, etc.)

The dataset is now ready for machine learning model training!

