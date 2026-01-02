# Environmental Data Prerequisites Guide

This guide provides detailed instructions for generating all required environmental data files needed by the PathWild data processing pipeline. These instructions are designed to be adaptable for other states and geographies.

## Overview

Before running the data pipeline (`scripts/run_data_pipeline.py`), you must generate the following environmental data files:

### Required Files (Pipeline will fail if missing)

**Raster Files (TIF):**
- `data/dem/{state}_dem.tif` - Digital Elevation Model
- `data/terrain/slope.tif` - Slope in degrees (derived from DEM)
- `data/terrain/aspect.tif` - Aspect in degrees (derived from DEM)
- `data/landcover/nlcd.tif` - NLCD Land Cover classification
- `data/canopy/canopy_cover.tif` - Tree canopy cover percentage

**Vector Files (GeoJSON):**
- `data/hydrology/water_sources.geojson` - Water sources (streams, lakes, springs)

### Optional Files (Will use defaults if missing)

**Vector Files (GeoJSON):**
- `data/infrastructure/roads.geojson` - Road network
- `data/infrastructure/trails.geojson` - Trail network
- `data/wildlife/wolf_packs.geojson` - Wolf pack territories
- `data/wildlife/bear_activity.geojson` - Bear activity centers
- `data/hunt_areas/hunt_areas.geojson` - Hunting area boundaries

## Directory Structure

Create the following directory structure under `data/`:

```
data/
├── dem/                    # Digital Elevation Model (raster)
│   └── {state}_dem.tif     # e.g., wyoming_dem.tif
├── terrain/                # Derived terrain rasters
│   ├── slope.tif          # Slope in degrees
│   └── aspect.tif         # Aspect in degrees
├── landcover/              # Land cover classification
│   └── nlcd.tif           # NLCD land cover codes
├── canopy/                 # Vegetation canopy cover
│   └── canopy_cover.tif   # Canopy cover percentage (0-100)
├── hydrology/              # Water sources (vector)
│   └── water_sources.geojson
├── infrastructure/         # Roads and trails (vector)
│   ├── roads.geojson
│   └── trails.geojson
├── wildlife/              # Predator data (vector)
│   ├── wolf_packs.geojson
│   └── bear_activity.geojson
└── boundaries/             # State/region boundaries (for clipping)
    └── {state}_boundary.shp
```

## Step-by-Step Instructions

### 1. Digital Elevation Model (DEM)

**Source:** USGS 3D Elevation Program (3DEP) 1 arc-second

**For Wyoming:**
```bash
# Download DEM tiles and create mosaic
python scripts/download_wyoming_dem.py --mosaic
```

**For Other States/Geographies:**

1. **Determine tile coverage:**
   - 1 arc-second DEM tiles are 1°x1° and named by northwest corner (nXXwYYY)
   - Calculate which tiles cover your area of interest
   - Example: Wyoming covers tiles from n41w111 to n45w104

2. **Download tiles:**
   ```bash
   # Modify scripts/download_wyoming_dem.py for your state
   # Update TILES list with your coverage
   # Update output filename (e.g., montana_dem.tif)
   python scripts/download_wyoming_dem.py --mosaic
   ```

3. **Alternative: Download from USGS National Map:**
   - Go to: https://apps.nationalmap.gov/downloader/
   - Select: "Elevation Products (3DEP)"
   - Choose: "1 arc-second (30m)"
   - Draw bounding box for your area
   - Download and extract to `data/dem/`
   - Mosaic tiles if needed using GDAL:
     ```bash
     gdal_merge.py -o data/dem/{state}_dem.tif data/dem/tiles/*.tif
     ```

**Verification:**
```bash
python scripts/analyze_dem.py data/dem/{state}_dem.tif
```

**Expected Output:**
- File size: ~1-3 GB (depends on area size)
- CRS: Geographic (WGS84) or UTM
- Resolution: ~30 meters
- Elevation range: Should match your geography

### 2. Slope and Aspect (Derived from DEM)

**Source:** Generated from DEM using Python script

**For Wyoming:**
```bash
python scripts/generate_slope_aspect.py
```

**For Other States/Geographies:**

1. **Update script parameters:**
   ```bash
   python scripts/generate_slope_aspect.py \
       --dem-file data/dem/{state}_dem.tif \
       --output-dir data/terrain
   ```

2. **The script will:**
   - Read DEM file
   - Calculate slope in degrees (0-90°)
   - Calculate aspect in degrees (0-360°)
   - Save to `data/terrain/slope.tif` and `data/terrain/aspect.tif`

**Verification:**
- Slope: Should be 0-90 degrees
- Aspect: Should be 0-360 degrees
- File sizes: ~40-50 MB each (for Wyoming-sized area)

### 3. Land Cover (NLCD)

**Source:** National Land Cover Database (NLCD)

**For Wyoming:**
```bash
# Download CONUS NLCD data
# Go to: https://www.mrlc.gov/data
# Download: NLCD Land Cover (most recent year)
# Save to: data/landcover/

# Clip to Wyoming boundary
python scripts/clip_conus_nlcd_to_wyoming.py
```

**For Other States/Geographies:**

1. **Download NLCD data:**
   - Go to: https://www.mrlc.gov/data
   - Download: "NLCD Land Cover" (most recent year, typically CONUS)
   - Save zip file to `data/landcover/`

2. **Create state boundary shapefile:**
   ```bash
   # Download state boundary from US Census TIGER/Line
   # Or use existing boundary shapefile
   # Place in data/boundaries/{state}_boundary.shp
   ```

3. **Clip NLCD to your area:**
   ```bash
   # Modify scripts/clip_conus_nlcd_to_wyoming.py
   # Update paths for your state:
   python scripts/clip_conus_nlcd_to_wyoming.py \
       --input data/landcover/nlcd_{year}_land_cover_conus.tif \
       --boundary data/boundaries/{state}_boundary.shp \
       --output data/landcover/nlcd.tif
   ```

4. **Alternative: Use GDAL directly:**
   ```bash
   gdalwarp -cutline data/boundaries/{state}_boundary.shp \
            -crop_to_cutline \
            -dstnodata 0 \
            data/landcover/nlcd_{year}_land_cover_conus.tif \
            data/landcover/nlcd.tif
   ```

**Verification:**
- File size: ~100-500 MB (depends on area)
- CRS: Should match DEM (typically Albers Equal Area)
- Values: NLCD land cover codes (11-95)

### 4. Canopy Cover

**Source:** NLCD Tree Canopy Cover

**For Wyoming:**
```bash
# Download CONUS canopy cover data
# Go to: https://www.mrlc.gov/data
# Download: NLCD Tree Canopy Cover (most recent year)
# Save to: data/canopy/

# Clip to Wyoming boundary
python scripts/clip_conus_canopy_to_wyoming.py
```

**For Other States/Geographies:**

1. **Download canopy cover data:**
   - Go to: https://www.mrlc.gov/data
   - Download: "NLCD Tree Canopy Cover" (most recent year, typically CONUS)
   - Save zip file to `data/canopy/`

2. **Clip to your area:**
   ```bash
   # Modify scripts/clip_conus_canopy_to_wyoming.py
   # Update paths for your state:
   python scripts/clip_conus_canopy_to_wyoming.py \
       --input data/canopy/nlcd_tcc_{year}_CONUS.tif \
       --boundary data/boundaries/{state}_boundary.shp \
       --output data/canopy/canopy_cover.tif
   ```

3. **Alternative: Use GDAL directly:**
   ```bash
   gdalwarp -cutline data/boundaries/{state}_boundary.shp \
            -crop_to_cutline \
            -dstnodata 0 \
            data/canopy/nlcd_tcc_{year}_CONUS.tif \
            data/canopy/canopy_cover.tif
   ```

**Verification:**
- File size: ~100-500 MB (depends on area)
- Values: 0-100 (percentage canopy cover)
- CRS: Should match DEM

### 5. Water Sources

**Source:** National Hydrography Dataset (NHD) High Resolution

**For Wyoming:**
See detailed guide: [Water Sources Integration](./water_sources_integration.md)

**Quick Summary:**
```bash
# 1. Generate URLs for northern regions (if applicable)
python scripts/generate_nhd_urls.py --output data/raw/nhd/nhd_download_urls.txt

# 2. Validate URLs
python scripts/validate_nhd_urls.py \
    --urls-file data/raw/nhd/nhd_download_urls.txt \
    --output data/raw/nhd/nhd_download_urls_validated.txt

# 3. Download files
python scripts/download_nhd_water_sources.py \
    --urls-file data/raw/nhd/nhd_download_urls_validated.txt \
    --output-dir data/raw/nhd

# 4. Extract shapefiles
python scripts/extract_all_nhd_shapefiles.py

# 5. Process to GeoJSON
python scripts/process_nhd_water_sources.py \
    --input-dir data/raw/nhd/shapefiles \
    --output data/hydrology/water_sources.geojson
```

**For Other States/Geographies:**

1. **Determine HUC8 coverage:**
   - NHD data is organized by Hydrologic Unit Code (HUC8) regions
   - Find which HUC8 regions cover your area:
     - Use USGS HUC Finder: https://water.usgs.gov/GIS/huc_name.html
     - Or use National Map Downloader to identify regions

2. **Download NHD data:**
   - **Option A: Manual download via National Map Downloader**
     - Go to: https://apps.nationalmap.gov/downloader/
     - Search for: "NHD High Resolution"
     - Draw bounding box for your area
     - Select all HU8 Shape files
     - Copy download URLs to `data/raw/nhd/{state}_urls.txt`
   
   - **Option B: Generate URLs programmatically**
     - Modify `scripts/generate_nhd_urls.py` for your HUC4 regions
     - Or use USGS TNM Access API (may require authentication)

3. **Download and process:**
   ```bash
   # Download files
   python scripts/download_nhd_water_sources.py \
       --urls-file data/raw/nhd/{state}_urls.txt \
       --output-dir data/raw/nhd
   
   # Extract shapefiles
   python scripts/extract_all_nhd_shapefiles.py
   
   # Process to GeoJSON
   python scripts/process_nhd_water_sources.py \
       --input-dir data/raw/nhd/shapefiles \
       --output data/hydrology/water_sources.geojson \
       --clip-boundary data/boundaries/{state}_boundary.shp  # Optional: clip to state
   ```

**Verification:**
```bash
python scripts/analyze_water_sources.py data/hydrology/water_sources.geojson
python scripts/verify_wyoming_coverage.py data/hydrology/water_sources.geojson  # Update for your state
```

### 6. Roads and Trails (Optional)

**Source:** OpenStreetMap (OSM) or state/local GIS data

**For Wyoming:**
- Currently using placeholder values
- To add real data:
  1. Download OSM data from Geofabrik: https://download.geofabrik.de/
  2. Extract roads and trails using OSM tools
  3. Convert to GeoJSON

**For Other States/Geographies:**

1. **Download OSM data:**
   ```bash
   # Download from Geofabrik
   wget https://download.geofabrik.de/north-america/us/{state}-latest.osm.pbf
   
   # Extract roads and trails using osmium or osmconvert
   osmium tags-filter {state}-latest.osm.pbf \
       nwr/highway=motorway,trunk,primary,secondary,tertiary,unclassified,residential \
       -o roads.osm.pbf
   
   osmium tags-filter {state}-latest.osm.pbf \
       nwr/highway=path,footway,bridleway,cycleway \
       -o trails.osm.pbf
   ```

2. **Convert to GeoJSON:**
   ```bash
   # Use ogr2ogr to convert OSM to GeoJSON
   ogr2ogr -f GeoJSON data/infrastructure/roads.geojson roads.osm.pbf lines
   ogr2ogr -f GeoJSON data/infrastructure/trails.geojson trails.osm.pbf lines
   ```

3. **Alternative: Use state/local GIS data**
   - Many states provide road/trail shapefiles
   - Download from state GIS portal
   - Convert to GeoJSON using ogr2ogr or QGIS

### 7. Wildlife Data (Optional)

**Source:** State wildlife agencies, research institutions, or OpenStreetMap

**For Wyoming:**
- Currently using placeholder values
- To add real data:
  1. Contact Wyoming Game and Fish Department
  2. Check research publications for wolf pack territories
  3. Use OpenStreetMap for bear activity areas

**For Other States/Geographies:**

1. **Wolf pack territories:**
   - Contact state wildlife agency
   - Check research publications (e.g., USGS, university research)
   - May need to digitize from maps or publications

2. **Bear activity:**
   - Similar sources as wolf packs
   - May use OSM data for bear activity areas
   - Or use habitat suitability models

3. **Convert to GeoJSON:**
   ```bash
   # If you have shapefiles:
   ogr2ogr -f GeoJSON data/wildlife/wolf_packs.geojson wolf_packs.shp
   ogr2ogr -f GeoJSON data/wildlife/bear_activity.geojson bear_activity.shp
   ```

## Adapting for Other States/Geographies

### Key Considerations

1. **Coordinate Reference System (CRS):**
   - Wyoming uses UTM Zone 12N (EPSG:32612) for distance calculations
   - For other states, determine appropriate UTM zone:
     - UTM zones: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
     - Update `src/data/processors.py` if needed (currently hardcoded to EPSG:32612)

2. **File Naming:**
   - Update filenames from `wyoming_*` to `{state}_*`
   - Update scripts to use state-specific names

3. **Boundary Files:**
   - Download state/region boundary from:
     - US Census TIGER/Line: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
     - Or use existing boundary shapefile

4. **Tile Coverage:**
   - DEM tiles: Calculate 1°x1° tile coverage for your area
   - NHD HUC8 regions: Use USGS HUC Finder to identify regions

5. **Data Sources:**
   - Some data sources are US-specific (NLCD, NHD, 3DEP)
   - For international use, find equivalent datasets:
     - DEM: SRTM, ASTER GDEM, or national elevation datasets
     - Land cover: National land cover datasets or MODIS
     - Water: National hydrography datasets or OSM

### State-Specific Modifications

1. **Update `scripts/download_wyoming_dem.py`:**
   - Change `WYOMING_TILES` to your state's tile coverage
   - Update output filename from `wyoming_dem.tif` to `{state}_dem.tif`

2. **Update `src/data/processors.py`:**
   - Change `wyoming_dem.tif` to `{state}_dem.tif` (or make configurable)
   - Update UTM zone if needed (currently EPSG:32612)

3. **Update clipping scripts:**
   - `scripts/clip_conus_nlcd_to_wyoming.py` → `scripts/clip_conus_nlcd_to_{state}.py`
   - Update boundary file path

4. **Update NHD scripts:**
   - Modify HUC4/HUC8 region lists for your state
   - Update coverage verification scripts

## Verification Checklist

Before running the pipeline, verify all required files exist:

```bash
# Check required files
python -c "
from pathlib import Path
data_dir = Path('data')

required = {
    'DEM': data_dir / 'dem' / 'wyoming_dem.tif',
    'Slope': data_dir / 'terrain' / 'slope.tif',
    'Aspect': data_dir / 'terrain' / 'aspect.tif',
    'Land Cover': data_dir / 'landcover' / 'nlcd.tif',
    'Canopy': data_dir / 'canopy' / 'canopy_cover.tif',
    'Water Sources': data_dir / 'hydrology' / 'water_sources.geojson',
}

print('Required Files:')
for name, path in required.items():
    status = '✓' if path.exists() else '✗'
    print(f'{status} {name}: {path}')
"
```

## Troubleshooting

### Common Issues

1. **"File not found" errors:**
   - Check file paths match expected locations
   - Verify filenames match (case-sensitive on Linux/Mac)
   - Ensure files are not corrupted

2. **CRS mismatch warnings:**
   - All rasters should use the same CRS (or be transformable)
   - Vector files should be in WGS84 (EPSG:4326) or transformable
   - Use `gdalinfo` to check CRS of raster files

3. **File size issues:**
   - Large files may take time to process
   - Ensure sufficient disk space (10-20 GB recommended)
   - Consider using cloud storage for large datasets

4. **Memory errors:**
   - Large rasters may require significant RAM
   - Consider processing in chunks or using cloud computing

## Additional Resources

- **USGS National Map:** https://apps.nationalmap.gov/downloader/
- **NLCD Data:** https://www.mrlc.gov/data
- **NHD Data:** https://www.usgs.gov/national-hydrography/national-hydrography-dataset
- **3DEP Elevation:** https://www.usgs.gov/3d-elevation-program
- **OpenStreetMap:** https://www.openstreetmap.org/
- **Geofabrik Downloads:** https://download.geofabrik.de/

## Quick Reference: Complete Workflow

For Wyoming (already configured):
```bash
# 1. DEM
python scripts/download_wyoming_dem.py --mosaic

# 2. Slope & Aspect
python scripts/generate_slope_aspect.py

# 3. Land Cover (download manually, then clip)
python scripts/clip_conus_nlcd_to_wyoming.py

# 4. Canopy Cover (download manually, then clip)
python scripts/clip_conus_canopy_to_wyoming.py

# 5. Water Sources
python scripts/generate_nhd_urls.py --output data/raw/nhd/nhd_download_urls.txt
python scripts/validate_nhd_urls.py --urls-file data/raw/nhd/nhd_download_urls.txt --output data/raw/nhd/nhd_download_urls_validated.txt
python scripts/download_nhd_water_sources.py --urls-file data/raw/nhd/nhd_download_urls_validated.txt --output-dir data/raw/nhd
python scripts/extract_all_nhd_shapefiles.py
python scripts/process_nhd_water_sources.py --input-dir data/raw/nhd/shapefiles --output data/hydrology/water_sources.geojson

# 6. Verify prerequisites
python scripts/run_data_pipeline.py  # Will check prerequisites automatically
```

For other states, follow the detailed instructions above, adapting file paths and scripts as needed.

