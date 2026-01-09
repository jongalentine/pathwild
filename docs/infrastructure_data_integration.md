# Infrastructure Data Integration Guide

This guide provides detailed steps for acquiring and integrating roads and trails data into the PathWild project.

## Overview

Two infrastructure datasets are required:
1. **Roads** - TIGER/Line roads from US Census Bureau
2. **Trails** - USGS National Map trails or OpenStreetMap trails

Both datasets will be processed and saved as GeoJSON files in `data/infrastructure/`.

---

## Prerequisites

1. **Wyoming boundary shapefile** must exist:
   ```bash
   # If not present, download it:
   python scripts/download_wyoming_boundary.py
   ```
   Expected location: `data/boundaries/wyoming_state.shp`

2. **Required Python packages** (already in environment.yml):
   - geopandas
   - pandas
   - requests
   - shapely

3. **Activate conda environment**:
   ```bash
   conda activate pathwild
   ```

---

## Temporal Alignment Consideration

**Important:** Your training data spans 2006-2019, but your model will predict future locations (targeting October 2026). 

**Recommendation:** Use **current (2023) roads/trails data** for the following reasons:

1. **Future predictions require current infrastructure** - The model will be used to predict where elk will be in 2026+, so current roads/trails are more relevant than historical ones.

2. **Infrastructure changes slowly** - Most roads/trails that existed in 2007-2019 still exist today. The road network is relatively stable over 5-10 year periods.

3. **Distance is a robust feature** - The signal (distance to nearest road/trail) is relatively insensitive to minor network changes. A location that was 2 miles from a road in 2015 is likely still ~2 miles from a road today.

4. **Historical data is harder to obtain** - TIGER/Line historical data requires manual archive access, and may not be available for all years.

**If you want to test both approaches:**
- Download 2023 data for production use (recommended)
- Optionally download 2015 data to test if it improves training fit
- Compare model performance metrics (AUC, accuracy) with both datasets
- If historical data significantly improves training fit, you could use it for training but still use current data for inference

**Bottom line:** For a model predicting future locations, current infrastructure data is the right choice. The temporal mismatch is minimal and outweighed by the benefit of using current data for predictions.

---

## Part 1: Download and Process Roads (TIGER/Line)

### Step 1: Download TIGER/Line Roads

The script automatically downloads TIGER/Line roads for Wyoming from the US Census Bureau.

**Quick Start (Recommended - Current Data):**
```bash
python scripts/download_tiger_roads.py
```

This will:
- Download TIGER/Line 2023 roads for Wyoming (FIPS code 56)
- Filter to relevant road types (primary, secondary, tertiary, unclassified)
- Clip to Wyoming boundary
- Convert to GeoJSON format
- Save to `data/infrastructure/roads.geojson`

**Alternative - Historical Data (for testing only):**
```bash
# Download 2015 data (closer to training period)
python scripts/download_tiger_roads.py --year 2015 --output data/infrastructure/roads_2015.geojson
```

**Note:** Historical TIGER/Line data may not be available for all years. If 2015 is unavailable, try 2010 or 2020.

**Expected output:**
- File size: ~50-100 MB (depending on simplification)
- Processing time: 2-5 minutes
- Road segments: ~100,000-200,000 segments

### Step 2: Verify Roads Data

```bash
# Quick verification
python -c "import geopandas as gpd; gdf = gpd.read_file('data/infrastructure/roads.geojson'); print(f'Loaded {len(gdf):,} roads')"

# Detailed inspection
python -c "
import geopandas as gpd
gdf = gpd.read_file('data/infrastructure/roads.geojson')
print(f'Total roads: {len(gdf):,}')
print(f'CRS: {gdf.crs}')
print(f'Columns: {list(gdf.columns)}')
if 'road_type' in gdf.columns:
    print('\nRoad type distribution:')
    print(gdf['road_type'].value_counts())
"
```

### Step 3: Advanced Options

**Use different year:**
```bash
python scripts/download_tiger_roads.py --year 2022
```

**Simplify geometries (reduces file size):**
```bash
python scripts/download_tiger_roads.py --simplify
```

**Custom road types:**
```bash
# Only primary and secondary roads
python scripts/download_tiger_roads.py --road-types S1100,S1200,S1400
```

**Skip clipping (not recommended):**
```bash
python scripts/download_tiger_roads.py --no-clip
```

### Manual Download (Alternative)

If the automated download fails:

1. Visit: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
2. Navigate to: **TIGER/Line Shapefiles** > **2023** (or desired year) > **Roads**
3. Download: `tl_2023_56_roads.zip` (56 is Wyoming's FIPS code)
4. Extract the ZIP file
5. Process with:
   ```bash
   # You would need to modify the script to accept a local file
   # Or use geopandas directly:
   python -c "
   import geopandas as gpd
   from pathlib import Path
   
   # Load and process
   roads = gpd.read_file('path/to/tl_2023_56_roads.shp')
   # ... (follow processing steps from script)
   roads.to_file('data/infrastructure/roads.geojson', driver='GeoJSON')
   "
   ```

---

## Part 2: Download and Process Trails

Trails data can be acquired via two methods:

### Method 1: USGS National Map (Recommended for Official Trails)

#### Step 1: Manual Download from USGS

1. **Visit National Map Downloader:**
   - URL: https://apps.nationalmap.gov/downloader/

2. **Select Area:**
   - Click "Select Area" or use "Define Area by Coordinates"
   - Enter Wyoming bounds:
     - West: -111.0
     - South: 41.0
     - East: -104.0
     - North: 45.0
   - Or use the map to draw a box around Wyoming

3. **Select Data Products:**
   - Expand "Structures" or "Transportation"
   - Check "Trails" or "National Trails System"
   - Click "Search Products"

4. **Download:**
   - Select the trails dataset(s) for Wyoming
   - Click "Download" for each product
   - Save ZIP files to `data/raw/infrastructure/trails/`

5. **Extract ZIP files:**
   ```bash
   mkdir -p data/raw/infrastructure/trails
   cd data/raw/infrastructure/trails
   unzip *.zip
   ```

#### Step 2: Process Trails Shapefile

```bash
# Find the shapefile
find data/raw/infrastructure/trails -name "*.shp" | head -1

# Process it (replace with actual path)
python scripts/download_usgs_trails.py \
    --input data/raw/infrastructure/trails/trails.shp
```

### Method 2: OpenStreetMap (Alternative - More Complete Coverage)

OpenStreetMap often has more comprehensive trail coverage, including unofficial trails.

**Quick Start:**
```bash
python scripts/download_usgs_trails.py --osm
```

This will:
- Query OpenStreetMap Overpass API for trails in Wyoming
- Download trails tagged as: `highway=path`, `highway=footway`, `highway=bridleway`, `route=hiking`, `route=horse`
- Clip to Wyoming boundary
- Convert to GeoJSON format
- Save to `data/infrastructure/trails.geojson`

**Expected output:**
- Processing time: 5-15 minutes (depends on API response time)
- Trail segments: Varies (OSM has more complete coverage)
- File size: ~10-50 MB

**Note:** OSM data may include unofficial trails and user-contributed data. Use with awareness of data quality.

### Step 3: Verify Trails Data

```bash
# Quick verification
python -c "import geopandas as gpd; gdf = gpd.read_file('data/infrastructure/trails.geojson'); print(f'Loaded {len(gdf):,} trails')"

# Detailed inspection
python -c "
import geopandas as gpd
gdf = gpd.read_file('data/infrastructure/trails.geojson')
print(f'Total trails: {len(gdf):,}')
print(f'CRS: {gdf.crs}')
print(f'Columns: {list(gdf.columns)}')
"
```

### Advanced Options

**Simplify geometries:**
```bash
python scripts/download_usgs_trails.py --osm --simplify
```

**Skip clipping (not recommended):**
```bash
python scripts/download_usgs_trails.py --osm --no-clip
```

---

## Part 3: Integration Testing

### Step 1: Verify Files Load in DataContextBuilder

The `DataContextBuilder` class automatically loads roads and trails if they exist. Test this:

```python
from pathlib import Path
from src.data.processors import DataContextBuilder

# Initialize builder
builder = DataContextBuilder(Path("data"))

# Check if roads/trails loaded
print(f"Roads loaded: {builder.roads is not None}")
print(f"Trails loaded: {builder.trails is not None}")

if builder.roads is not None:
    print(f"  Roads: {len(builder.roads):,} segments")
if builder.trails is not None:
    print(f"  Trails: {len(builder.trails):,} segments")
```

### Step 2: Test Context Building

Test that road/trail distances are calculated correctly:

```python
from pathlib import Path
from src.data.processors import DataContextBuilder
from datetime import datetime

# Initialize builder
builder = DataContextBuilder(Path("data"))

# Test location (example: near a road in Wyoming)
test_location = {"lat": 41.8350, "lon": -106.4250}  # Area 048 center
test_date = "2023-10-15"

# Build context
context = builder.build_context(test_location, test_date)

# Check road/trail distances
print(f"Road distance: {context.get('road_distance_miles', 'N/A')} miles")
print(f"Trail distance: {context.get('trail_distance_miles', 'N/A')} miles")
```

### Step 3: Test with Sample Data

Run the integration script on a small sample:

```bash
# Test with limited rows
python scripts/integrate_environmental_features.py \
    data/processed/combined_southern_bighorn_presence_absence.csv \
    --limit 10 \
    --output data/features/test_with_roads_trails.csv
```

Check the output CSV for `road_distance_miles` and `trail_distance_miles` columns.

### Step 4: Validate Distance Calculations

Verify that distances are reasonable:

```python
import pandas as pd

# Load test output
df = pd.read_csv('data/features/test_with_roads_trails.csv')

# Check road distances
print("Road distance statistics:")
print(df['road_distance_miles'].describe())

# Check trail distances
print("\nTrail distance statistics:")
print(df['trail_distance_miles'].describe())

# Verify reasonable ranges (0-50 miles typical)
assert df['road_distance_miles'].min() >= 0
assert df['road_distance_miles'].max() < 100  # Should be within Wyoming
assert df['trail_distance_miles'].min() >= 0
assert df['trail_distance_miles'].max() < 100
```

---

## Part 4: Full Pipeline Integration

### Step 1: Update Data Pipeline

The `run_data_pipeline.py` script should automatically detect roads/trails if they exist. Verify:

```bash
# Check prerequisites
python scripts/run_data_pipeline.py --check-prerequisites
```

### Step 2: Run Full Pipeline

```bash
# Process a dataset with roads/trails integrated
python scripts/run_data_pipeline.py --dataset southern_bighorn
```

### Step 3: Verify in Feature Files

Check that road/trail distances are included in feature files:

```python
import pandas as pd

# Load feature file
df = pd.read_csv('data/features/southern_bighorn_features.csv')

# Verify columns exist
assert 'road_distance_miles' in df.columns
assert 'trail_distance_miles' in df.columns

# Check for missing values
print(f"Road distance missing: {df['road_distance_miles'].isna().sum()}")
print(f"Trail distance missing: {df['trail_distance_miles'].isna().sum()}")
```

---

## Troubleshooting

### Issue: Roads/Trails Not Loading

**Symptoms:** `DataContextBuilder` shows `roads = None` or `trails = None`

**Solutions:**
1. Verify files exist:
   ```bash
   ls -lh data/infrastructure/roads.geojson
   ls -lh data/infrastructure/trails.geojson
   ```

2. Check file format:
   ```bash
   python -c "import geopandas as gpd; gdf = gpd.read_file('data/infrastructure/roads.geojson'); print('Valid GeoJSON')"
   ```

3. Verify CRS is WGS84 (EPSG:4326):
   ```python
   import geopandas as gpd
   gdf = gpd.read_file('data/infrastructure/roads.geojson')
   print(f"CRS: {gdf.crs}")  # Should be EPSG:4326
   ```

### Issue: Distance Calculations Return Default Values

**Symptoms:** All locations show same road/trail distance (e.g., 2.0 miles for roads)

**Solutions:**
1. Check that geometries are valid:
   ```python
   import geopandas as gpd
   gdf = gpd.read_file('data/infrastructure/roads.geojson')
   print(f"Invalid geometries: {~gdf.geometry.is_valid}.sum()")
   ```

2. Verify spatial indexing:
   ```python
   # GeoPandas should build spatial index automatically
   # But you can force it:
   gdf.sindex  # This will build the index
   ```

3. Test distance calculation manually:
   ```python
   from shapely.geometry import Point
   import geopandas as gpd
   
   roads = gpd.read_file('data/infrastructure/roads.geojson')
   point = Point(-106.4250, 41.8350)  # Wyoming location
   
   # Calculate distance
   roads_utm = roads.to_crs('EPSG:32612')
   point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
   point_utm = point_gdf.to_crs('EPSG:32612')
   
   distances = roads_utm.geometry.distance(point_utm.geometry.iloc[0])
   nearest_distance_m = distances.min()
   nearest_distance_mi = nearest_distance_m / 1609.34
   print(f"Nearest road: {nearest_distance_mi:.2f} miles")
   ```

### Issue: File Size Too Large

**Symptoms:** GeoJSON files are very large (>500 MB)

**Solutions:**
1. Simplify geometries:
   ```bash
   python scripts/download_tiger_roads.py --simplify
   python scripts/download_usgs_trails.py --osm --simplify
   ```

2. Filter to essential road types:
   ```bash
   # Only primary and secondary roads
   python scripts/download_tiger_roads.py --road-types S1100,S1200,S1400
   ```

3. Use shapefile format instead (smaller, but requires more processing):
   ```python
   # Save as shapefile instead of GeoJSON
   roads.to_file('data/infrastructure/roads.shp')
   ```

### Issue: OSM Download Times Out

**Symptoms:** OpenStreetMap API request fails or times out

**Solutions:**
1. Try a different Overpass API server:
   ```python
   # Edit download_usgs_trails.py
   # Change: overpass_url = "https://overpass-api.de/api/interpreter"
   # To: overpass_url = "https://overpass.kumi.systems/api/interpreter"
   ```

2. Split the query into smaller bounding boxes (modify script)

3. Use manual USGS download instead (Method 1)

---

## Data Maintenance

### Updating Roads Data

TIGER/Line roads are updated annually. To update:

```bash
# Download latest year
python scripts/download_tiger_roads.py --year 2024  # When available
```

### Temporal Alignment Testing (Optional)

If you want to test whether historical roads/trails data improves model training:

1. **Download historical data:**
   ```bash
   # Download 2015 roads (closer to training period)
   python scripts/download_tiger_roads.py --year 2015 --output data/infrastructure/roads_2015.geojson
   ```

2. **Temporarily use historical data:**
   ```bash
   # Backup current data
   mv data/infrastructure/roads.geojson data/infrastructure/roads_2023.geojson
   # Use historical data
   cp data/infrastructure/roads_2015.geojson data/infrastructure/roads.geojson
   ```

3. **Train model and compare metrics:**
   - Train with historical roads
   - Compare training metrics (AUC, accuracy, F1) with current roads
   - If improvement is significant (>2-3%), consider using historical for training
   - **But still use current data for inference/predictions**

4. **Restore current data:**
   ```bash
   mv data/infrastructure/roads_2023.geojson data/infrastructure/roads.geojson
   ```

**Expected result:** Historical data is unlikely to significantly improve training metrics because:
- Road networks change slowly
- Distance to nearest road is a robust feature
- Most infrastructure from 2015 still exists today

### Updating Trails Data

Trails data changes less frequently, but you may want to update:

- **USGS:** Re-download from National Map Downloader
- **OSM:** Re-run `--osm` option (OSM data is continuously updated)

### Backup

Before updating, backup existing files:

```bash
cp data/infrastructure/roads.geojson data/infrastructure/roads.geojson.backup
cp data/infrastructure/trails.geojson data/infrastructure/trails.geojson.backup
```

---

## Next Steps

After successfully integrating roads and trails:

1. ✅ Verify data loads in `DataContextBuilder`
2. ✅ Test distance calculations with sample locations
3. ✅ Run full pipeline on a dataset
4. ✅ Validate feature files include road/trail distances
5. ✅ Test heuristics that depend on roads/trails:
   - `HuntingPressureHeuristic` (access.py)
   - `SecurityHabitatHeuristic` (security.py)

---

## References

- **TIGER/Line Documentation:** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- **USGS National Map Downloader:** https://apps.nationalmap.gov/downloader/
- **OpenStreetMap Overpass API:** https://wiki.openstreetmap.org/wiki/Overpass_API
- **GeoPandas Documentation:** https://geopandas.org/

---

## Summary Checklist

- [ ] Wyoming boundary shapefile exists (`data/boundaries/wyoming_state.shp`)
- [ ] Downloaded and processed TIGER/Line roads (`data/infrastructure/roads.geojson`)
- [ ] Downloaded and processed trails (`data/infrastructure/trails.geojson`)
- [ ] Verified files load in `DataContextBuilder`
- [ ] Tested distance calculations with sample locations
- [ ] Ran integration script on sample data
- [ ] Validated feature files include road/trail distances
- [ ] Tested heuristics that depend on infrastructure data

