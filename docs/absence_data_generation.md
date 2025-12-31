# Absence Data Generation for PathWild

## Overview

This document describes the absence data generation system for PathWild's elk location prediction model. Since we only have **presence data** (GPS points where elk were observed), we need to generate **absence data** (locations where elk were NOT present) to train a binary classifier.

## Strategies

The system uses four complementary strategies to generate high-quality absence points:

### 1. Environmental Pseudo-Absences (40%)
- **Purpose**: Represent "available but unused" habitat
- **Criteria**:
  - ≥2,000m from any presence point
  - Elevation: 6,000-13,500 ft (suitable range; elk use high alpine areas in summer)
  - Slope: <45°
  - Water distance: <5 miles
- **Rationale**: These locations are environmentally suitable but elk chose not to use them

### 2. Unsuitable Habitat Absences (30%)
- **Purpose**: High-confidence absences from areas elk physically cannot be
- **Criteria**:
  - Elevation <4,000 ft OR >14,000 ft (very low or extreme high elevations)
  - Slope >60°
  - Urban areas, water bodies, barren land (NLCD codes: 11-12, 21-24, 31)
  - Water distance >10 miles
- **Rationale**: Elk cannot survive in these conditions. Note: Elk use elevations up to 13,500+ ft in summer, so only extreme elevations (>14,000 ft) are considered unsuitable.

### 3. Random Background Points (20%)
- **Purpose**: Represent "available habitat" vs "used habitat"
- **Criteria**:
  - ≥500m from presence points
  - Within study area
  - No other filters
- **Rationale**: Pure random sampling of available habitat

### 4. Temporal Absences (10%)
- **Purpose**: Learn seasonal patterns
- **Criteria**:
  - Same locations as presence points
  - Different time periods (e.g., summer presence → winter absence)
- **Rationale**: Helps model learn that habitat suitability varies by season

## Usage

### Basic Usage

```bash
# Generate absence data for South Bighorn dataset
python scripts/generate_absence_data.py \
    --presence-file data/processed/south_bighorn_points.csv \
    --output-file data/processed/combined_south_bighorn_presence_absence.csv
```

### Advanced Options

```bash
# Custom ratio and skip feature enrichment (faster for testing)
python scripts/generate_absence_data.py \
    --presence-file data/processed/south_bighorn_points.csv \
    --output-file data/processed/combined_south_bighorn_presence_absence.csv \
    --ratio 1.5 \
    --skip-enrichment
```

### Command-Line Arguments

- `--presence-file`: Path to presence data CSV (default: `data/processed/south_bighorn_points.csv`)
- `--output-file`: Path to output CSV (default: `data/processed/combined_south_bighorn_presence_absence.csv`)
- `--data-dir`: Path to data directory (default: `data`)
- `--ratio`: Ratio of absence to presence points (default: 1.0)
- `--skip-enrichment`: Skip environmental feature enrichment (faster for testing)

## Data Requirements

### Required Files

1. **Presence Data CSV**: Must have `latitude` and `longitude` columns
   - Optional: `firstdate`, `lastdate`, or `date` columns for temporal absences

### Optional Files (for better quality)

1. **Wyoming State Boundary**: `data/boundaries/wyoming_state.shp`
   - If not found, uses Wyoming bounding box or Area 048 as fallback

2. **Environmental Data** (for environmental/unsuitable strategies):
   - `data/dem/wyoming_dem.tif` - Digital Elevation Model
   - `data/terrain/slope.tif` - Slope raster
   - `data/landcover/nlcd.tif` - NLCD land cover
   - `data/hydrology/water_sources.geojson` - Water sources

### Downloading Wyoming Boundary

#### Method 1: Automated Download (Recommended)

Use the provided script to automatically download and extract Wyoming boundary:

```bash
python scripts/download_wyoming_boundary.py
```

This will:
- Download the latest US Census Bureau boundary file
- Extract and filter for Wyoming
- Save to `data/boundaries/wyoming_state.shp`

Options:
```bash
# Specify output directory
python scripts/download_wyoming_boundary.py --output-dir data/boundaries

# Use different year (2020, 2021, 2022, 2023)
python scripts/download_wyoming_boundary.py --year 2023

# Use different resolution (500k, 5m, 20m)
python scripts/download_wyoming_boundary.py --resolution 500k
```

#### Method 2: Manual Download

If the automated script doesn't work, download manually:

1. **US Census Bureau** (Recommended):
   - Visit: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
   - Navigate to "Cartographic Boundary Files" → "States"
   - Download: `cb_2023_us_state_500k.zip` (or latest year)
   - Extract the ZIP file
   - Open in QGIS/ArcGIS and filter for Wyoming (NAME='Wyoming' or STUSPS='WY')
   - Export Wyoming only and save to `data/boundaries/wyoming_state.shp`

2. **Alternative Sources**:
   - **Wyoming GIS**: https://gis.wyo.gov/ (may require registration)
   - **Natural Earth**: https://www.naturalearthdata.com/downloads/ (simplified boundaries)

**Note**: The absence generation script will work without this file (uses bounding box fallback), but having the actual boundary improves quality.

## Output Format

The output CSV contains:

### Required Columns
- `latitude`, `longitude`: Point coordinates
- `elk_present`: Binary label (1 = presence, 0 = absence)

### Environmental Features (if enrichment enabled)
- `elevation`: Elevation in meters
- `slope_degrees`: Slope in degrees
- `aspect_degrees`: Aspect in degrees
- `water_distance_miles`: Distance to nearest water source
- `road_distance_miles`: Distance to nearest road
- `land_cover_type`: NLCD land cover description
- `snow_depth_inches`: Snow depth (temporal)
- `temperature_f`: Temperature (temporal)
- `ndvi`: Normalized Difference Vegetation Index (temporal)
- ... (see `DataContextBuilder` for full list)

### Metadata Columns
- `absence_strategy`: Strategy used to generate absence point
  - Values: `environmental`, `unsuitable`, `background`, `temporal`

## Validation

The script automatically validates absence data quality:

1. **Spatial Separation**: Mean distance from absences to presences >1,000m
2. **Geographic Coverage**: Absences cover similar extent as presences
3. **Class Balance**: Presence/absence ratio between 0.5 and 2.0

## Performance

- **Generation Speed**: ~100-1000 points/second (depends on constraints)
- **Feature Enrichment**: ~1-10 seconds per point (depends on API availability)
- **Total Time**: For 4,650 points with enrichment: ~1-2 hours

**Tip**: Use `--skip-enrichment` for fast testing, then enrich separately if needed.

## Troubleshooting

### Issue: "Only generated X/Y absences after N attempts"

**Cause**: Constraints too strict or study area too small

**Solutions**:
- Reduce `min_distance_meters` in generator classes
- Expand study area
- Increase `max_attempts` parameter
- Use different strategy ratios

### Issue: "No boundary file found"

**Solution**: Download Wyoming boundary or use existing hunt area shapefile

### Issue: Environmental data not loading

**Solution**: 
- Check file paths in `data/` directory
- Generators will use defaults if files not found
- Quality may be reduced but generation will continue

## Testing

Run tests with:

```bash
pytest tests/test_absence_generators.py -v
```

## Future Improvements

1. **Parallel Processing**: Generate absences in parallel for faster processing
2. **Caching**: Cache environmental data lookups to avoid redundant API calls
3. **Adaptive Sampling**: Adjust strategy ratios based on data availability
4. **Quality Metrics**: Add more sophisticated validation metrics
5. **Visualization**: Generate maps showing presence/absence distribution

## References

- Elith, J., & Leathwick, J. R. (2009). Species distribution models: ecological explanation and prediction across space and time. *Annual Review of Ecology, Evolution, and Systematics*, 40, 677-697.
- Barbet-Massin, M., et al. (2012). Selecting pseudo-absences for species distribution models: how, where and how many? *Methods in Ecology and Evolution*, 3(2), 327-338.

