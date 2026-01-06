# Automated Data Processing Pipeline

## Overview

The PathWild automated data processing pipeline orchestrates the complete workflow from raw elk GPS collar data to training-ready datasets. The pipeline automates all steps that were previously done manually in Jupyter notebooks.

**⚠️ Prerequisites:** Before running the pipeline, ensure all required environmental data files are present. The pipeline will automatically check prerequisites and fail fast if required files are missing. See [Environmental Data Prerequisites Guide](./environmental_data_prerequisites.md) for detailed instructions on generating all required files (DEM, slope, aspect, land cover, canopy, water sources, etc.).

## Pipeline Steps

The pipeline consists of six main steps:

1. **Process Raw Presence Data** - Converts raw data files (shapefiles, CSVs) into standardized presence points
2. **Generate Absence Data** - Creates balanced absence points using multiple strategies
3. **Integrate Environmental Features** - Adds environmental features (elevation, water, landcover, NDVI, weather, etc.)
4. **Analyze Integrated Features** - Validates feature integration and data quality
5. **Assess Training Readiness** - Evaluates dataset readiness for model training
6. **Prepare Training Features** - Creates training-ready feature datasets by excluding metadata columns

**Note:** The pipeline automatically prepares training features in `data/features/` by excluding metadata columns. Use these feature datasets (not the combined datasets) for model training.

## Quick Start

### Process All Datasets

```bash
# Run complete pipeline for all datasets
python scripts/run_data_pipeline.py
```

### Process Specific Dataset

```bash
# Process only north_bighorn dataset
python scripts/run_data_pipeline.py --dataset north_bighorn
```

### Skip Already-Complete Steps

```bash
# Skip steps that are already done
python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
```

### Force Full Regeneration

```bash
# Force regeneration of all features (even if placeholders don't exist)
python scripts/run_data_pipeline.py --force
```

## Detailed Usage

### Step 1: Process Raw Presence Data

Converts raw elk GPS collar data into standardized presence points CSV files.

**Script:** `scripts/process_raw_presence_data.py`

**Usage:**
```bash
# Process all datasets
python scripts/process_raw_presence_data.py

# Process specific dataset
python scripts/process_raw_presence_data.py --dataset north_bighorn

# Custom input/output directories
python scripts/process_raw_presence_data.py \
    --input-dir data/raw \
    --output-dir data/processed
```

**Input:** Raw data files in `data/raw/elk_*/` directories
- Shapefiles: `*.shp` (migration routes)
- CSV files: `*.csv` (GPS collar points)

**Output:** Presence points CSV files in `data/processed/`
- Format: `{dataset_name}_points.csv`
- Columns: `latitude`, `longitude`, plus original metadata columns

**Supported Datasets:**
- `north_bighorn` - Northern Bighorn herd migration routes
- `southern_bighorn` - Southern Bighorn herd migration routes
- `national_refuge` - National Elk Refuge GPS collar data
- `southern_gye` - Southern Greater Yellowstone Ecosystem GPS data

### Step 2: Generate Absence Data

Generates balanced absence points using multiple strategies and combines with presence data.

**Script:** `scripts/generate_absence_data.py`

**Usage:**
```bash
# Generate absence data for a dataset
python scripts/generate_absence_data.py \
    --presence-file data/processed/north_bighorn_points.csv \
    --output-file data/processed/combined_north_bighorn_presence_absence.csv \
    --data-dir data
```

**Input:** Presence points CSV file

**Output:** Combined presence/absence CSV file
- Format: `combined_{dataset_name}_presence_absence.csv`
- Columns: `latitude`, `longitude`, `elk_present` (1=presence, 0=absence)

**Absence Strategies:**
- **Environmental pseudo-absences (40%)** - Points in environmentally different areas
- **Unsuitable habitat (30%)** - Points in unsuitable habitat types
- **Random background (20%)** - Random points within study area
- **Temporal absences (10%)** - Points at different times/seasons

**Options:**
- `--ratio`: Ratio of absence to presence points (default: 1.0)
- `--n-processes`: Number of parallel processes (default: auto-detect)
- `--skip-enrichment`: Skip environmental feature enrichment (faster for testing)

### Step 3: Integrate Environmental Features

Adds environmental features to presence/absence datasets, including:
- Static features: elevation, slope, aspect, land cover, canopy, water, roads, trails
- Temporal features: NDVI (via AppEEARS), weather/temperature (via PRISM for historical, Open-Meteo for forecasts), snow depth (via SNOTEL)

**Script:** `scripts/integrate_environmental_features.py`

**Usage:**
```bash
# Integrate features (incremental - only processes placeholders)
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv

# Force full regeneration of all features
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv \
    --force

# Test on subset first
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv \
    --limit 100
```

**Input:** Combined presence/absence CSV file

**Output:** Same file with environmental features added (overwrites input)

**Environmental Features Added:**

**Static Features:**
- `elevation` - Elevation in meters (from DEM)
- `slope_degrees` - Slope in degrees (from terrain data)
- `aspect_degrees` - Aspect in degrees (from terrain data)
- `canopy_cover_percent` - Canopy cover percentage (0-100)
- `land_cover_code` - NLCD land cover code
- `land_cover_type` - Land cover type description
- `water_distance_miles` - Distance to nearest water source in miles
- `water_reliability` - Water source reliability (0.0-1.0)
- `road_distance_miles` - Distance to nearest road in miles
- `trail_distance_miles` - Distance to nearest trail in miles
- `security_habitat_percent` - Percentage of security habitat in buffer

**Temporal Features:**
- `snow_depth_inches` - Snow depth (via SNOTEL AWDB API)
- `snow_water_equiv_inches` - Snow water equivalent
- `snow_crust_detected` - Snow crust detection flag
- `temperature_f` - Average daily temperature (°F)
  - Historical: PRISM gridded data (1981-present)
  - Forecasts: Open-Meteo API
- `precip_last_7_days_inches` - 7-day cumulative precipitation
- `cloud_cover_percent` - Cloud cover percentage (if available)
- `ndvi` - Normalized Difference Vegetation Index (0-1)
  - Source: AppEEARS (Landsat NDVI) when credentials available
  - Fallback: Seasonal placeholder values
- `ndvi_age_days` - Days since NDVI data acquisition
- `irg` - Instantaneous Rate of Green-up (rate of change)
- `summer_integrated_ndvi` - Integrated NDVI for June-September

**Note**: NDVI and weather data use real providers when credentials/setup is available, with automatic fallback to placeholders. See `docs/ndvi_weather_integration_status.md` for setup instructions.

**Incremental Processing:**
By default, the script only processes rows with placeholder values, making incremental updates much faster. Use `--force` to regenerate all rows.

**Performance:**
- Auto-detects optimal number of workers based on hardware
- Auto-detects optimal batch size based on dataset size
- Uses parallel processing for large datasets
- Processes ~1000-2000 rows/minute per worker

### Step 4: Analyze Integrated Features

Validates that environmental features were successfully integrated.

**Script:** `scripts/analyze_integrated_features.py`

**Usage:**
```bash
# Analyze a dataset
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv
```

**Output:** Analysis report printed to stdout, including:
- Feature value ranges
- Placeholder value detection
- Missing value counts
- Feature distribution statistics
- Data quality warnings

### Step 5: Assess Training Readiness

Evaluates dataset readiness for model training.

**Script:** `scripts/assess_training_readiness.py`

**Usage:**
```bash
# Assess all datasets
python scripts/assess_training_readiness.py

# Assess specific dataset
python scripts/assess_training_readiness.py \
    data/processed/combined_north_bighorn_presence_absence.csv
```

**Output:** Comprehensive readiness assessment including:
- Data volume assessment
- Feature richness analysis
- Class balance evaluation
- Signal strength analysis
- Data quality checks
- Model training recommendations
- Overall readiness score (0-5)

## Complete Pipeline Example

### Process Single Dataset End-to-End

```bash
# 1. Process raw data
python scripts/process_raw_presence_data.py --dataset north_bighorn

# 2. Generate absence data
python scripts/generate_absence_data.py \
    --presence-file data/processed/north_bighorn_points.csv \
    --output-file data/processed/combined_north_bighorn_presence_absence.csv \
    --data-dir data

# 3. Integrate environmental features
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv

# 4. Analyze features
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv

# 5. Assess readiness
python scripts/assess_training_readiness.py \
    data/processed/combined_north_bighorn_presence_absence.csv
```

### Using the Orchestrator (Recommended)

```bash
# Run complete pipeline for all datasets
python scripts/run_data_pipeline.py

# Run for specific dataset
python scripts/run_data_pipeline.py --dataset north_bighorn

# Skip steps that are already complete
python scripts/run_data_pipeline.py \
    --dataset north_bighorn \
    --skip-steps process_raw,generate_absence

# Force full regeneration
python scripts/run_data_pipeline.py --dataset north_bighorn --force
```

## Pipeline Orchestrator

The `run_data_pipeline.py` script orchestrates all pipeline steps automatically.

### Features

- **Automatic step ordering** - Runs steps in correct sequence
- **Dependency checking** - Verifies inputs exist before running steps
- **Skip completed steps** - Detects already-complete steps and skips them
- **Progress tracking** - Shows consistent progress reporting across all steps with tqdm progress bars and batch completion messages
- **Error handling** - Stops pipeline on failure with clear error messages
- **Force mode** - Option to force regeneration of all features
- **Logging** - All pipeline output logged to timestamped files in `logs/` directory
- **Retry logic** - Automatic retry with exponential backoff for transient API errors (5xx)
- **Warning suppression** - Duplicate warnings suppressed to reduce log noise

### Command-Line Options

```bash
python scripts/run_data_pipeline.py [OPTIONS]

Options:
  --data-dir PATH      Base data directory (default: data)
  --dataset NAME       Specific dataset to process (e.g., "north_bighorn")
  --skip-steps LIST    Comma-separated steps to skip (e.g., "process_raw,generate_absence")
  --force              Force regeneration of all features
```

### Step Names for --skip-steps

- `process_raw` - Process raw presence data
- `generate_absence` - Generate absence data
- `integrate_features` - Integrate environmental features
- `analyze_features` - Analyze integrated features
- `assess_readiness` - Assess training readiness

## Data Requirements

### Required Environmental Data

The pipeline requires the following environmental data files:

**Terrain Data:**
- `data/dem/wyoming_dem.tif` - Digital elevation model
- `data/terrain/slope.tif` - Slope raster
- `data/terrain/aspect.tif` - Aspect raster

**Land Cover:**
- `data/landcover/wyoming_nlcd.tif` - NLCD land cover raster

**Canopy Cover:**
- `data/canopy/canopy_cover.tif` - Canopy cover percentage raster

**Water Sources:**
- `data/hydrology/water_sources.geojson` - Water sources vector data

**Infrastructure:**
- `data/infrastructure/roads.geojson` - Roads vector data
- `data/infrastructure/trails.geojson` - Trails vector data

**Boundaries:**
- `data/boundaries/wyoming_state.shp` - Wyoming state boundary (optional, uses bounding box if missing)

### Data Directory Structure

```
data/
├── raw/                          # Raw input data
│   ├── elk_north_bighorn/       # Northern Bighorn data
│   ├── elk_southern_bighorn/    # Southern Bighorn data
│   ├── elk_national_refuge/     # National Refuge data
│   └── elk_southern_gye/        # Southern GYE data
├── processed/                    # Processed datasets
│   ├── {dataset}_points.csv    # Presence points
│   └── combined_{dataset}_presence_absence.csv  # Final datasets
├── dem/                         # Elevation data
├── terrain/                     # Slope/aspect data
├── landcover/                   # Land cover data
├── canopy/                      # Canopy cover data
├── hydrology/                   # Water sources
├── infrastructure/              # Roads/trails
└── boundaries/                  # Study area boundaries
```

## Performance Considerations

### Processing Times (Approximate)

For a dataset with ~50,000 presence points:

- **Process raw data:** < 1 minute
- **Generate absence data:** 5-15 minutes (depends on strategies used)
- **Integrate features:** 30-60 minutes (depends on hardware, ~1000-2000 rows/min/worker)
- **Analyze features:** < 1 minute
- **Assess readiness:** < 1 minute

**Total pipeline time:** ~40-80 minutes per dataset

### Optimization Tips

1. **Use incremental feature integration** - Only processes rows with placeholders by default
2. **Parallel processing** - Auto-detects optimal worker count
3. **Skip completed steps** - Use `--skip-steps` to avoid re-running completed steps
4. **Test with --limit** - Test feature integration on a subset first

## Troubleshooting

### Common Issues

**Issue: "No datasets found"**
- **Solution:** Ensure raw data files exist in `data/raw/elk_*/` directories

**Issue: "Required input not found"**
- **Solution:** Run pipeline steps in order, or use `--skip-steps` to skip completed steps

**Issue: "Environmental data not found"**
- **Solution:** Ensure all required environmental data files exist (see Data Requirements)

**Issue: "Placeholders not replaced"**
- **Solution:** Check that environmental data files are correct and accessible. Use `--force` to regenerate.

**Issue: "Pipeline step failed"**
- **Solution:** Check the error message for the specific step. Common issues:
  - Missing input files
  - Incorrect file formats
  - Insufficient disk space
  - Memory issues (try reducing `--workers`)

### Debugging

Enable verbose logging:
```bash
# Set logging level
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_data_pipeline.py --dataset north_bighorn
```

Run individual steps manually:
```bash
# Test each step individually
python scripts/process_raw_presence_data.py --dataset north_bighorn
python scripts/generate_absence_data.py --presence-file data/processed/north_bighorn_points.csv ...
```

## Testing

### Run Unit Tests

```bash
# Run all pipeline tests
pytest tests/test_data_pipeline.py -v

# Run integration tests
pytest tests/test_pipeline_integration.py -v

# Run all tests
pytest tests/ -v
```

### Test with Small Dataset

```bash
# Create test dataset
mkdir -p data/raw/elk_test_dataset
# Add test CSV file with a few points

# Run pipeline on test dataset
python scripts/run_data_pipeline.py --dataset test_dataset
```

## Best Practices

1. **Always test on a subset first** - Use `--limit` when integrating features
2. **Backup before running** - The pipeline overwrites files
3. **Run steps incrementally** - Process one dataset at a time initially
4. **Monitor disk space** - Processed datasets can be large (50-100 MB each)
5. **Check logs** - Review output for warnings and errors
6. **Validate outputs** - Use analysis scripts to verify data quality

## Preparing Training Features

**Note:** The pipeline automatically prepares training features as Step 6. This section explains what happens and how to customize it if needed.

The combined datasets contain both environmental features and dataset-specific metadata columns. The pipeline automatically creates training-ready feature datasets by excluding metadata columns that could cause data leakage or add noise.

### Why Exclude Metadata?

Dataset-specific metadata columns (identifiers, source info, etc.) should be excluded because they:
- **Cause data leakage**: Columns like `absence_strategy` reveal how data was generated
- **Add noise**: Identifiers like `route_id` or `Elk_ID` don't help predict elk presence
- **Reduce generalizability**: Model should learn from environmental features, not dataset-specific info
- **Waste resources**: Unnecessary columns increase model complexity

### Automatic Feature Preparation

The pipeline automatically runs `scripts/prepare_training_features.py` as Step 6, creating feature datasets in `data/features/`. You don't need to run it manually unless you want to customize the process.

### Manual Feature Preparation (Optional)

If you need to customize which columns are excluded or regenerate features:

```bash
# Prepare features for a single dataset
python scripts/prepare_training_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv \
    data/features/north_bighorn_features.csv

# Prepare features for all datasets
python scripts/prepare_training_features.py --all-datasets

# Exclude temporal columns (year, month) to prevent potential data leakage
python scripts/prepare_training_features.py --all-datasets --exclude-temporal
```

### Excluded Columns

The script automatically excludes:
- **Identifiers**: `route_id`, `id`, `Elk_ID`, `elk_id`
- **Area-specific metadata**: `distance_to_area_048_km`, `inside_area_048`
- **Source temporal metadata**: `mig`, `firstdate`, `lastdate`, `season`
- **UTM coordinates**: `UTM_X`, `UTM_Y`, `utm_easting`, `utm_northing` (we have lat/lon)
- **Absence generation metadata**: `absence_strategy` (prevents data leakage)
- **Dataset-specific columns**: `feedground`, `date`, `day_of_year`

### Included Features

The script keeps all environmental and biological features:
- **Location**: `latitude`, `longitude`
- **Terrain**: `elevation`, `slope_degrees`, `aspect_degrees`
- **Habitat**: `canopy_cover_percent`, `land_cover_code`, `land_cover_type`
- **Water**: `water_distance_miles`, `water_reliability`
- **Infrastructure**: `road_distance_miles`, `trail_distance_miles`
- **Security**: `security_habitat_percent`
- **Predators**: `wolves_per_1000_elk`, `bear_activity_distance_miles`
- **Weather**: `temperature_f`, `precip_last_7_days_inches`, `snow_depth_inches`
- **Vegetation**: `ndvi`, `irg`, `summer_integrated_ndvi`
- **Biological**: `pregnancy_rate`
- **Target**: `elk_present`

### Temporal Columns (Optional)

By default, `year` and `month` are included as they can capture seasonal patterns. However, if you suspect they might encode dataset-specific information (data leakage), use `--exclude-temporal` to remove them.

## Next Steps

After running the pipeline:

1. **Prepare training features** - Use `scripts/prepare_training_features.py` to exclude metadata
2. **Review assessment reports** - Check training readiness scores
3. **Address data quality issues** - Fix any placeholder values or missing data
4. **Combine datasets** - Merge multiple datasets if needed
5. **Proceed to model training** - Use feature datasets (not combined datasets) for ML training

## Related Documentation

- [Environmental Data Integration](./environmental_data_integration.md) - Details on environmental features
- [Water Sources Integration](./water_sources_integration.md) - Water data integration workflow
- [Absence Data Generation](./absence_data_generation.md) - Absence generation strategies

