# SNOTEL Data Integration via snotelr R Package

This guide explains how to set up and use the `snotelr` R package for accessing real SNOTEL data in PathWild.

## Why snotelr?

The USDA AWDB REST API format is not well documented, making direct API calls unreliable. The `snotelr` R package provides a proven, maintained solution for accessing SNOTEL data.

## Quick Start

1. **Install dependencies** (R, snotelr, rpy2)
2. **Download station metadata**: `python scripts/download_snotel_stations_manual.py` (if station file doesn't exist)
3. **Map station IDs**: `python scripts/map_snotel_station_ids.py` (maps USDA triplets to snotelr site IDs)
4. **Test integration**: Run `pytest tests/test_snotel_integration.py -v`

## Prerequisites

1. **R** (version 4.0 or higher)
2. **rpy2** (Python-R interface)
3. **snotelr** R package

## Installation

### Option 1: Using Conda (Recommended)

Conda can install R, rpy2, and snotelr together:

```bash
# Update your environment
conda env update -f environment.yml

# Or install manually
conda install -c conda-forge r-base r-snotelr rpy2
```

### Option 2: Manual Installation

#### Step 1: Install R

**macOS:**
```bash
brew install r
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install r-base
```

**Or download from:** https://cran.r-project.org/

#### Step 2: Install snotelr R Package

```bash
R -e "install.packages('snotelr', repos='https://cloud.r-project.org')"
```

Or from R console:
```r
install.packages("snotelr")
```

#### Step 3: Install rpy2 in Python

```bash
pip install rpy2
```

Or via conda:
```bash
conda install -c conda-forge rpy2
```

## Setup Steps

### Step 1: Download Station Metadata

First, create the station GeoJSON file:

```bash
python scripts/download_snotel_stations_manual.py
```

This creates `data/cache/snotel_stations_wyoming.geojson` with USDA station triplets.

### Step 2: Map Station IDs

Map USDA station triplets to snotelr site IDs:

```bash
python scripts/map_snotel_station_ids.py
```

This script:
- Uses `snotelr::snotel_info()` to get all SNOTEL stations
- Matches stations by name and location
- Adds `snotelr_site_id` column to the station GeoJSON file

**Important:** This mapping step is required before SNOTEL data can be retrieved.

### Step 3: Verify Setup

Test that everything is set up correctly using the comprehensive test suite:

```bash
pytest tests/test_snotel_integration.py -v
```

Or test R/snotelr directly:

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Try to load snotelr
snotelr = importr('snotelr')
print("✓ snotelr loaded successfully!")
```

## How It Works

The `SNOTELClient` uses `rpy2` to call R functions from Python:

1. **Find nearest station** - Uses the station GeoJSON file with mapped `snotelr_site_id`
2. **Download data** - Calls `snotelr::snotel_download()` with the mapped site_id
3. **Extract values** - Filters to the requested date and extracts snow depth/SWE
4. **Convert units** - Converts from mm (R output) to inches (PathWild format)
5. **Cache results** - Caches in memory to avoid repeated R calls

## Station ID Mapping

**Important:** snotelr uses its own site ID system that doesn't match USDA station triplets. The mapping script (`map_snotel_station_ids.py`) is required to:

- Match USDA station triplets (e.g., "SNOTEL:WY:967") to snotelr site IDs
- Add `snotelr_site_id` column to the station GeoJSON file
- Enable correct data retrieval for each station

Without this mapping, the code will fall back to elevation-based estimates.

## Data Format

`snotelr::snotel_download()` returns a data frame with columns:
- `date` - Date
- `snow_water_equivalent` - SWE in mm
- `snow_depth` - Snow depth in mm (if available)
- `precipitation` - Precipitation in mm
- `temperature_max` - Maximum temperature
- `temperature_min` - Minimum temperature
- And more...

The implementation converts mm to inches for consistency with PathWild's units.

## Troubleshooting

### "Could not load snotelr R package"

**Solution:** Install snotelr in R:
```bash
R -e "install.packages('snotelr', repos='https://cloud.r-project.org')"
```

### "rpy2 not available"

**Solution:** Install rpy2:
```bash
pip install rpy2
# Or
conda install -c conda-forge rpy2
```

### "R not found"

**Solution:** Make sure R is installed and in your PATH:
```bash
which R
R --version
```

### "No data available for station"

**Possible causes:**
- Station may not have data for that date
- Station may be inactive
- Date may be too recent (data may have delay)

**Solution:** The implementation falls back to elevation-based estimates when no data is available.

## Performance Notes

- R initialization has a small overhead (~1-2 seconds on first call)
- Subsequent calls are faster due to caching
- Consider batch processing multiple dates to reduce R startup overhead

## Alternative: Pre-download Data

For better performance, you could pre-download SNOTEL data for all Wyoming stations and cache it locally, then query the cached data instead of calling R each time.

## References

- snotelr package: https://bluegreen-labs.github.io/snotelr/
- rpy2 documentation: https://rpy2.github.io/
- SNOTEL data: https://wcc.sc.egov.usda.gov/snow/

