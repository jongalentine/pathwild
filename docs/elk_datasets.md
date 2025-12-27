# Public Elk Population Datasets for PathWild

## Overview
This document lists publicly available elk datasets that can be used to train and validate the PathWild prediction model. All datasets are free and publicly accessible.

## Primary Data Sources

### 1. USGS Science Data Catalog
**URL:** https://data.usgs.gov/datacatalog/

The USGS provides extensive GPS collar telemetry data for elk in Wyoming and surrounding states.

#### Wyoming-Specific Datasets

**Elk GPS Collar Data - National Elk Refuge (2006-2015)**
- **Location:** National Elk Refuge, Jackson, Wyoming
- **Coverage:** 17 adult female elk, 2006-2015
- **Data:** GPS locations, timestamps, migration patterns
- **Use Case:** Spring migration patterns, winter range behavior
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:5a9f2782e4b0b1c392e502ea

**Elk GPS Collar Data - Southern Greater Yellowstone Ecosystem (2007-2015)**
- **Location:** 22 Wyoming winter supplemental feedgrounds
- **Coverage:** 288 adult and yearling female elk
- **Data:** GPS locations during brucellosis risk period (February-July)
- **Use Case:** Seasonal movement patterns, disease risk modeling
- **Link:** https://catalog.data.gov/dataset/elk-gps-collar-data-in-southern-gye-2007-2015

**Migration Routes - Fossil Buttes Population (2005-2010)**
- **Location:** Fossil Butte herd, Wyoming
- **Coverage:** 72 elk, 207 migration sequences
- **Data:** GPS locations every 2-8 hours
- **Use Case:** Migration route prediction, seasonal habitat use
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:5f8db61082ce32418791d56b

**Migration Routes - Wiggins Fork Herd (2015-2018)**
- **Location:** Wiggins Fork population, Wyoming
- **Coverage:** 16 elk, 80 migration sequences
- **Data:** GPS locations every 2 hours
- **Use Case:** High-frequency movement patterns, migration timing
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:620e4abdd34e6c7e83baa368

**Migration Routes - Jackson Herd**
- **Location:** Jackson Hole, Wyoming
- **Coverage:** Winter range to summer range migrations
- **Data:** Average 39 miles one-way, some up to 168 miles
- **Use Case:** Long-distance migration modeling
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:5f8db61982ce32418791d56d

**Migration Routes - South Wind River Herd**
- **Location:** Wind River Range, Wyoming
- **Coverage:** Low-elevation winter to high-elevation summer ranges
- **Data:** Altitudinal migration patterns
- **Use Case:** Elevation-based habitat selection
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:5f8db62782ce32418791d572

**Migration Routes - South Bighorn Herd** ⭐ **BEST FOR area 048 AREA**
- **Location:** Bighorn Mountains, Wyoming (Area 048 area!)
- **Coverage:** Western foothills to mountainous regions
- **Data:** Spring/fall altitudinal migrations
- **Winter Range:** Western foothills of southern Bighorn Mountains, just east of Route 434 (Upper Nowood Road) - **Very close to Area 048!**
- **Migration Distance:** Average 24 miles one-way (10-62 miles range)
- **Population:** ~4,000 elk
- **Use Case:** Seasonal elevation preferences, October hunting predictions
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:620e4ab3d34e6c7e83baa362

**Migration Routes - North Bighorn Herd**
- **Location:** Northern Bighorn Mountains, Wyoming
- **Coverage:** Eastern foothills to mountainous regions
- **Winter Range:** Eastern foothills, just west of Sheridan
- **Migration Distance:** Average 21 miles one-way (5-83 miles range)
- **Population:** ~5,500 elk
- **Use Case:** Altitudinal migration patterns, northern Bighorn area
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:620e4ab0d34e6c7e83baa360

**Remotely Sensed Elk Locations - National Elk Refuge (2017-2019)**
- **Location:** National Elk Refuge, Jackson, Wyoming
- **Coverage:** Winter 2018, Winter/Spring 2019
- **Data:** Satellite imagery, UAS imagery, GPS collar data
- **Use Case:** Population density estimation, habitat use validation
- **Link:** https://data.usgs.gov/datacatalog/data/USGS:61533df9d34e0df5fb9c5c6c

### 2. Wyoming Game and Fish Department (WGFD)

**Harvest Data & Surveys**
- **Source:** Annual harvest reports and hunter surveys
- **Data:** Harvest statistics, success rates, hunt unit data
- **Access:** 
  - Website: https://wgfd.wyo.gov/hunting-trapping/harvest-reports-surveys
  - Email: wgf.inforequest@wyo.gov
- **Use Case:** Validation data, population estimates, seasonal activity patterns
- **Note:** Historical data available upon request

**Wyoming Survey & Analysis Center (WYSAC)**
- **Contact:** wyhunter@uwyo.edu or 1-866-966-2715
- **Data:** Detailed harvest survey data, hunter success statistics
- **Use Case:** Ground truth validation for predictions

### 3. Movebank (Wildlife Tracking Data)

**Platform:** https://www.movebank.org/
- **Description:** Global repository for animal movement data
- **Access:** Free registration required
- **Data:** GPS telemetry from multiple research projects
- **Use Case:** Additional elk tracking data, cross-validation
- **Note:** Search for "elk" and filter by location (Wyoming)

### 4. Data.gov

**Catalog:** https://catalog.data.gov/
- **Search Terms:** "elk", "Wyoming", "GPS", "telemetry"
- **Multiple datasets** from USGS, USFWS, and other agencies
- **Use Case:** Comprehensive dataset discovery

## Data Format & Structure

Most GPS collar datasets include:
- **Unique elk identifier**
- **Timestamp** (date/time)
- **GPS coordinates** (latitude/longitude, often also UTM)
- **Capture location** (if applicable)
- **Metadata** (age, sex, capture date, etc.)

## Recommended Datasets for PathWild

### For Area 048, Wyoming / Bighorn Mountains (Specific Location)
1. **South Bighorn Herd Migration Routes** - ⭐ **BEST FOR YOUR AREA** - Same geographic region, altitudinal migrations, October patterns
2. **North Bighorn Herd Migration Routes** - Also in Bighorn Mountains, different migration pattern
3. **Elk GPS Collar Data - Southern GYE (2007-2015)** - Large sample size for general patterns
4. **National Elk Refuge Data (2006-2015)** - Long time series, well-documented

### For Training (General/High Priority)
1. **Elk GPS Collar Data - Southern GYE (2007-2015)** - Largest sample size (288 elk)
2. **National Elk Refuge Data (2006-2015)** - Long time series, well-documented
3. **Wiggins Fork Herd (2015-2018)** - Recent data, high-frequency sampling
4. **South Bighorn Herd** - ⭐ **Essential for Bighorn Mountains predictions**

### For Validation
1. **Wyoming Game and Fish Harvest Data** - Real-world validation
2. **Remotely Sensed Elk Locations (2017-2019)** - Independent validation source

### For Feature Engineering
1. **Migration Route datasets** - Understand seasonal patterns
2. **Multiple herd datasets** - Generalize across different regions

## Integration with PathWild

### Data Processing Pipeline
1. **Download** GPS collar data from USGS
2. **Extract** location, timestamp, and metadata
3. **Join** with environmental data:
   - Elevation (DEM)
   - Weather (NOAA)
   - Snow depth (SNOTEL)
   - Vegetation (NDVI)
4. **Create** training features matching your heuristic system
5. **Label** positive/negative examples based on GPS locations

### Feature Alignment
Your existing heuristics align well with available data:
- **Elevation** - GPS coordinates → DEM lookup
- **Snow conditions** - Timestamp → SNOTEL data
- **Water distance** - GPS → Hydrology layer
- **Seasonal patterns** - Timestamp → Month/season
- **Migration status** - GPS sequence → Movement patterns

## Data Access Workflow

### Step 1: Download from USGS
```python
# Example: Access USGS data catalog via API or direct download
# Most datasets available as CSV or shapefile
```

### Step 2: Load into PathWild
```python
# Add to src/data/processors.py
def load_elk_gps_data(file_path: str) -> pd.DataFrame:
    """Load GPS collar data from USGS dataset"""
    # Implementation
```

### Step 3: Create Training Dataset
```python
# Match GPS locations with environmental context
# Create labeled examples for ML model
```

## Additional Resources

### Research Papers
- USGS publications on elk migration
- Wyoming Game and Fish research reports
- Peer-reviewed journals on elk behavior

### APIs & Real-time Data
- **SNOTEL API** - Snow depth (already in your stack)
- **NOAA Weather API** - Weather data (already in your stack)
- **USGS Water Services** - Streamflow data

## Next Steps

1. **Download** 2-3 primary datasets (start with National Elk Refuge + Southern GYE)
2. **Explore** data structure and quality
3. **Integrate** with existing `DataContextBuilder`
4. **Create** training pipeline in `src/data/processors.py`
5. **Validate** against WGFD harvest data

## Contact Information

- **USGS Data Inquiries:** Contact information in dataset metadata
- **Wyoming Game and Fish:** wgf.inforequest@wyo.gov
- **WYSAC:** wyhunter@uwyo.edu

## Notes

- Most datasets are **free and open access**
- Some may require **data use agreements** (check metadata)
- **Citation required** for research publications
- **GPS collar data** is the gold standard for training
- **Harvest data** provides validation but may have bias (hunter behavior)
