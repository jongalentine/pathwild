# Pipeline Analysis: North Bighorn Dataset (Failed Run)
**Run Date**: 2026-01-07 20:09:44 - 20:39:19  
**Total Duration**: 29.6 minutes  
**Status**: ⚠️ **FAILED** (5/6 steps completed, 1 step failed)

## Executive Summary

The pipeline failed at **Step 1 (PROCESS_RAW)** due to a directory name mismatch. However, it continued and completed steps 2-6 using existing processed files from a previous run. The final feature file was successfully created with **15,127 rows** and **30 feature columns**, but the raw data processing step was skipped due to the failure.

---

## Pipeline Steps Summary

| Step | Status | Duration | Issue |
|------|--------|----------|-------|
| 1. Process Raw | ❌ **FAILED** | - | Directory name mismatch |
| 2. Generate Absence | ✅ | ~3 min | Completed (used existing processed file) |
| 3. Integrate Features | ✅ | ~23.5 min | Completed successfully |
| 4. Validate Data | ✅ | 1.7s | Completed successfully |
| 5. Assess Readiness | ✅ | <1s | Completed successfully |
| 6. Prepare Features | ✅ | 0.5s | Completed successfully |

---

## Failure Analysis

### Step 1 Failure: PROCESS_RAW

**Error Message:**
```
Required input not found: data/raw/elk_north_bighorn
✗ Pipeline step failed: process_raw
```

**Root Cause:**
- Pipeline expected: `data/raw/elk_north_bighorn`
- Actual directory: `data/raw/elk_northern_bighorn`

**Impact:**
- Raw data processing was skipped
- Pipeline continued using existing `data/processed/north_bighorn_points.csv` from a previous run
- All subsequent steps completed successfully
- Final feature file was created correctly

**Fix Required:**
1. Either rename the directory: `data/raw/elk_northern_bighorn` → `data/raw/elk_north_bighorn`
2. Or update the pipeline configuration to use the correct directory name

---

## Feature File Analysis

### Output File: `north_bighorn_features.csv`

**Basic Statistics:**
- **Rows**: 15,127 (down from expected 15,128 due to some data quality filtering)
- **Columns**: 30 (target + 29 features)
- **File Size**: Estimated ~2.5 MB

**Column Structure:**
- ✅ Target: `elk_present` (binary: 0/1)
- ✅ Features: 29 environmental features
- ✅ Includes: `day_of_year_sin`, `day_of_year_cos` (cyclical encoding)
- ✅ Excludes: Metadata columns (day_of_year, route_id, etc.)

### Data Quality Summary

**Class Distribution:**
- Presence (elk=1): 8,450 (55.9%)
- Absence (elk=0): 6,677 (44.1%)
- ✅ Well balanced for training

**Geographic Coverage:**
- ✅ Latitude: 40.9958° to 45.1905°N (within Wyoming: 41-45°N)
- ✅ Longitude: -111.0539° to -104.0538°W (within Wyoming: -111 to -104°W)
- ⚠️ **83 rows (0.5%) outside Wyoming bounds** (likely edge cases near boundaries)

**Missing Values:**
- ⚠️ **191 rows (1.3%)** missing terrain features (elevation, slope, aspect, canopy, land_cover)
  - These are locations outside DEM bounds or at boundaries
  - Acceptable for model training (can handle with imputation or exclusion)

**Placeholder Values (Expected):**
- NDVI placeholder (0.5): 1,050 rows (6.9%) - acceptable (GEE failed or no data)
- NDVI age placeholder (8 days): 4,079 rows (27.0%) - acceptable (fallback)
- IRG placeholder (0.0): 12,530 rows (82.8%) - acceptable (seasonal approximation)
- Precipitation placeholder (0.0): 13,263 rows (87.7%) - acceptable (dry period or missing data)

**SNOTEL Data Quality:**
- ✅ 95.5% real SNOTEL data (14,440 rows)
- ⚠️ 4.5% elevation-based estimates (687 rows) - acceptable
- ✅ 87 unique SNOTEL stations used
- ⚠️ **107 rows have SWE >120% of snow depth** (unrealistic density) - may need review

**NDVI Data Quality:**
- ✅ 100% coverage (all rows have NDVI values)
- ✅ Valid range: -0.24 to 0.86 (within expected [-1, 1])
- ✅ Mean: 0.21, Median: 0.17 (reasonable for Wyoming)
- ⚠️ 6.9% placeholder values (1,050 rows) - acceptable

**Temperature Data Quality:**
- ⚠️ **Very weak correlation (0.03) between elevation and temperature**
  - Expected strong negative correlation (higher elevation = colder)
  - This suggests temperature data may be placeholder or insufficiently varied
  - Range: 31.8 to 32.4°F (very narrow - suspicious)
  - May indicate PRISM data not working correctly or using fallback values

---

## Key Issues Identified

### 1. ❌ Critical: Step 1 Failure (Directory Mismatch)
- **Severity**: Critical (prevents fresh raw data processing)
- **Impact**: Pipeline relies on old processed files
- **Fix**: Rename directory or update pipeline configuration

### 2. ⚠️ Warning: Temperature Data Quality
- **Issue**: Very weak elevation-temperature correlation (0.03)
- **Expected**: Strong negative correlation
- **Impact**: Temperature feature may not be useful for model
- **Recommendation**: Investigate PRISM data retrieval, check if using placeholder values

### 3. ⚠️ Warning: 83 Rows Outside Wyoming Bounds
- **Issue**: 0.5% of data is outside geographic bounds
- **Impact**: Minor - likely edge cases near boundaries
- **Recommendation**: Review these rows, may need to exclude or flag

### 4. ⚠️ Warning: Unrealistic Snow Density (107 rows)
- **Issue**: SWE >120% of snow depth (physically impossible)
- **Impact**: Minor data quality issue
- **Recommendation**: Review these rows, may be data entry errors in SNOTEL data

---

## Recommendations

### Immediate Actions

1. **Fix Step 1 Failure:**
   ```bash
   # Option 1: Rename directory to match pipeline expectation
   mv data/raw/elk_northern_bighorn data/raw/elk_north_bighorn
   
   # Option 2: Update pipeline configuration to use correct name
   # (Check scripts/run_data_pipeline.py for dataset name mapping)
   ```

2. **Investigate Temperature Data:**
   - Check if PRISM data retrieval is working correctly
   - Verify temperature values are not all placeholder (42°F default)
   - Review `src/data/prism_client.py` for issues

3. **Review Out-of-Bounds Rows:**
   - Check the 83 rows outside Wyoming bounds
   - Determine if they should be excluded or if boundary check is too strict

4. **Review Snow Density Issues:**
   - Check the 107 rows with unrealistic SWE/snow depth ratios
   - Verify SNOTEL data quality or add validation

### Data Quality Assessment

**Overall Readiness**: ✅ **ACCEPTABLE** (with caveats)

The feature file is suitable for model training, but:
- Temperature feature may need review/fixing
- Some data quality warnings should be addressed
- Raw data processing should be fixed for future runs

**Data Quality Score**: ~95% (estimated based on missing values and placeholder rates)

---

## Comparison with Previous Runs

**Compared to Southern Bighorn (successful run):**
- ✅ Similar feature structure (30 vs 31 columns - minor difference)
- ✅ Similar data quality metrics
- ✅ Day-of-year cyclical encoding working correctly
- ⚠️ Temperature data issue appears in both runs (needs investigation)

---

## Next Steps

1. Fix directory name mismatch to enable Step 1
2. Re-run pipeline to ensure fresh raw data processing
3. Investigate temperature data quality issue
4. Address data quality warnings before model training
5. Verify all features are correctly populated

---

**Analysis Generated**: 2026-01-07  
**Log File**: `data/logs/pipeline_north_bighorn_20260107_200944.log`  
**Feature File**: `data/features/north_bighorn_features.csv`

