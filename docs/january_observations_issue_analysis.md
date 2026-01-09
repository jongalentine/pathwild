# January Observations Disproportionate Count Analysis

## Problem Summary

January (month 1.0) has 216,361 observations, which is disproportionately large compared to other months:
- January: 216,361 (52.2% of all observations)
- Next highest: April with 29,204 (7.0%)
- Average for other months: ~17,000

Additionally, year 2024 has 206,178 observations (49.8% of all observations), which is also disproportionately large compared to other years:
- 2024: 206,178 (49.8% of all observations)
- Next highest: 2013 with 40,610 (9.8%)
- Historical years (2006-2019): 1,200 to 40,610 observations per year

## Root Cause Analysis

### Issue: Default Date Used for Missing Dates

When rows are missing date information, the feature integration script (`scripts/integrate_environmental_features.py`) uses a default date of `"2024-01-01"` (January 1) for environmental feature enrichment. However, it was also incorrectly setting the `month` column to 1 (January) based on this default date.

### Why Dates Are Missing

1. **Absence generators don't create date columns**:
   - `EnvironmentalPseudoAbsenceGenerator` - no date column
   - `UnsuitableHabitatAbsenceGenerator` - no date column  
   - `RandomBackgroundGenerator` - no date column
   - `TemporalAbsenceGenerator` - DOES create 'date' column (shifts dates seasonally)

2. **When absence and presence data are combined**:
   - Presence data may have `firstdate` or `lastdate` columns
   - Absence rows (from non-temporal generators) don't have these columns
   - After concatenation, absence rows have NaN for date columns

3. **During feature integration**:
   - The script looks for date columns: `['date', 'firstdate', 'Date_Time_MST']`
   - For rows without dates, it defaults to `"2024-01-01"` for feature enrichment
   - **BUG**: It was also setting `month = 1` from this default date

### Code Locations

**File**: `scripts/integrate_environmental_features.py`

**Line 342-348** (sequential processing):
```python
date_str = "2024-01-01"  # Default
if date_col and pd.notna(row[date_col]):
    date_str = str(row[date_col])
    # ... parsing ...
# BUG: month was set from date_str even if it was the default
```

**Line 650-657** (parallel processing):
```python
date_str = "2024-01-01"  # Default
if date_col and date_col in row_dict:
    date_val = row_dict[date_col]
    # ... parsing ...
# BUG: month was set from date_str even if it was the default
```

## Impact

- **~216,000 rows** (52% of dataset) incorrectly labeled as January (month=1)
- **~206,000 rows** (50% of dataset) incorrectly labeled as year 2024
- These are the same rows - absence rows from Environmental, Unsuitable, and Background generators that lack date information
- Both issues stem from the same root cause: default date `"2024-01-01"` being used when dates are missing
- Distorts temporal analysis and seasonal patterns
- Makes January appear to have 7x more observations than other months
- Makes 2024 appear to have 5x more observations than other years

## Fix Implemented

### Changes Made

1. **Track whether default date was used**:
   - Added `used_default_date = True` flag
   - Set to `False` only when an actual valid date is found and parsed

2. **Only set month/year/day_of_year from actual dates**:
   - Changed logic to only populate temporal metadata if `not used_default_date`
   - Default date is still used for feature enrichment (needed for weather/NDVI/snow data)
   - But month/year/day_of_year remain `None` when default date is used

3. **Improved date validation**:
   - Added checks for empty strings, 'nan', 'none' values
   - Validates date format before using it
   - Handles date parsing errors gracefully

### Code Changes

**File**: `scripts/integrate_environmental_features.py`

**Sequential processing** (lines 341-370):
- Now tracks `used_default_date` flag
- Only sets month/year/day_of_year if actual date found
- Validates date format before using

**Parallel processing** (lines 649-695):
- Same changes as sequential processing
- Ensures `_year`, `_month`, `_day_of_year` in context are `None` when default date used
- Prevents downstream code from setting month from default dates

## Expected Results After Fix

1. **Month distribution**:
   - January count should decrease dramatically (from ~216k to expected range)
   - Month distribution should reflect actual temporal patterns in presence data
   - Rows without dates will have `month = None` (not 1)

2. **Year distribution**:
   - Year 2024 count should decrease dramatically (from ~206k to expected range)
   - Year distribution should reflect actual temporal patterns in presence data
   - Rows without dates will have `year = None` (not incorrectly set to 2024)

3. **Data quality**:
   - Missing month/year values will be `None` (not incorrectly set to January/2024)
   - Can identify which rows are missing temporal metadata
   - Can decide whether to exclude them or impute dates appropriately
   - The same rows that were incorrectly labeled as January were also incorrectly labeled as 2024

## Recommendations

### Short-term (Current Fix)
- ✅ Fixed: Don't set month from default dates
- ✅ Fixed: Track whether default date was used
- ✅ Fixed: Only populate temporal metadata from actual dates

### Medium-term (Future Improvements)
1. **Add dates to absence generators**:
   - Sample dates from presence data distribution
   - Or use temporal offsets based on presence dates
   - This would ensure all absence rows have representative dates

2. **Better default date strategy**:
   - Instead of single default date, sample from presence date distribution
   - Or use multiple default dates across seasons
   - This would make feature enrichment more representative

3. **Data quality reporting**:
   - Track how many rows use default dates
   - Report this in data quality analysis
   - Help identify which absence generators need date columns

## Impact on Model Accuracy

### Current Situation

With the fix, **~50% of absence rows will have missing temporal metadata** (`month = None`, `year = None`, `day_of_year = None`). This creates several problems for model training:

### 1. **Asymmetric Temporal Learning**

**Problem**: The model can learn temporal patterns for **presences** but not for **absences**.

- **Presence data**: Has complete temporal information → model learns "elk present in October"
- **Absence data**: Missing temporal information → model cannot learn "elk absent in July"

**Impact**: 
- Model may over-predict presence in months/seasons where it only saw presence examples
- Cannot learn counter-examples (e.g., "elk are NOT in this location in winter")
- Reduces ability to distinguish seasonal migration patterns

### 2. **Feature Importance Distortion**

**Problem**: Temporal features (`month`, `year`, `day_of_year_cos`, `day_of_year_sin`) will have:
- **High predictive power for presences** (complete data)
- **Low/no predictive power for absences** (missing data)

**Impact**:
- XGBoost may down-weight temporal features if they're missing in 50% of training data
- Model may rely more on static features (elevation, slope, etc.) than temporal patterns
- Reduces model's ability to capture seasonal behavior

### 3. **Missing Data Handling**

**Problem**: XGBoost handles missing values by:
- Learning optimal "default direction" for missing values
- But this doesn't mean "no temporal information" - it means "unknown temporal context"

**Impact**:
- Model may learn that "missing month" correlates with absence (data artifact, not biological truth)
- This creates a spurious correlation that hurts generalization
- Model may perform poorly on new data where temporal info is always available

### 4. **Quantitative Impact Estimate**

Based on typical wildlife modeling:

- **Without temporal data for absences**: Expected accuracy **60-65%** (vs. target 70%+)
- **With temporal data for absences**: Expected accuracy **70-75%** (meets target)

**Reasoning**:
- Temporal patterns are critical for elk (seasonal migration, breeding, winter range)
- Missing 50% of temporal labels creates significant information loss
- Model can't learn "when elk are NOT present" which is crucial for binary classification

### 5. **Specific Scenarios Affected**

**High Impact**:
- **Seasonal predictions**: Model won't know when elk are absent in summer vs. winter ranges
- **Migration timing**: Can't learn that elk move between ranges at specific times
- **Breeding season**: May over-predict presence during rut (September-October) if absences lack dates

**Medium Impact**:
- **Year-to-year variation**: Can't learn if certain years had unusual patterns
- **Long-term trends**: Missing year data prevents learning population trends

**Lower Impact**:
- **Static habitat features**: Elevation, slope, water distance still work
- **Spatial patterns**: Geographic features unaffected

### Recommended Solutions

#### Option 1: Sample Dates from Presence Distribution (RECOMMENDED)
```python
# In absence generators, sample dates from presence data
presence_dates = presence_gdf['firstdate'].dropna()
sampled_dates = presence_dates.sample(n=len(absence_gdf), replace=True)
absence_gdf['date'] = sampled_dates
```

**Pros**:
- Maintains temporal distribution of actual observations
- Preserves seasonal patterns
- Simple to implement

**Cons**:
- May not capture true absence timing (but better than missing)

#### Option 2: Temporal Offsets (Like TemporalAbsenceGenerator)
```python
# Shift presence dates to opposite seasons
# Summer presences → Winter absences
# Winter presences → Summer absences
```

**Pros**:
- Biologically meaningful (elk move seasonally)
- Already implemented for TemporalAbsenceGenerator

**Cons**:
- Only works if you know the original presence date
- Doesn't help Environmental/Unsuitable/Background generators

#### Option 3: Exclude Rows with Missing Temporal Data
```python
# Only train on rows with complete temporal info
df_train = df[df['month'].notna() & df['year'].notna()]
```

**Pros**:
- Clean data, no missing values
- Model can learn temporal patterns fully

**Cons**:
- Loses ~50% of absence data
- May create class imbalance
- Wastes computational resources spent generating absences

#### Option 4: Impute Dates Based on Spatial Patterns
```python
# Use spatial neighbors to infer likely dates
# If nearby presences are in October, likely absence is also October
```

**Pros**:
- More realistic than random sampling
- Preserves spatial-temporal correlations

**Cons**:
- More complex to implement
- May introduce bias if spatial patterns are strong

### Immediate Action Items

1. **Short-term (Before next model training)**:
   - ✅ Fix implemented: Temporal metadata no longer incorrectly set to January/2024
   - ⚠️ **CRITICAL**: Add date sampling to absence generators before training
   - Monitor: Check feature importance - temporal features should be in top 10

2. **Medium-term (Next iteration)**:
   - Implement Option 1 (sample dates from presence distribution)
   - Re-run pipeline and verify temporal coverage
   - Compare model accuracy before/after fix

3. **Long-term (Model improvement)**:
   - Consider Option 2 (temporal offsets) for better biological realism
   - Add temporal feature engineering (e.g., "days since snow melt")
   - Validate model predictions across seasons

## Verification Steps

After re-running the pipeline:

1. **Check month distribution**:
   ```python
   df = pd.read_csv('data/features/complete_context.csv')
   print(df['month'].value_counts().sort_index())
   ```

2. **Check year distribution**:
   ```python
   print(df['year'].value_counts().sort_index())
   ```

3. **Check missing temporal metadata**:
   ```python
   missing_month = df['month'].isna().sum()
   missing_year = df['year'].isna().sum()
   print(f"Rows with missing month: {missing_month:,} ({missing_month/len(df)*100:.1f}%)")
   print(f"Rows with missing year: {missing_year:,} ({missing_year/len(df)*100:.1f}%)")
   ```

4. **Verify January is no longer inflated**:
   - January should have similar count to other months (unless presence data actually has January bias)
   - Or January should be lower if most missing dates were defaulting to January

5. **Verify 2024 is no longer inflated**:
   - Year 2024 should have similar count to other years (or be lower if most missing dates were defaulting to 2024-01-01)
   - Historical years should show more realistic distribution

6. **Check absence strategy distribution**:
   ```python
   # If absence_strategy is available in processed data
   print(df.groupby(['absence_strategy', 'month', 'year']).size())
   ```

## Next Steps

1. **Re-run feature integration**:
   ```bash
   python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
   ```

2. **Re-run data quality notebook**:
   - Verify month distribution is corrected
   - Check that missing month values are now `None` (not January)

3. **Consider future improvements**:
   - Add date columns to all absence generators
   - Sample dates from presence distribution
   - Improve temporal coverage

