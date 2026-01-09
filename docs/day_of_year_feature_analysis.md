# Day of Year Feature Analysis

## Question
Is `day_of_year` important for model accuracy, or is `month` sufficient?

## Current Situation
- **Feature files contain**: `month`, `year` (but not `day_of_year`)
- **Combined files contain**: `date`, `day_of_year`, `month`, `year`
- **Script behavior**: `prepare_training_features.py` removes `day_of_year` and `date` by default

## Trade-offs Analysis

### Arguments FOR Including `day_of_year`

1. **Within-Month Variation**
   - Month is coarse (30-31 day buckets)
   - Early vs late month can have different conditions
   - Example: Early October (day 274) vs late October (day 304) - 30 days difference
   - Elk behavior may change within a month (e.g., rut timing, migration)

2. **Seasonal Transitions**
   - Captures gradual transitions between seasons
   - Example: Late winter (day 60-90) vs early spring (day 90-120)
   - More granular than month boundaries

3. **Biological Patterns**
   - Elk rut typically peaks mid-September to early October
   - Migration timing is day-specific, not month-specific
   - Calving season has specific timing windows

4. **Weather/Environmental Patterns**
   - Snow melt timing is day-specific
   - NDVI changes gradually day-by-day
   - Temperature patterns vary within months

### Arguments AGAINST Including `day_of_year`

1. **Redundancy with Month**
   - Month already captures most seasonal patterns
   - Day-level granularity may not add much signal
   - Risk of overfitting to specific days

2. **Data Leakage Concerns** (Original Script Comment)
   - If `day_of_year` correlates with dataset-specific collection dates
   - Could encode information about which dataset the observation came from
   - However, this is less of a concern if datasets span multiple years

3. **Circular Encoding Required**
   - Raw `day_of_year` (1-365) treats day 365 and day 1 as very different
   - But Dec 31 and Jan 1 are temporally close
   - Must encode cyclically (sin/cos) to work properly

4. **Model Complexity**
   - Adds 2 features (sin/cos) instead of 1 (month)
   - May not improve accuracy enough to justify complexity

## Recommendation

### ✅ **Include `day_of_year` with Cyclical Encoding**

**Rationale:**
1. **Biological Relevance**: Elk behavior is day-specific (rut timing, migration)
2. **Environmental Patterns**: Weather and vegetation change day-by-day
3. **Low Risk**: If properly encoded cyclically, minimal risk of data leakage
4. **Easy to Test**: Can compare models with/without to measure impact

### Implementation

The updated `prepare_training_features.py` now:
- **Includes `day_of_year` by default** (encoded cyclically)
- Creates `day_of_year_sin` and `day_of_year_cos` features
- Removes raw `day_of_year` (avoids circular encoding issues)
- Can be disabled with `--exclude-day-of-year` flag

**Cyclical Encoding:**
```python
day_of_year_sin = sin(2π × day_of_year / 365.25)
day_of_year_cos = cos(2π × day_of_year / 365.25)
```

This ensures:
- Day 365 (Dec 31) and Day 1 (Jan 1) are treated as similar
- Smooth transitions throughout the year
- Model can learn circular temporal patterns

## Testing Strategy

### Experiment 1: Compare Feature Sets
1. **Baseline**: `month` only
2. **Test 1**: `month` + `day_of_year` (cyclical)
3. **Test 2**: `day_of_year` (cyclical) only
4. **Test 3**: All temporal features (`year`, `month`, `day_of_year` cyclical)

### Metrics to Track
- Model accuracy (target: 70%+)
- Feature importance (SHAP values)
- Cross-validation performance
- Overfitting indicators

### Expected Outcomes
- **Best case**: `day_of_year` adds 2-5% accuracy improvement
- **Likely case**: `day_of_year` adds 1-2% accuracy improvement
- **Worst case**: No improvement, but no harm either

## Usage

### Include day_of_year (default, recommended):
```bash
python scripts/prepare_training_features.py --all-datasets
```

### Exclude day_of_year:
```bash
python scripts/prepare_training_features.py --all-datasets --exclude-day-of-year
```

### Exclude all temporal features:
```bash
python scripts/prepare_training_features.py --all-datasets --exclude-temporal
```

## Conclusion

**Recommendation: Include `day_of_year` with cyclical encoding**

- Low risk, potential for meaningful accuracy gains
- Biologically and environmentally relevant
- Properly encoded to avoid circular issues
- Easy to test and compare

The cyclical encoding (sin/cos) is critical - raw `day_of_year` would be problematic. The updated script handles this automatically.

## Next Steps

1. Regenerate feature files with `day_of_year` included:
   ```bash
   python scripts/prepare_training_features.py --all-datasets
   ```

2. Train models with both feature sets and compare:
   - With `day_of_year_sin`/`day_of_year_cos`
   - Without (using `--exclude-day-of-year`)

3. Analyze feature importance to see if `day_of_year` features rank highly

4. If no improvement, can easily exclude with `--exclude-day-of-year`


