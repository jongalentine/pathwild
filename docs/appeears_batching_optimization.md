# AppEEARS Batching Optimization Analysis

## Current Configuration

- **date_buffer_days**: 5 (target date ± 5 days = 11 day range)
- **max_points_per_batch**: 100
- **Current performance**: ~3,750 batches for 178,448 points needing replacement
- **Speedup**: ~95x vs. individual requests

## Optimization Potential

### Analysis Results

Based on analysis of all 4 datasets (national_refuge, northern_bighorn, southern_bighorn, southern_gye):

| Date Buffer | Batch Size | Total Batches | Avg/Batch | Utilization | Speedup |
|-------------|------------|---------------|-----------|-------------|---------|
| **5 days** | **100** | **1,979** | 90.2 | 90.2% | **98.2x** |
| 7 days | 100 | 1,931 | 92.4 | 92.4% | 92.4x |
| 10 days | 100 | 1,886 | 94.6 | 94.6% | 94.6x |
| 14 days | 200 | 971 | 183.8 | 91.9% | 183.8x |
| 21 days | 200 | 940 | 189.8 | 94.9% | 189.8x |
| **30 days** | **200** | **926** | **192.7** | **96.4%** | **192.7x** |
| 30 days | 500 | 391 | 456.4 | 91.3% | 456.4x |
| **30 days** | **1000** | **214** | **833.9** | **83.4%** | **833.9x** |

## Recommendations

### Option 1: Conservative Optimization (Recommended)

**Configuration:**
- `date_buffer_days = 10` (from 5)
- `max_points_per_batch = 200` (from 100)

**Benefits:**
- Reduces batches from ~3,750 to ~999 (73% reduction)
- Maintains high utilization (89.3%)
- Moderate increase in date range (still reasonable for NDVI matching)
- Lower risk of hitting API limits
- **Speedup: 178.6x** (vs. current 95x)

**Trade-offs:**
- Slightly wider date ranges (20 days vs. 11 days)
- Still processes more data than strictly needed, but manageable

### Option 2: Aggressive Optimization

**Configuration:**
- `date_buffer_days = 30`
- `max_points_per_batch = 1000`

**Benefits:**
- Reduces batches from ~3,750 to ~214 (94% reduction)
- Maximum speedup: **833.9x**
- Fewer API calls = lower rate limiting risk

**Trade-offs:**
- Very wide date ranges (60 days per point)
- Returns significantly more data than needed
- May hit API limits (AppEEARS doesn't document max coordinates)
- Larger response payloads
- More processing to find best match

### Option 3: Balanced Optimization

**Configuration:**
- `date_buffer_days = 14`
- `max_points_per_batch = 200`

**Benefits:**
- Reduces batches from ~3,750 to ~971 (74% reduction)
- Good utilization (91.9%)
- Moderate date range (28 days)
- **Speedup: 183.8x**

**Trade-offs:**
- Still wider than current, but reasonable

## API Constraints

Based on AppEEARS API documentation review:

- **No documented hard limit** on coordinates per request
- API evaluates `request_size` and may throttle large/complex requests
- Recommendation: Start conservative and test larger batches
- Rate limiting: 429 errors are handled with retry logic

## Implementation Recommendations

1. **Start with Option 1 (Conservative)**: Test with `date_buffer_days=10`, `max_points_per_batch=200`
2. **Monitor API responses**: Watch for:
   - Rate limiting (429 errors)
   - Task rejection due to size
   - Response times
3. **Gradually increase if stable**: If Option 1 works well, try Option 3 or even Option 2
4. **Add configuration**: Make these parameters configurable via environment variables or config file

## Code Changes Needed

Update `src/data/appeears_client.py`:

```python
# Current defaults
date_buffer_days: int = 5
max_points_per_batch: int = 100

# Recommended defaults
date_buffer_days: int = 10  # or 14 for balanced
max_points_per_batch: int = 200  # or 500 for more aggressive
```

Or make configurable:

```python
import os

date_buffer_days: int = int(os.getenv('APPEEARS_DATE_BUFFER_DAYS', '10'))
max_points_per_batch: int = int(os.getenv('APPEEARS_MAX_POINTS_PER_BATCH', '200'))
```

## Expected Impact

With **Option 1 (Conservative)**:
- **Before**: ~3,750 batches, ~95x speedup
- **After**: ~999 batches, ~179x speedup
- **Reduction**: 73% fewer API calls
- **Time savings**: Significant reduction in total processing time

## Testing Strategy

1. Test with a small subset (e.g., 1,000 points) using new parameters
2. Verify:
   - API accepts requests without errors
   - Results are correct (NDVI values match expected)
   - No rate limiting issues
   - Response times are acceptable
3. Scale up gradually if successful
