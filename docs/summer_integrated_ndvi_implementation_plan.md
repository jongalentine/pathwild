# Summer Integrated NDVI Implementation Plan

## Overview

**Summer Integrated NDVI** (iNDVI) is the sum of NDVI values from June through September, representing cumulative forage quality during the critical summer growing season. This metric is essential for predicting elk body condition and winterkill risk.

## Significance

### Why It Matters

1. **Body Condition Prediction**: Summer forage quality directly correlates with fall body fat percentage
   - High summer iNDVI → High body fat (13-15%) → Low winterkill risk
   - Low summer iNDVI → Low body fat (7-9%) → High winterkill risk

2. **Population Dynamics**: Affects calf survival and cow survival rates
   - Used in `nutrition.py` heuristic to predict recruitment rates
   - Critical for population modeling and management decisions

3. **Model Accuracy Impact**: Estimated **+5-8% accuracy improvement** for fall/winter predictions

### Current Status

- **Placeholder**: All values set to 60.0 (placeholder)
- **Impact**: Nutrition heuristic uses placeholder, limiting model accuracy
- **Priority**: **HIGH** - Most impactful of the three remaining features

## Technical Requirements

### Data Needed

1. **Time Series NDVI**: NDVI values for June 1 - September 30
2. **Multiple Satellite Passes**: 
   - Landsat 8: ~16-day revisit cycle → ~8 images per summer
   - Sentinel-2: ~5-day revisit cycle → ~24 images per summer
3. **Cloud-Free Images**: Need sufficient cloud-free coverage for accurate integration

### Calculation Method

```
summer_integrated_ndvi = Σ(NDVI_i × days_between_i)
```

Where:
- `NDVI_i` = NDVI value for image i
- `days_between_i` = Days between image i and image i+1 (or end of period)

**Alternative (simpler)**: Sum of all NDVI values (assumes equal weighting)

## Implementation Approaches

### Approach 1: GEE Time Series Collection (Recommended)

**Pros:**
- Server-side computation (no downloads)
- Handles cloud filtering automatically
- Can aggregate multiple images efficiently

**Cons:**
- Requires multiple GEE API calls
- Slower than single-point queries
- May hit rate limits for large datasets

**Implementation Steps:**

1. **Add `get_summer_integrated_ndvi` method to `GEENDVIClient`**
   ```python
   def get_summer_integrated_ndvi(
       self,
       lat: float,
       lon: float,
       year: int,
       max_cloud_cover: float = 30.0
   ) -> Optional[float]:
       """
       Calculate integrated NDVI for summer (June-September).
       
       Returns sum of NDVI values from all cloud-free images in period.
       """
   ```

2. **Use GEE ImageCollection aggregation**
   - Filter collection for June 1 - September 30
   - Filter by cloud cover
   - Calculate NDVI for each image
   - Sum NDVI values (or integrate with time weighting)

3. **Cache results** (optional)
   - Summer iNDVI doesn't change once summer ends
   - Can cache by (lat, lon, year) for performance

### Approach 2: Batch Historical NDVI Retrieval

**Pros:**
- Reuses existing `get_ndvi_for_points` infrastructure
- Can parallelize across multiple dates
- Easier to debug and test

**Cons:**
- More API calls (one per satellite pass)
- Slower for large datasets
- Requires date range iteration

**Implementation Steps:**

1. **Generate date list** for June 1 - September 30
2. **Fetch NDVI for each date** using existing methods
3. **Sum NDVI values** (or integrate with time weighting)
4. **Handle missing data** (cloudy days, no coverage)

### Approach 3: Hybrid (Recommended for MVP)

**Pros:**
- Uses GEE for efficient time series
- Falls back to batch retrieval if needed
- Best of both approaches

**Cons:**
- More complex implementation
- Requires both methods

## Implementation Plan

### Phase 1: Core Implementation (2-3 hours)

1. **Add `get_summer_integrated_ndvi` to `GEENDVIClient`**
   - Use GEE ImageCollection filtering
   - Aggregate NDVI values over June-September
   - Return integrated sum

2. **Update `build_context` to use summer iNDVI**
   - Check if observation date is after September
   - If yes, calculate summer iNDVI for current year
   - If no, use previous year's summer iNDVI (if available)

3. **Add caching** (optional but recommended)
   - Cache by (lat, lon, year) to avoid redundant calculations
   - Use in-memory cache or file-based cache

### Phase 2: Integration (1 hour)

1. **Update `add_ndvi` method** to support summer iNDVI
2. **Update `integrate_environmental_features.py`** to use new method
3. **Handle edge cases**:
   - Dates before June (use previous year)
   - Dates during summer (use partial integration)
   - Missing data (fallback to placeholder or estimate)

### Phase 3: Testing (1-2 hours)

1. **Unit tests** for `get_summer_integrated_ndvi`
2. **Integration tests** with `DataContextBuilder`
3. **Validation tests** against known good values
4. **Performance tests** for large datasets

### Phase 4: Optimization (1 hour)

1. **Batch processing** for multiple points
2. **Caching strategy** implementation
3. **Error handling** improvements
4. **Logging** and monitoring

## Code Structure

### New Method in `GEENDVIClient`

```python
@retry_on_ee_exception(max_retries=3, delay=2.0)
def get_summer_integrated_ndvi(
    self,
    lat: float,
    lon: float,
    year: int,
    max_cloud_cover: float = 30.0
) -> Optional[float]:
    """
    Calculate integrated NDVI for summer (June 1 - September 30).
    
    Args:
        lat: Latitude
        lon: Longitude
        year: Year for summer period
        max_cloud_cover: Maximum cloud cover threshold
        
    Returns:
        Integrated NDVI value (sum of NDVI values), or None if insufficient data
    """
    point = ee.Geometry.Point([lon, lat])
    
    # Define summer period
    summer_start = f"{year}-06-01"
    summer_end = f"{year}-09-30"
    
    # Filter collection
    collection = ee.ImageCollection(self.collection_config['id']) \
        .filterBounds(point) \
        .filterDate(summer_start, summer_end) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
    
    # Calculate NDVI for each image
    def add_ndvi(image):
        red_band = self.collection_config['red_band']
        nir_band = self.collection_config['nir_band']
        ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
        return image.addBands(ndvi)
    
    collection_ndvi = collection.map(add_ndvi)
    
    # Sample NDVI at point for each image
    ndvi_values = collection_ndvi.map(
        lambda img: img.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=self.collection_config['scale']
        ).get('NDVI')
    )
    
    # Get all NDVI values and sum them
    ndvi_list = ndvi_values.getInfo()
    
    if not ndvi_list or len(ndvi_list) == 0:
        return None
    
    # Filter out None values and sum
    valid_ndvi = [v for v in ndvi_list if v is not None]
    
    if len(valid_ndvi) < 2:  # Need at least 2 images for integration
        return None
    
    # Sum NDVI values (simple integration)
    integrated_ndvi = sum(valid_ndvi)
    
    return float(integrated_ndvi)
```

### Update `build_context`

```python
# Summer integrated NDVI (for nutritional condition)
if dt.month >= 9:  # After summer
    # Use current year's summer
    if self.use_gee_ndvi and self.ndvi_client:
        summer_indvi = self.ndvi_client.get_summer_integrated_ndvi(
            lat=lat,
            lon=lon,
            year=dt.year,
            max_cloud_cover=30.0
        )
        if summer_indvi is not None:
            context["summer_integrated_ndvi"] = summer_indvi
        else:
            # Fallback to placeholder
            context["summer_integrated_ndvi"] = 60.0
    else:
        # Use SatelliteClient fallback
        summer_start = datetime(dt.year, 6, 1)
        summer_end = datetime(dt.year, 9, 1)
        context["summer_integrated_ndvi"] = \
            self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
else:
    # Use previous year's summer
    if self.use_gee_ndvi and self.ndvi_client:
        summer_indvi = self.ndvi_client.get_summer_integrated_ndvi(
            lat=lat,
            lon=lon,
            year=dt.year - 1,
            max_cloud_cover=30.0
        )
        if summer_indvi is not None:
            context["summer_integrated_ndvi"] = summer_indvi
        else:
            context["summer_integrated_ndvi"] = 60.0
    else:
        summer_start = datetime(dt.year - 1, 6, 1)
        summer_end = datetime(dt.year - 1, 9, 1)
        context["summer_integrated_ndvi"] = \
            self.satellite_client.get_integrated_ndvi(lat, lon, summer_start, summer_end)
```

## Testing Strategy

### Unit Tests

1. **Test `get_summer_integrated_ndvi` with known dates**
   - Test with Wyoming coordinates
   - Verify returns reasonable values (typically 40-80)
   - Test with insufficient data (should return None)

2. **Test edge cases**
   - Dates before June (use previous year)
   - Dates during summer (partial integration)
   - Missing satellite coverage

### Integration Tests

1. **Test with `build_context`**
   - Verify summer iNDVI is included in context
   - Test fallback behavior when GEE unavailable

2. **Test with `add_ndvi`**
   - Verify summer iNDVI is added to DataFrame
   - Test batch processing

### Validation Tests

1. **Compare with known values**
   - Use test points with known summer iNDVI
   - Verify accuracy within expected range

2. **Performance tests**
   - Measure time for single point
   - Measure time for batch of 100 points
   - Verify caching improves performance

## Performance Considerations

### Expected Performance

- **Single point**: 5-10 seconds (multiple GEE API calls)
- **Batch of 100 points**: 10-15 minutes (with parallelization)
- **With caching**: 50-80% faster for repeated queries

### Optimization Strategies

1. **Caching**: Cache results by (lat, lon, year)
2. **Batch processing**: Process multiple points in parallel
3. **Lazy loading**: Only calculate when needed (after September)
4. **Approximation**: Use fewer images if full integration too slow

## Risk Mitigation

### Potential Issues

1. **Rate Limiting**: GEE may throttle requests
   - **Solution**: Add retry logic, batch processing, caching

2. **Insufficient Data**: Not enough cloud-free images
   - **Solution**: Lower cloud cover threshold, use interpolation

3. **Performance**: Too slow for large datasets
   - **Solution**: Caching, parallelization, approximation

4. **Data Quality**: Inaccurate integration
   - **Solution**: Validation tests, comparison with known values

## Success Criteria

1. ✅ Summer iNDVI calculated correctly for June-September period
2. ✅ Values in expected range (40-80 for Wyoming)
3. ✅ Integrated into `build_context` and `add_ndvi`
4. ✅ Unit and integration tests passing
5. ✅ Performance acceptable (<15 min for 100 points)
6. ✅ Fallback behavior works when GEE unavailable

## Timeline Estimate

- **Phase 1 (Core)**: 2-3 hours
- **Phase 2 (Integration)**: 1 hour
- **Phase 3 (Testing)**: 1-2 hours
- **Phase 4 (Optimization)**: 1 hour

**Total**: 5-7 hours

## Next Steps

1. Implement `get_summer_integrated_ndvi` method
2. Update `build_context` to use new method
3. Add unit tests
4. Add integration tests
5. Test with real data
6. Optimize performance if needed

## References

- GEE ImageCollection documentation: https://developers.google.com/earth-engine/guides/ic_filtering
- NDVI integration methods: https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index
- Elk nutrition research: Body condition thresholds from wildlife management literature

