# SNOTEL Station Mapping Status

This document tracks the mapping status of USDA SNOTEL stations to `snotelr` site IDs.

## Summary

- **Total stations**: 36
- **Mapped stations**: 31 (86.1%)
- **Unmapped stations**: 5 (13.9%)

## Mapping Process

The mapping script (`scripts/map_snotel_station_ids.py`) uses multiple strategies to match USDA station triplets to `snotelr` site IDs:

1. **Exact normalized name match** - Removes common suffixes and normalizes whitespace
2. **Exact original match** - Case-insensitive exact match
3. **Fuzzy string matching (high threshold)** - Similarity ≥ 85%
4. **Partial match** - Matches first 2-3 words of station name
5. **Location-based matching (close)** - Within ~5.5 km
6. **Location-based matching (relaxed)** - Within ~11 km
7. **Fuzzy string matching (low threshold)** - Similarity ≥ 70%

## Unmapped Stations

The following 5 stations cannot be mapped because they do not exist in the `snotelr` database:

| Station Name | Triplet | Location | Status |
|-------------|---------|----------|--------|
| ELKHORN PARK | SNOTEL:WY:967 | 41.5667, -106.8667 | ⚠️ Not in snotelr |
| LARAMIE RIVER | SNOTEL:WY:972 | 41.2833, -105.9667 | ⚠️ Not in snotelr |
| LONG POND | SNOTEL:WY:974 | 44.6833, -110.35 | ⚠️ Not in snotelr |
| TOWER FALLS | SNOTEL:WY:986 | 44.9167, -110.4167 | ⚠️ Not in snotelr |
| YELLOWSTONE LAKE | SNOTEL:WY:990 | 44.5333, -110.3667 | ⚠️ Not in snotelr |

**These stations likely:**
- Were discontinued or closed
- Were renamed/merged with other stations
- Only exist in USDA database but not in `snotelr`

## Impact

For unmapped stations, the system automatically falls back to **elevation-based estimates** for snow data. This is handled gracefully by the `SNOTELClient` class, which:

1. Attempts to find the nearest SNOTEL station
2. If no station is found or mapped, uses elevation-based estimates
3. Marks the data source as `"estimate"` in the data quality tracking fields

The `snow_data_source` field in the context will be:
- `"snotel"` for mapped stations (real SNOTEL data)
- `"estimate"` for unmapped stations (elevation-based estimates)

## Recent Mappings (2026-01-02)

Successfully mapped 6 stations using fuzzy matching:
- BATTLE CREEK → Castle Creek (site_id: 1130)
- BLACK HALL MOUNTAIN → Blackhall Mtn (site_id: 1119)
- BROOKS LAKE → Brooklyn Lake (site_id: 367)
- CROUSE CREEK → Crow Creek (site_id: 1045)
- KENDALL RANGER STATION → Kendall R.S. (site_id: 555)
- TWIN CREEK → Tie Creek (site_id: 818)

## Updating Mappings

To update station mappings:

```bash
conda activate pathwild
python scripts/map_snotel_station_ids.py
```

The script will:
1. Preserve existing mappings
2. Attempt to map any unmapped stations using all available strategies
3. Save the updated mappings to `data/cache/snotel_stations_wyoming.geojson`

## Verification

Test the mappings:

```bash
conda activate pathwild
pytest tests/test_snotel_integration.py -v
```

Or test with a small dataset:

```bash
python scripts/integrate_environmental_features.py \
  data/processed/combined_southern_bighorn_presence_absence.csv \
  --limit 50
```

Check the output for `snow_data_source` field to see which rows used real SNOTEL data vs. estimates.

