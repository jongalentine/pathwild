# Dataset Naming Standardization Guide

## Overview

✅ **COMPLETED** - The pipeline has been standardized to use `northern_bighorn` (instead of `north_bighorn`) to match the actual raw data directory name.

All code references and files have been renamed.

## Code Changes ✅

All code references have been updated in:
- `scripts/run_data_pipeline.py` - Updated dataset name in docs and examples
- `scripts/assess_training_readiness.py` - Updated dataset mapping
- `scripts/process_raw_presence_data.py` - Updated examples and help text
- `scripts/integrate_environmental_features.py` - Updated example
- `scripts/prepare_training_features.py` - Updated example
- `scripts/analyze_integrated_features.py` - Updated example

## File Renaming Required

To complete the standardization, the following files need to be renamed:

### Processed Files

```bash
# Main combined file
mv data/processed/combined_north_bighorn_presence_absence.csv \
   data/processed/combined_northern_bighorn_presence_absence.csv

# Points file (if using north_bighorn_points.csv - check if northern_bighorn_points.csv exists)
# If both exist, check which is correct and remove the old one
# If only north_bighorn_points.csv exists, rename it:
mv data/processed/north_bighorn_points.csv \
   data/processed/northern_bighorn_points.csv

# Other processed files (backups, test files, etc.)
mv data/processed/combined_north_bighorn_presence_absence_test.csv \
   data/processed/combined_northern_bighorn_presence_absence_test.csv

# Cleaned/fixed versions (if needed - these may be temporary)
# mv data/processed/combined_north_bighorn_presence_absence_cleaned.csv \
#    data/processed/combined_northern_bighorn_presence_absence_cleaned.csv
# etc.

# Marker files (if any)
mv data/processed/.combined_north_bighorn_presence_absence.analyze_features.complete \
   data/processed/.combined_northern_bighorn_presence_absence.analyze_features.complete
mv data/processed/.combined_north_bighorn_presence_absence_test.analyze_features.complete \
   data/processed/.combined_northern_bighorn_presence_absence_test.analyze_features.complete

# PNG files (if any)
mv data/processed/north_bighorn_routes_area_048.png \
   data/processed/northern_bighorn_routes_area_048.png
```

### Feature Files

```bash
mv data/features/north_bighorn_features.csv \
   data/features/northern_bighorn_features.csv

mv data/features/north_bighorn_features_test.csv \
   data/features/northern_bighorn_features_test.csv
```

### Backup Files

```bash
# If you want to keep backups consistent
mv data/processed/backups/combined_north_bighorn_presence_absence_*.csv \
   data/processed/backups/combined_northern_bighorn_presence_absence_*.csv
```

## Verification

After renaming, verify the pipeline works:

```bash
# Test with the new name
python scripts/run_data_pipeline.py --dataset northern_bighorn --limit 10
```

## Notes

- **Raw directory is already correct**: `data/raw/elk_northern_bighorn/` (no change needed)
- **Both points files exist**: Check which `*_bighorn_points.csv` file is the correct one
  - `north_bighorn_points.csv` (old name)
  - `northern_bighorn_points.csv` (may already exist)
- **Backup files**: Only rename if you want consistency. They're historical backups.

## Automated Rename Script

You can use this to rename all files automatically:

```bash
#!/bin/bash
cd /Users/jongalentine/Projects/pathwild

# Processed files
for file in data/processed/*north_bighorn*; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/north_bighorn/northern_bighorn/g')
        echo "Renaming: $file -> $new_name"
        mv "$file" "$new_name"
    fi
done

# Feature files
for file in data/features/*north_bighorn*; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/north_bighorn/northern_bighorn/g')
        echo "Renaming: $file -> $new_name"
        mv "$file" "$new_name"
    fi
done

echo "Done!"
```

