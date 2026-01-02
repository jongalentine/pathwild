# Data Pipeline Quick Reference

## One-Command Pipeline Execution

```bash
# Process all datasets
python scripts/run_data_pipeline.py

# Process specific dataset
python scripts/run_data_pipeline.py --dataset north_bighorn
```

## Individual Step Commands

### 1. Process Raw Data
```bash
python scripts/process_raw_presence_data.py --dataset north_bighorn
```

### 2. Generate Absence Data
```bash
python scripts/generate_absence_data.py \
    --presence-file data/processed/north_bighorn_points.csv \
    --output-file data/processed/combined_north_bighorn_presence_absence.csv
```

### 3. Integrate Features
```bash
# Incremental (only placeholders)
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv

# Force full regeneration
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv --force
```

### 4. Analyze Features
```bash
python scripts/analyze_integrated_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv
```

### 5. Assess Readiness
```bash
python scripts/assess_training_readiness.py
```

## Common Workflows

### First-Time Setup
```bash
# 1. Process all raw datasets
python scripts/process_raw_presence_data.py

# 2. Generate absence for each dataset
for dataset in north_bighorn southern_bighorn national_refuge southern_gye; do
    python scripts/generate_absence_data.py \
        --presence-file data/processed/${dataset}_points.csv \
        --output-file data/processed/combined_${dataset}_presence_absence.csv
done

# 3. Integrate features for each dataset
for dataset in north_bighorn southern_bighorn national_refuge southern_gye; do
    python scripts/integrate_environmental_features.py \
        data/processed/combined_${dataset}_presence_absence.csv
done

# 4. Assess all datasets
python scripts/assess_training_readiness.py
```

### Incremental Update (New Environmental Data)
```bash
# Only re-integrate features (skips rows without placeholders)
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv
```

### Full Regeneration
```bash
# Force regeneration of all features
python scripts/integrate_environmental_features.py \
    data/processed/combined_north_bighorn_presence_absence.csv --force
```

## Pipeline Output Files

### Presence Points
- `data/processed/{dataset}_points.csv` - Presence points from raw data

### Combined Datasets
- `data/processed/combined_{dataset}_presence_absence.csv` - Final training-ready dataset

### Backups
- `data/processed/backups/combined_{dataset}_presence_absence_{timestamp}.csv` - Timestamped backups

## Expected Processing Times

| Step | Time (50K points) | Notes |
|------|-------------------|-------|
| Process raw | < 1 min | Fast |
| Generate absence | 5-15 min | Depends on strategies |
| Integrate features | 30-60 min | ~1000-2000 rows/min/worker |
| Analyze features | < 1 min | Fast |
| Assess readiness | < 1 min | Fast |

**Total:** ~40-80 minutes per dataset

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No datasets found" | Check `data/raw/elk_*/` directories exist |
| "Required input not found" | Run steps in order or use `--skip-steps` |
| "Environmental data not found" | Verify all environmental data files exist |
| "Placeholders not replaced" | Check data files, use `--force` to regenerate |
| Slow processing | Reduce `--workers` or use incremental mode |

## Testing

```bash
# Run all pipeline tests
pytest tests/test_data_pipeline.py tests/test_pipeline_integration.py -v

# Test individual components
pytest tests/test_integrate_environmental_features.py -v
```

