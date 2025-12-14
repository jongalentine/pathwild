# Conda Environment Setup for PathWild

## Initial Setup (First Time)

```bash
# Navigate to project directory
cd /Users/jongalentine/Projects/pathwild

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate pathwild
```

## Activating the Environment

```bash
# Activate the pathwild environment
conda activate pathwild
```

## Updating the Environment

When `environment.yml` is updated, update your environment:

```bash
# Activate first (if not already active)
conda activate pathwild

# Update environment from environment.yml
conda env update -f environment.yml --prune

# Or if you prefer to recreate (cleaner, but slower):
conda deactivate
conda env remove -n pathwild
conda env create -f environment.yml
conda activate pathwild
```

## Deactivating the Environment

```bash
# Deactivate the current environment
conda deactivate
```

## Useful Commands

```bash
# List all conda environments
conda env list

# Check which environment is active
conda info --envs

# List packages in current environment
conda list

# Export current environment (if you make manual changes)
conda env export > environment.yml

# Install additional package
conda install package_name
# or
pip install package_name  # (if not available in conda)

# Remove environment (if needed)
conda deactivate
conda env remove -n pathwild
```

## Quick Reference

```bash
# Daily workflow:
conda activate pathwild          # Activate
# ... do your work ...
conda deactivate                  # When done

# After pulling updates:
conda activate pathwild
conda env update -f environment.yml --prune
```

## Troubleshooting

If you encounter issues:

1. **Environment not found**: Make sure you're using the correct name (`pathwild`)
2. **Package conflicts**: Try `conda env update --prune` to remove unused packages
3. **Geospatial packages fail**: Use conda (not pip) for rasterio/geopandas - they have complex binary dependencies
4. **Clean reinstall**: Remove and recreate the environment if things get messy
