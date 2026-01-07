# PathWild 🌲

AI-powered wildlife location prediction platform

## Overview
PathWild uses machine learning to predict wildlife locations and activity patterns based on weather, terrain, and temporal data. The system helps outdoor enthusiasts, wildlife photographers, and hunters ethically optimize their planning and increase success rates.

## Current Focus
- **Species:** Rocky Mountain Elk (*Cervus canadensis nelsoni*)
- **Location:** Wyoming, USA
- **Model:** XGBoost classification (target: 70%+ accuracy)
- **Deployment:** AWS SageMaker
- **Website:** https://pathwild.ai

## Vision
Democratize wildlife prediction using AI, making expert-level insights accessible to everyone from weekend enthusiasts to professional guides.

## Future Expansion
- Mule deer, whitetail deer
- Waterfowl migration patterns
- Wildlife photography applications
- Multi-state coverage (Montana, Colorado, Idaho)

## Project Status
🚧 **In Development** - MVP targeting October 2026 field validation

## Tech Stack
- **ML:** Python 3.11, PyTorch, scikit-learn, XGBoost
- **MLOps:** MLflow, SHAP
- **Geospatial:** Rasterio, GeoPandas, Shapely (for terrain, landcover, and spatial analysis)
- **Cloud:** AWS SageMaker, Lambda, API Gateway
- **Web:** Flask, Bootstrap
- **Data:** NOAA Weather API, Wyoming Game & Fish, SNOTEL, satellite imagery

## Quick Start

### Setup Environment
```bash
# Create conda environment (recommended - handles geospatial dependencies better)
conda env create -f environment.yml
conda activate pathwild

# Or install via pip (if not using conda)
pip install -r requirements.txt
```

**Note:** Geospatial packages (rasterio, geopandas) have binary dependencies (GDAL). Conda is recommended as it handles these dependencies automatically.

### Run Automated Data Pipeline

Process raw elk GPS collar data into training-ready datasets:

```bash
# Process all datasets end-to-end
python scripts/run_data_pipeline.py

# Process specific dataset
python scripts/run_data_pipeline.py --dataset north_bighorn

# Skip already-complete steps
python scripts/run_data_pipeline.py --skip-steps process_raw,generate_absence
```

**⚠️ Prerequisites:** Before running the pipeline, ensure all required environmental data files are present. The pipeline will automatically check prerequisites and fail fast if required files are missing. See [Environmental Data Prerequisites Guide](./docs/environmental_data_prerequisites.md) for detailed instructions.

**📊 NDVI & Weather Data**: The pipeline now supports real NDVI (AppEEARS) and weather data (PRISM + Open-Meteo). Set `APPEEARS_USERNAME` and `APPEEARS_PASSWORD` environment variables to enable real NDVI data. See [NDVI/Weather Integration Status](./docs/ndvi_weather_integration_status.md) for details.

See [Automated Data Pipeline Documentation](./docs/automated_data_pipeline.md) for details.



## Project Structure
```
pathwild/
├── config.yaml        # Configuration file
├── data/              # Data files (not in Git)
│   ├── raw/          # Original data
│   ├── processed/    # Cleaned data
│   ├── features/     # ML-ready features
│   ├── dem/          # Digital elevation models
│   ├── terrain/      # Slope, aspect data
│   ├── landcover/    # Land cover classifications
│   ├── canopy/       # Canopy cover data
│   ├── hydrology/    # Water sources
│   ├── infrastructure/ # Roads, trails
│   └── wildlife/     # Predator territories, activity
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Production code
│   ├── data/          # Data processing and context building
│   │   └── processors.py  # DataContextBuilder, SNOTEL, Weather, Satellite clients
│   ├── examples/      # Usage examples
│   │   └── example_usage.py
│   ├── features/      # Feature engineering
│   ├── inference/     # Inference engine
│   │   └── engine.py
│   ├── models/        # Model training/prediction
│   ├── scoring/       # Scoring and heuristic modules
│   │   ├── aggregator.py
│   │   └── heuristics/  # Individual heuristic modules
│   │       ├── access.py
│   │       ├── elevation.py
│   │       ├── nutrition.py
│   │       ├── predation.py
│   │       ├── security.py
│   │       ├── snow.py
│   │       ├── vegetation.py
│   │       ├── water.py
│   │       └── winterkill.py
│   └── deployment/    # AWS deployment
└── tests/             # Unit tests
    ├── test_aggregator.py
    ├── test_data_context.py
    ├── test_heuristics.py
    ├── test_inference_engine.py
    ├── test_integration.py
    └── test_validation.py
```

## Development Roadmap

### Phase 1 (Months 1-3): Wyoming Elk MVP
- [x] Environment setup
- [ ] Data collection (WGFD harvest data, NOAA weather)
- [ ] Feature engineering
- [ ] Model training (target: 70%+ accuracy)
- [ ] AWS deployment
- [ ] Web interface at pathwild.ai

### Phase 2 (Months 4-12): Validation & Refinement
- [ ] October 2026 field validation
- [ ] Model refinement based on real-world results
- [ ] Community beta testing
- [ ] Performance optimization

### Phase 3 (Year 2+): Multi-Species Expansion
- [ ] Mule deer predictions
- [ ] Waterfowl migration patterns
- [ ] Wildlife photography mode
- [ ] Mobile apps (iOS/Android)

## Contributing
This is a personal learning project, but feedback welcome! Open an issue or reach out.

## Contact
Jon Galentine  - jongalentine@gmail.com  
Project: https://github.com/jongalentine/pathwild  
Website: https://pathwild.ai

## License
MIT License (or your choice)

---

**PathWild** - Predict. Plan. Succeed.
