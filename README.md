# PathWild 🌲

AI-powered wildlife location prediction platform

## Overview
PathWild uses machine learning to predict wildlife locations and activity patterns based on weather, terrain, and temporal data. The system helps outdoor enthusiasts, wildlife photographers, and hunters optimize their planning and increase success rates.

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
- **Cloud:** AWS SageMaker, Lambda, API Gateway
- **Web:** Flask, Bootstrap
- **Data:** NOAA Weather API, Wyoming Game & Fish

## Quick Start

### Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate pathwild

# Install dependencies
pip install -r requirements.txt
```

### Run Jupyter Lab
```bash
jupyter lab
```

### Train Model (after data collection)
```bash
python src/models/train.py
```

### Run Web App Locally
```bash
python app/app.py
# Visit http://localhost:5000
```

## Project Structure
```
pathwild/
├── data/              # Data files
│   ├── raw/          # Original data (not in Git)
│   ├── processed/    # Cleaned data
│   └── features/     # ML-ready features
├── notebooks/         # Jupyter notebooks
├── src/              # Production code
│   ├── data/         # Data collection
│   ├── features/     # Feature engineering
│   ├── models/       # Model training/prediction
│   └── deployment/   # AWS deployment
├── models/           # Trained models (not in Git)
├── app/              # Web application
└── tests/            # Unit tests
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
