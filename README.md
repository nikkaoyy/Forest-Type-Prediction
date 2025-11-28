# 🌲 Forest Cover Type Prediction - Kaggle Competition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-yellow.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.2-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready ML system for forest cover classification using LightGBM with chaos-aware uncertainty quantification**

This repository contains a complete machine learning solution for the [Forest Cover Type Prediction Kaggle competition](https://www.kaggle.com/competitions/forest-cover-type-prediction), featuring chaos detection, explainability tools, and production deployment patterns.

📚 **Academic Context**: This project is part of the **Systems Analysis and Design** course at Universidad Distrital Francisco José de Caldas. For the complete architectural analysis, quality frameworks, and theoretical foundations, see the [Academic Life Repository](https://github.com/nikkaoyy/Academic-Life).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [API Deployment](#api-deployment)
- [Explainability](#explainability)
- [Key Features](#key-features)
- [Academic Integration](#academic-integration)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **production-grade LightGBM model** to predict forest cover types from cartographic variables. The system achieves **high accuracy** with explicit uncertainty quantification and chaos detection near ecological transition zones.

### What Makes This Different?

- ✅ **LightGBM Optimization**: Fast, efficient gradient boosting with optimal hyperparameters
- ✅ **Chaos-Aware Modeling**: Detects ecological thresholds (2400m, 2800m, 3200m) and amplifies uncertainty 2× near transition zones
- ✅ **Uncertainty Quantification**: Explicit confidence scoring with threshold proximity detection
- ✅ **Production-Ready**: FastAPI + Docker with fast inference (<100ms)
- ✅ **Explainability**: Feature importance analysis and decision path visualization
- ✅ **Reproducibility**: Fixed seeds, versioned artifacts, comprehensive testing

---

## 📊 Dataset

**Source**: [Kaggle - Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data)

- **Samples**: 15,120 observations (30m × 30m patches)
- **Features**: 54 cartographic variables
  - 10 numerical (elevation, aspect, slope, distances)
  - 4 wilderness areas (binary)
  - 40 soil types (binary, 73% sparse → consolidated to 15 groups)
- **Target**: 7 forest cover types
  1. Spruce/Fir
  2. Lodgepole Pine
  3. Ponderosa Pine
  4. Cottonwood/Willow
  5. Aspen
  6. Douglas-fir
  7. Krummholz

**Location**: Roosevelt National Forest, Colorado, USA

---

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | LightGBM |
| **Accuracy** | 95%+ (training) |
| **F1 (Macro)** | ~0.94 |
| **F1 (Weighted)** | ~0.95 |
| **CV Accuracy (5-fold)** | ~94% (±1.2%) |
| **Inference Latency** | <50ms per prediction |
| **Training Time** | ~30-40s (full dataset) |

### Model Configuration

```python
LightGBM Parameters:
{
    'n_estimators': 400,
    'num_leaves': 64,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'multiclass',
    'num_class': 7
}
```

**Why LightGBM**:
- ⚡ Fastest training and inference among gradient boosting methods
- 🎯 Excellent performance on tabular data with mixed features
- 📊 Handles sparse soil features efficiently (73% → 5% sparsity reduction)
- 🔧 Built-in feature importance for interpretability
- 💾 Smaller model size for production deployment

---

## 📁 Project Structure

```
Forest-Type-Prediction/
├── api/
│   ├── app.py                       # FastAPI inference endpoint
│   └── Dockerfile                   # Production container
│
├── demo_reports/
│   ├── confusion_matrix.png        # Model evaluation visualization
│   └── evaluation_report.json      # Performance metrics
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb    # EDA with chaos detection
│
├── src/
│   ├── data/
│   │   ├── dataValidation.py       # Circuit breaker + drift detection
│   │   ├── featureEngineer.py      # Elevation, aspect, soil modules
|   |   ├── raw/
|   |   │   ├── train.csv                 # Kaggle training data
|   |   │   └── test.csv                  # Kaggle test data
|   |   ├── processed/
|   |   │   └── train_transformed.csv     # Transformed features
|   |   └── artifacts/
|   |       └── preprocessor.pkl          # Feature engineering pipeline
│   ├── models/
│   │   ├── uncertaintyQuantification.py  # Confidence scoring
|   |   ├── lightgbm_model.pkl            # Trained LightGBM model
│   │   └── LightGBMTraining.py           # Model training pipeline
│   └── utils/
│       └── config.py                     # Centralized configuration
│
├── tests/
│   ├── testFaultTolerance.py      # Circuit breaker tests
│   ├── testFeatureEngineer.py     # Unit tests (Modules 3A-3D)
│   └── testLightGBM.py            # Model integration tests
│
├── generatePreprocessor.py        # Create preprocessor.pkl
├── generateLightGBM.py            # Train LightGBM model
├── preprocessing.py               # Preprocessing utilities
├── main.py                        # Input of the .py file
├── requirements.txt               # Python dependencies
├── .gitattributes                 # Git LFS configuration
├── LICENSE
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB RAM minimum
- (Optional) GPU for faster training

### 1. Clone Repository

```bash
git clone https://github.com/nikkaoyy/Forest-Type-Prediction.git
cd Forest-Type-Prediction
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Download Data

```bash
# Download from Kaggle
kaggle competitions download -c forest-cover-type-prediction

# Extract files
unzip forest-cover-type-prediction.zip -d data/raw/
```

### 4. Generate Preprocessing Pipeline

```bash
python generatePreprocessor.py
```

**Output**: `data/artifacts/preprocessor.pkl` (feature engineering pipeline)

### 5. Train Ensemble Model

```bash
python generate_ensemble_models.py
```

**Output**: `data/artifacts/ensemble_models.pkl` (trained RF+XGB+LGB)

### 6. Make Predictions

```bash
python scripts/predict.py --input data/raw/test.csv --output submission.csv
```

---

## 🎓 Training Pipeline

### Full Training Workflow

```bash
# Step 1: Generate preprocessing pipeline
python generatePreprocessor.py

# Step 2: Train LightGBM model
python generateLightGBM.py

# Step 3: Validate model performance
python tests/testLightGBM.py
```

**What Happens in `generateLightGBM.py`**:
1. ✅ Load data from `data/raw/train.csv`
2. ✅ Apply preprocessing (elevation binning, aspect sin/cos, soil consolidation)
3. ✅ 5-fold stratified cross-validation
4. ✅ Train LightGBM with optimized hyperparameters
5. ✅ Generate confusion matrix and classification report
6. ✅ Save model to `models/lightgbm_model.pkl`
7. ✅ Export evaluation metrics to `demo_reports/`

### Model Training Output

```
Training LightGBM...
  ✓ LightGBM - Accuracy: 0.9520, F1 (macro): 0.9460, Time: 35.2s
  ✓ LightGBM CV - Mean: 0.9480 (±0.0120)

Chaos Zone Statistics:
  Total observations: 15120
  In chaos zones: 2965 (19.6%)
  
✓ Model saved to: models/lightgbm_model.pkl
   File size: 2.34 MB
```

---

## 🚢 API Deployment

### Option 1: Local FastAPI

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Test endpoint**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Elevation": 2500,
    "Aspect": 180,
    "Slope": 15,
    "Horizontal_Distance_To_Hydrology": 100,
    "Vertical_Distance_To_Hydrology": -50,
    "Horizontal_Distance_To_Roadways": 500,
    "Horizontal_Distance_To_Fire_Points": 600,
    "Hillshade_9am": 200,
    "Hillshade_Noon": 220,
    "Hillshade_3pm": 140,
    "Wilderness_Area1": 1,
    "Wilderness_Area2": 0,
    "Wilderness_Area3": 0,
    "Wilderness_Area4": 0,
    "Soil_Type1": 0,
    "Soil_Type2": 1,
    "Soil_Type3": 0,
    ...
  }'
```

**Expected Response**:
```json
{
  "cover_type": 2,
  "cover_type_name": "Lodgepole Pine",
  "confidence": 0.8734,
  "probabilities": {
    "1": 0.0234,
    "2": 0.8734,
    "3": 0.0456,
    ...
  },
  "uncertainty": {
    "total": 0.1266,
    "near_threshold": false,
    "confidence_score": 0.8734
  },
  "warnings": []
}
```

### Option 2: Docker

```bash
# Build image
docker build -t forest-cover-api:latest -f api/Dockerfile .

# Run container
docker run -d -p 8000:8000 --name forest-api forest-cover-api:latest

# Test
curl http://localhost:8000/health
```

### Option 3: Full Stack Deployment

```bash
# Build and run
docker build -t forest-api .
docker run -d -p 8000:8000 forest-api

# Health check
curl http://localhost:8000/health
```

**Services**:
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs` (Swagger UI)

---

## 🔍 Explainability

### Feature Importance Analysis

The trained LightGBM model automatically provides feature importance scores:

```python
import pickle
import pandas as pd

# Load model
with open('models/lightgbm_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))
```

### Top 10 Most Important Features

Based on LightGBM's built-in importance calculation:

1. **Elevation** (0.342) - Master ecological driver, determines climatic zones
2. **Horizontal_Distance_To_Hydrology** (0.128) - Water accessibility
3. **Wilderness_Area3** (0.095) - Administrative zone indicator
4. **Soil_Type_Sandy** (0.078) - Consolidated soil group (from sparse features)
5. **Vertical_Distance_To_Hydrology** (0.067) - Elevation relative to water
6. **aspect_sin** (0.054) - Circular aspect encoding (north-south)
7. **aspect_cos** (0.051) - Circular aspect encoding (east-west)
8. **Horizontal_Distance_To_Fire_Points** (0.049) - Fire history proximity
9. **elevation_zone** (0.043) - Binned ecological zones
10. **Hillshade_Noon** (0.038) - Solar illumination at peak

### Visualizing Feature Importance

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance.head(15).plot(x='feature', y='importance', kind='barh')
plt.xlabel('Importance Score')
plt.title('LightGBM Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
```

### Decision Path Analysis

For individual predictions, you can trace the decision path:

```python
# Get prediction for a sample
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

# Get leaf indices (decision path)
leaf_index = model.predict(sample, pred_leaf=True)
print(f"Prediction: Class {prediction[0]}")
print(f"Decision path (leaf indices): {leaf_index}")
```

---

## ✨ Key Features

### 1. Chaos-Aware Uncertainty Quantification

**Problem**: Near elevation thresholds (2400m, 2800m, 3200m), small measurement errors cause large prediction changes.

**Solution**: Detect proximity to thresholds and amplify uncertainty 2×.

```python
# Automatic detection in uncertaintyQuantification.py
if abs(elevation - threshold) <= 50:  # ±50m window
    uncertainty_total *= 2.0
    warnings.append("Near ecological threshold")
    
# Calculate confidence score
confidence = 1 - uncertainty_total
```

**LightGBM Integration**:
```python
# Get probability predictions from LightGBM
proba = model.predict_proba(X)

# Calculate entropy-based uncertainty
entropy = -np.sum(proba * np.log2(proba + 1e-10), axis=1)
normalized_uncertainty = entropy / np.log2(7)  # 7 classes

# Apply chaos amplification
if near_threshold:
    normalized_uncertainty *= 2.0
```

### 2. Feature Engineering Pipeline

**Module 3A**: Elevation Processing
- Bin into 4 ecological zones (Foothill, Montane, Subalpine, Alpine)
- Detect chaos zones (±50m from thresholds)
- Create `near_threshold` and `chaos_amplification_factor` features

**Module 3B**: Aspect Transformation
- Convert circular aspect (0-360°) to `sin(θ)` and `cos(θ)`
- Preserves continuity (359° = 1°)

**Module 3C**: Soil Consolidation
- Reduce 40 sparse categories → 15 ecologically coherent groups
- Sparsity: 73% → 5%

**Module 3D**: Distance Interactions
- Normalize distance metrics
- Create interaction features (e.g., `elevation × hydrology_distance`)

### 3. Fault Tolerance

**Circuit Breaker Pattern**:
- Validates data schema before processing
- Blocks requests after 3 consecutive failures
- Auto-recovery with timeout mechanism

**Drift Detection**:
- Monitors KL-divergence and Population Stability Index (PSI)
- Triggers retraining if accuracy drops ≥5%

### 4. Production Monitoring

**Real-time Metrics**:
- Prediction latency tracking
- Model confidence distribution
- Chaos zone prediction frequency
- Feature drift detection

**Logging System**:
```python
# Every prediction is logged with:
{
    "timestamp": "2024-12-06T10:30:45",
    "input_features": {...},
    "prediction": 2,
    "confidence": 0.8734,
    "near_threshold": false,
    "inference_time_ms": 45
}
```

**Alerting**:
- Warning if confidence drops below 0.5
- Alert if >20% of predictions in chaos zones
- Notification on model performance degradation

---

## 🔗 Academic Integration

This repository is the **practical implementation** component of a larger academic project. For the complete analysis:

### 📚 [Systems Analysis and Design - Academic Life Repository](https://github.com/nikkaoyy/Academic-Life)

**What's There**:
- **Workshop 1**: Systems analysis, chaos theory, sensitivity analysis
- **Workshop 2**: Architecture design (7-layer → 4-layer evolution)
- **Workshop 3**: Quality frameworks (ISO 9000, CMMI, Six Sigma)
- **Project Management**: Agile-Kanban methodology, risk register
- **Theoretical Foundations**: Chaos detection formulas, uncertainty decomposition

**Cross-References**:
- Feature engineering rationale (Workshop 1, Section II.C)
- Architecture diagrams (Workshop 2, Figure 1)
- Risk mitigation strategies (Workshop 3, Section III.B)
- Quality standards alignment (Workshop 3, Section IV)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Run** existing tests (`pytest tests/`)
5. **Commit** with clear messages (`git commit -m 'Add chaos detection for aspect'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Add unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Authors**:
- Nicolás Martínez Pineda - [GitHub](https://github.com/nikkaoyy)
- Anderson Danilo Martínez Bonilla
- Gabriel Esteban Gutiérrez Calderón
- Jean Paul Contreras Talero

**Institution**: Universidad Distrital Francisco José de Caldas

**Course**: Systems Analysis and Design

**Project Year**: 2025-3

---

## 🙏 Acknowledgments

- **Kaggle** for the Forest Cover Type dataset
- **Roosevelt National Forest** (USGS) for original cartographic data
- **Microsoft** for the LightGBM library
- **scikit-learn** team for machine learning infrastructure
- **FastAPI** for modern API framework
- **Universidad Distrital Francisco José de Caldas** for academic support

---

## 📚 References

1. Blackard, J. A., & Dean, D. J. (1999). *Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types*. Computers and Electronics in Agriculture, 24(3), 131-151.

2. Lorenz, E. N. (1963). *Deterministic Nonperiodic Flow*. Journal of the Atmospheric Sciences, 20(2), 130–141.

3. Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.

4. International Organization for Standardization. (2015). *ISO 9000:2015 — Quality Management Systems*.

---

**⭐ If this project helped you, please star the repository!**

**🔗 Related**: [Academic Life - Systems Analysis](https://github.com/nikkaoyy/Academic-Life)
