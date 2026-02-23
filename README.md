# 🌍 Climate Change Prediction — Pune, Maharashtra

Advanced ML & Deep Learning system for climate forecasting using 73 years of historical data (1951–2024) from Pune, India.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

A comprehensive climate prediction system that:

- Analyses 73 years of historical climate data from Pune, Maharashtra
- Trains **6 AI models** (2 ML + 2 DL + 2 Ensembles) for temperature forecasting
- Provides an interactive multi-page Streamlit dashboard with real-time predictions
- Supports 4 data-source inputs: Manual, IoT Sensor, CSV, and live Open-Meteo API
- Fully **config-driven** — all hyper-parameters live in `config.yaml`, zero code edits required

## 🚀 Features

### Model Architectures
| Type | Model | Technique |
|------|-------|-----------|
| ML | **XGBoost** | Gradient Boosting |
| ML | **Random Forest** | Bagging Ensemble |
| DL | **CNN-LSTM Hybrid** | 1-D Conv → LSTM |
| DL | **Transformer** | Multi-Head Self-Attention |
| Ensemble | **ML Ensemble** | Average of XGBoost + Random Forest |
| Ensemble | **DL Ensemble** | Average of CNN-LSTM + Transformer |

### Interactive Dashboard (6 pages)
| Page | Description |
|------|-------------|
| **Home** | Model leaderboard, best-model banner, temperature trend chart |
| **Overview** | Project statistics, dataset summary, model cards |
| **Data Explorer** | Interactive time-series charts, correlations, distributions |
| **Model Arena** | Side-by-side model comparison with metrics |
| **Forecast** | Real predictions using trained models with confidence intervals |
| **Benchmark** | RMSE / MAE / R² comparison across all 6 models |

### Advanced Feature Engineering
- Lag features (1, 6, 12 months)
- Rolling-window statistics (3, 6, 12 months)
- Cyclical month encoding (sin/cos)
- CO₂ concentration as exogenous feature

## 📊 Dataset

Historical climate data from Pune (1951–2024):

| Feature | Description |
|---------|-------------|
| `temp_C` | Average temperature (°C) — **target variable** |
| `humidity_pct` | Relative humidity (%) |
| `rainfall_mm` | Monthly rainfall (mm) |
| `solar_MJ` | Solar radiation (MJ/m²) |
| `co2_ppm` | Atmospheric CO₂ (ppm) |

**Records**: ~27,000 daily observations → resampled to monthly

## 🛠️ Installation

### Prerequisites
- **Python 3.11** (recommended for TensorFlow compatibility)
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Bishundev79/Climate_change_prediction_pune1.git
cd Climate_change_prediction_pune1

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# macOS Apple Silicon (M1/M2/M3) — use the optimised requirements:
pip install -r requirements-macos.txt
```

## 🎮 Usage

### 1. Train Models

Train all 6 AI models and save results:

```bash
python train.py
```

This will:
- Process and prepare the climate data (monthly resampling, imputation)
- Train XGBoost, Random Forest, CNN-LSTM, and Transformer models
- Compute ML Ensemble and DL Ensemble predictions
- Save trained models to `models/`
- Generate performance metrics in `results/test_metrics.csv`

Expected training time: ~5–10 minutes (depends on hardware)

### 2. Launch Dashboard

Start the interactive Streamlit web application:

```bash
cd app && streamlit run streamlit_app.py
```

Access the dashboard at: **http://localhost:8501**

### 3. Run IoT Sensor Simulation (Optional)

For real-time forecasting demonstration:

```bash
python fake_iot_sensor.py
```

Starts a Flask server on **http://localhost:5001** that simulates live IoT sensor readings. The Forecast page fetches data from this endpoint when "IoT Sensor (REST)" is selected.

## 📁 Project Structure

```
Climate_change_prediction_pune1/
├── app/
│   ├── streamlit_app.py          # Main dashboard (Home page)
│   ├── pages/
│   │   ├── 01__Overview.py       # Project statistics
│   │   ├── 02__Data_Explorer.py  # Interactive charts
│   │   ├── 03__Model_Arena.py    # Model comparison
│   │   ├── 04__Forecast.py       # Real-time predictions
│   │   └── 05__Benchmark.py      # Performance benchmarks
│   ├── static/
│   │   └── styles.css            # Centralised CSS (zero inline styles)
│   └── utils/
│       └── shared.py             # Reusable UI components & helpers
├── data/
│   └── pune_climate_with_co2.csv # 73 years of climate data
├── src/
│   ├── config.py                 # YAML-driven configuration singleton
│   ├── data_pipeline.py          # Data loading, resampling, splitting
│   ├── feature_engine.py         # Lag, rolling, cyclical features
│   ├── ml_models.py              # XGBoost + Random Forest wrappers
│   ├── dl_models.py              # CNN-LSTM + Transformer wrappers
│   ├── evaluator.py              # RMSE, MAE, R² evaluation
│   └── logger.py                 # Centralised logging
├── models/                       # Saved model artefacts (.pkl, .keras)
├── results/                      # Metrics CSVs & feature importances
├── config.yaml                   # All hyper-parameters (single source of truth)
├── train.py                      # Training orchestration script
├── fake_iot_sensor.py            # IoT REST API simulator (Flask)
├── requirements.txt              # Dependencies (Linux / generic)
└── requirements-macos.txt        # Dependencies (Apple Silicon)
```

## 🏆 Model Performance

| Model | RMSE (°C) | MAE (°C) | R² Score |
|-------|-----------|----------|----------|
| **XGBoost** | 0.8049 | 0.5945 | 0.9328 |
| **ML Ensemble** | 0.8063 | 0.5782 | 0.9326 |
| **Random Forest** | 0.8410 | 0.5820 | 0.9267 |
| **CNN-LSTM** | 1.3336 | 1.0214 | 0.8173 |
| **DL Ensemble** | 2.0970 | 1.6321 | 0.5483 |
| **Transformer** | 3.1176 | 2.4649 | 0.0016 |

> **Note**: ML models significantly outperform DL models on this dataset due to the relatively small sample size (~800 monthly records after resampling). The Transformer's poor R² demonstrates that complex architectures aren't always better — a finding worth discussing.

## 🔧 Configuration

All hyper-parameters are managed in `config.yaml` — no code edits needed:

```yaml
training:
  lookback: 24              # Months of history for DL sequence input
  batch_size: 32
  epochs: 200
  patience: 25              # Early stopping patience (epochs)
  lag_features: [1, 6, 12]  # Lag offsets in months
  rolling_windows: [3, 6, 12]

models:
  xgboost:
    n_estimators: 500
    max_depth: 8
    learning_rate: 0.05
  # ... see config.yaml for full list
```

## 📚 Technical Details

### Data Pipeline
1. Load daily climate CSV (1951–2024)
2. Resample to monthly frequency (aggregation per feature)
3. Impute missing values via linear interpolation + back-fill
4. Chronological split: 70% train / 15% validation / 15% test

### ML Models (XGBoost, Random Forest)
- Trained on engineered tabular features (lags, rolling stats, cyclical month)
- Saved as `.pkl` via joblib — fast inference
- Config-driven hyper-parameters

### DL Models (CNN-LSTM, Transformer)
- Trained on raw scaled time-series sequences (lookback = 24 months)
- StandardScaler for features and target (scalers saved for inference)
- Saved as `.keras` checkpoints
- Predictions inverse-transformed to original scale

### Ensembles
- **ML Ensemble**: Simple average of XGBoost + Random Forest predictions
- **DL Ensemble**: Simple average of CNN-LSTM + Transformer predictions
- Computed at training time and available for live inference in the Forecast page

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Bishun Dev**
- GitHub: [@Bishundev79](https://github.com/Bishundev79)

## 🙏 Acknowledgments

- Climate data sourced from historical weather records for Pune, Maharashtra
- CO₂ data from atmospheric monitoring networks
- Built with TensorFlow, Scikit-learn, XGBoost, and Streamlit

---

⭐ If you find this project useful, please consider giving it a star!
