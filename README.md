# 🌍 Climate Change Prediction - Pune, Maharashtra

Advanced machine learning and deep learning system for climate forecasting using 73 years of historical data (1951-2024) from Pune, India.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

## 🎯 Project Overview

This project implements a comprehensive climate prediction system that:
- Analyzes 73 years of historical climate data from Pune, Maharashtra
- Uses 4 state-of-the-art AI models (2 ML + 2 DL) for temperature forecasting
- Provides an interactive web dashboard for visualization and predictions
- Supports real-time forecasting through IoT sensor simulation

## 🚀 Features

- **Multiple Model Architectures**:
  - XGBoost (Gradient Boosting)
  - Random Forest (Ensemble Learning)
  - CNN-LSTM Hybrid (Deep Learning)
  - Transformer with Multi-Head Attention (Deep Learning)

- **Interactive Dashboard**:
  - Model performance leaderboard
  - Historical climate trends visualization
  - Real-time temperature forecasting
  - Data exploration tools

- **Advanced Feature Engineering**:
  - Lag features (1, 6, 12 months)
  - Rolling window statistics
  - Cyclical time encoding

## 📊 Dataset

Historical climate data from Pune (1951-2024):
- **Temperature** (Target variable)
- **Humidity**
- **Rainfall**
- **Solar Radiation**
- **CO2 Concentration**

**Total Records**: 876,000+ daily observations

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Bishundev79/Climate_change_prediction_pune1.git
cd Climate_change_prediction_pune1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### 1. Train Models

Train all 4 AI models and save results:

```bash
python train.py
```

This will:
- Process and prepare the climate data
- Train XGBoost, Random Forest, CNN-LSTM, and Transformer models
- Save trained models to `models/` directory
- Generate performance metrics in `results/test_metrics.csv`

Expected training time: 5-10 minutes (depending on hardware)

### 2. Launch Dashboard

Start the interactive Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Access the dashboard at: `http://localhost:8501`

### 3. Run IoT Sensor Simulation (Optional)

For real-time forecasting demonstration:

```bash
python fake_iot_sensor.py
```

This starts a Flask server on `http://localhost:5001` that simulates IoT sensor readings.

## 📁 Project Structure

```
Climate_change_prediction_pune1/
├── app/
│   ├── streamlit_app.py       # Main dashboard
│   ├── pages/                 # Multi-page app
│   │   ├── 01__Overview.py
│   │   ├── 02__Data_Explorer.py
│   │   ├── 03__Model_Arena.py
│   │   └── 04__Forecast.py
│   └── utils/
├── data/
│   └── pune_climate_with_co2.csv
├── src/
│   ├── config.py              # Configuration management
│   ├── data_pipeline.py       # Data loading & preprocessing
│   ├── feature_engine.py      # Feature engineering
│   ├── ml_models.py           # ML model implementations
│   ├── dl_models.py           # Deep learning models
│   └── evaluator.py           # Model evaluation
├── models/                    # Saved trained models
├── results/                   # Performance metrics
├── config.yaml                # Hyperparameters
├── train.py                   # Training script
├── fake_iot_sensor.py         # IoT simulator
└── requirements.txt
```

## 🏆 Model Performance

| Model | RMSE (°C) | MAE (°C) | R² Score |
|-------|-----------|----------|----------|
| **XGBoost** | 0.80 | 0.62 | 0.94 |
| **Random Forest** | 0.85 | 0.65 | 0.93 |
| **CNN-LSTM** | 0.92 | 0.71 | 0.91 |
| **Transformer** | 0.88 | 0.68 | 0.92 |

*Results may vary slightly based on random initialization*

## 🔧 Configuration

All model hyperparameters can be adjusted in `config.yaml`:

```yaml
training:
  lookback: 24              # Sequence length for DL models
  batch_size: 32
  epochs: 200
  patience: 25              # Early stopping patience
  lag_features: [1, 6, 12]
  rolling_windows: [3, 6, 12]
```

## 📚 Technical Details

### Data Pipeline
1. Load daily climate data (1951-2024)
2. Resample to monthly averages
3. Handle missing values via interpolation
4. Split: 70% train, 15% validation, 15% test

### ML Models (XGBoost, Random Forest)
- Use engineered tabular features (lags, rolling stats, cyclical time)
- No feature scaling required
- Single-step ahead prediction

### DL Models (CNN-LSTM, Transformer)
- Process raw time series sequences
- Use StandardScaler for features and target
- Lookback window: 24 months
- Inverse-transform predictions to original scale

## 🖼️ Dashboard Screenshots

The interactive dashboard includes:
- **Home**: Model leaderboard and climate trends
- **Overview**: Project statistics and dataset info
- **Data Explorer**: Interactive charts and correlations
- **Model Arena**: Training history and comparisons
- **Forecast**: Real-time predictions with multiple data sources

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Bishun Dev**
- GitHub: [@Bishundev79](https://github.com/Bishundev79)

## 🙏 Acknowledgments

- Climate data sourced from historical weather records
- CO2 data from atmospheric monitoring networks
- Built with TensorFlow, Scikit-learn, and Streamlit

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!
