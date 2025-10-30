# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Climate change prediction system for Pune, Maharashtra using ML and DL models to forecast temperature based on historical climate data (1951-2024). The system includes:
- 2 traditional ML models (XGBoost, Random Forest)
- 2 deep learning models (CNN-LSTM Hybrid, Transformer with Attention)
- Interactive Streamlit dashboard for visualization and forecasting
- Real-time IoT sensor simulation for live forecasting

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if using .venv)
source .venv/bin/activate

# Or if using venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train all models (ML + DL)
python train.py

# Results are saved to:
# - models/ directory (.pkl for ML, .keras for DL)
# - results/test_metrics.csv (evaluation metrics)
```

### Running the Dashboard
```bash
# Start main Streamlit app
streamlit run app/streamlit_app.py

# The app will be available at http://localhost:8501
```

### IoT Sensor Simulation
```bash
# Start fake IoT sensor endpoint (for forecast page)
python fake_iot_sensor.py

# Runs Flask server on port 5001
# Endpoint: http://localhost:5001/iot/latest
```

## Architecture

### Data Pipeline Flow
1. **DataPipeline** (`src/data_pipeline.py`): Loads daily data from CSV → resamples to monthly averages → splits into train/val/test (70/15/15)
2. **FeatureEngine** (`src/feature_engine.py`): Creates lag features, rolling window features, and cyclical time features (month_sin/month_cos)
3. **Models**: Train on engineered features with different approaches
4. **Evaluator** (`src/evaluator.py`): Computes RMSE, MAE, R² metrics

### Model Architecture Differences

**ML Models** (XGBoost, Random Forest):
- Use engineered tabular features (lags, rolling windows, cyclical encodings)
- No scaling required
- Predict single-step ahead on tabular data

**DL Models** (CNN-LSTM, Transformer):
- Use raw time series sequences (via `SequenceGenerator`)
- Require StandardScaler for both features and target
- Input shape: (batch_size, lookback=24, n_features=5)
- Predictions must be inverse-transformed from scaled space

### Key Configuration
All hyperparameters are centralized in `config.yaml`:
- `lookback: 24` - sequence length for DL models
- `lag_features: [1, 6, 12]` - months to look back for lag features
- `rolling_windows: [3, 6, 12]` - window sizes for rolling averages
- `epochs: 200` with `patience: 25` early stopping

Config is loaded via `src/config.py` and accessed throughout codebase as `from src.config import config`.

### Streamlit App Structure
Multi-page app with navigation in `app/streamlit_app.py`:
- **Home**: Model leaderboard and climate trends overview
- **01__Overview.py**: Project details and data statistics
- **02__Data_Explorer.py**: Interactive data visualization
- **03__Model_Arena.py**: Model comparison and training history
- **04__Forecast.py**: Real-time forecasting with multiple data sources (IoT sensor, CSV, Open-Meteo API)

### Data Format
CSV at `data/pune_climate_with_co2.csv` with columns:
- `date` (datetime): Daily records from 1951-2024
- `temp_C`: Temperature (target variable)
- `humidity_pct`: Relative humidity
- `rainfall_mm`: Daily rainfall
- `solar_MJ`: Solar radiation
- `co2_ppm`: CO2 concentration

Note: Data contains missing values that are handled via monthly resampling and interpolation in DataPipeline.

## Development Patterns

### Adding a New Model
1. Create model class in `src/ml_models.py` or `src/dl_models.py`
2. Implement `train()`, `predict()`, and `save()` methods
3. Set `self.name` for display in results
4. Add training logic to `train.py` (use appropriate preprocessing)
5. Update `config.yaml` with model hyperparameters if needed

### Scaling Gotcha for DL Models
Deep learning models in this project require careful scaling:
- Fit scalers on training data only: `sc_X.fit_transform(X_train_raw)`
- Transform validation/test with same scaler: `sc_X.transform(X_val_raw)`
- Use separate scaler for target variable (`sc_y`)
- Always inverse-transform predictions before evaluation: `sc_y.inverse_transform(predictions)`

### Model Persistence
- ML models: Save with `joblib.dump()` as `.pkl` files
- DL models: Use Keras native `model.save()` as `.keras` files (TensorFlow 2.x format)

## Testing

No formal test suite is currently implemented. Manual verification workflow:
1. Run `python train.py` to train all models
2. Check `results/test_metrics.csv` for reasonable RMSE values (~0.8-1.5°C)
3. Start Streamlit app and verify visualizations render correctly
4. Test forecast page with and without IoT sensor running
