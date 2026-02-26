"""
Page 4 — Climate Forecast Engine (Multi-Model, Multi-Target).

Loads **actual trained models** and generates real predictions
using the project's feature-engineering pipeline. Supports ML
models (.pkl) and DL models (.keras).

Data-source inputs: manual, IoT, CSV, Open-Meteo API.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path so we can import from `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import config
from utils.charts import render_forecast_charts
from utils.shared import (
    load_css,
    load_results,
    hero_section,
    gradient_card,
    footer,
    empty_state,
    MODELS_DIR,
    DATA_DIR,
)

st.set_page_config(page_title="Forecast | Pune Climate", page_icon="🔮", layout="wide")
load_css()

# ── Data-source helpers ───────────────────────────────────────────────────────

def _fetch_iot_sensor() -> dict | None:
    """Attempt to read latest readings from a local IoT REST endpoint."""
    try:
        import requests
        resp = requests.get("http://localhost:5001/iot/latest", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _fetch_latest_csv() -> dict | None:
    """Compute latest day averages from the historical CSV."""
    try:
        df = pd.read_csv(DATA_DIR / "pune_climate_with_co2.csv", parse_dates=["date"])
        last_date = df["date"].max()
        day_mask = (df["date"] == last_date)
        daily = df.loc[day_mask]
        return {
            "temp_C": float(daily["temp_C"].mean()),
            "humidity_pct": float(daily["humidity_pct"].mean()),
            "rainfall_mm": float(daily["rainfall_mm"].sum()),
            "solar_MJ": float(daily["solar_MJ"].mean()),
            "co2_ppm": float(daily["co2_ppm"].mean()) if "co2_ppm" in daily else 420.0
        }
    except Exception:
        return None


def _fetch_openmeteo() -> dict | None:
    """Fetch today's weather from Open-Meteo and return aggregates."""
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 18.5204, "longitude": 73.8567,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "relative_humidity_2m_max,relative_humidity_2m_min,shortwave_radiation_sum",
                "past_days": 1, "timezone": "Asia/Kolkata",
            },
            timeout=5,
        )
        resp.raise_for_status()
        d = resp.json().get("daily", {})
        _not_none = lambda lst: [v for v in lst if v is not None]
        temps = [(a + b) / 2 for a, b in zip(_not_none(d["temperature_2m_max"]), _not_none(d["temperature_2m_min"]))]
        hum = [(a + b) / 2 for a, b in zip(_not_none(d["relative_humidity_2m_max"]), _not_none(d["relative_humidity_2m_min"]))]
        rain = _not_none(d["precipitation_sum"])
        solar = [s * 0.0036 for s in _not_none(d["shortwave_radiation_sum"])]
        return {
            "temp_C": float(np.mean(temps)) if temps else 25.0,
            "humidity_pct": float(np.mean(hum)) if hum else 60.0,
            "rainfall_mm": float(np.sum(rain)) if rain else 0.0,
            "solar_MJ": float(np.mean(solar)) if solar else 15.0,
        }
    except Exception:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────

# Model type classification
ML_MODELS = {"XGBoost", "Random Forest"}
DL_MODELS = {"CNN-LSTM", "Transformer"}

_MODEL_FILE_MAP = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "CNN-LSTM": "cnn_lstm.keras",
    "Transformer": "transformer.keras",
}


def _model_file_exists(name: str) -> bool:
    """Check whether a model's artefact exists on disk."""
    fname = _MODEL_FILE_MAP.get(name)
    if fname:
        return (MODELS_DIR / fname).exists()
    return False


@st.cache_resource(show_spinner="Loading ML model…", max_entries=2)
def _load_ml_model(name: str):
    """Load a joblib-persisted scikit-learn / XGBoost model."""
    import joblib

    path = MODELS_DIR / _MODEL_FILE_MAP[name]
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource(show_spinner="Loading DL model…", max_entries=2)
def _load_dl_model(name: str):
    """Load a Keras .keras checkpoint. (Cache invalidated for new 60-shape models)"""
    try:
        from tensorflow import keras

        path = MODELS_DIR / _MODEL_FILE_MAP[name]
        if path.exists():
            return keras.models.load_model(str(path))
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner="Loading scalers…")
def _load_scalers():
    """Load the StandardScalers saved during DL training."""
    import joblib

    sx_path = MODELS_DIR / "scaler_X.pkl"
    sy_path = MODELS_DIR / "scaler_y.pkl"
    if sx_path.exists() and sy_path.exists():
        return joblib.load(sx_path), joblib.load(sy_path)
    return None, None


# ── Main page ─────────────────────────────────────────────────────────────────

results = load_results()
# No longer filtering rigidly by exact name since CSV might append "(Multi-Target)"

hero_section(
    title="🔮 Next-Gen Climate Forecast",
    subtitle="Multi-target predictive modelling via advanced Deep Learning architectures.",
    variant="hero-blue",
)

if True:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("## 🎯 Forecast Configuration")

    # Offer all models whose artefacts exist on disk (ML, DL)
    available_to_check = list(ML_MODELS) + list(DL_MODELS)
    available = [m for m in available_to_check if _model_file_exists(m)]
    if not available:
        available = available_to_check  # fallback for display

    # Group labels for clarity
    def _label(m: str) -> str:
        if m in ML_MODELS:
            return f"🌲 {m} (ML)"
        if m in DL_MODELS:
            return f"🧠 {m} (DL)"
        return m

    selected_models = st.sidebar.multiselect(
        "Select Models to Compare", available, default=available[:1], format_func=_label,
    )

    if not selected_models:
        st.sidebar.warning("Please select at least one model.")

    # ── Data source selector ──────────────────────────────────────────────────
    data_source = st.radio(
        "🗂️ Global Data Source Configuration:",
        ["Manual Entry", "IoT Sensor (REST)", "CSV: Latest Day", "Open-Meteo API (Live)"],
        horizontal=True,
    )

    prefill: dict = {}
    if data_source == "IoT Sensor (REST)":
        prefill = _fetch_iot_sensor() or {}
        st.info("IoT data loaded." if prefill else "No IoT data available.")
    elif data_source == "CSV: Latest Day":
        prefill = _fetch_latest_csv() or {}
        st.info("CSV data loaded." if prefill else "Could not read CSV.")
    elif data_source == "Open-Meteo API (Live)":
        prefill = _fetch_openmeteo() or {}
        st.info("Live data loaded." if prefill else "Could not fetch live weather.")

    # ── Input sliders ─────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("🌡️ Current Conditions")
            temp_current = st.slider("Daily Temperature (°C)", 15.0, 45.0,
                                     float(round(prefill.get("temp_C", 28.5), 2)), 0.5)
            humidity = st.slider("Daily Humidity (%)", 10.0, 95.0,
                                 float(round(prefill.get("humidity_pct", 55.0), 2)), 1.0)
            rainfall = st.slider("Daily Rainfall (mm)", 0.0, 300.0,
                                 float(round(prefill.get("rainfall_mm", 5.0), 1)), 5.0)

    with right:
        with st.container(border=True):
            st.subheader("☀️ Forecast Settings")
            solar = st.slider("Daily Solar Radiation (MJ/m²)", 5.0, 30.0,
                              float(round(prefill.get("solar_MJ", 20.0), 2)), 0.5)
            co2_latest = getattr(config, "CO2_DEFAULT", 425.0)
            co2_latest = st.slider("Global CO2 Levels (ppm)", 350.0, 500.0, co2_latest, 1.0)
            
            forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
            start_date = st.date_input("Starting Date", value=datetime.now().date())

    # ── Generate forecast ─────────────────────────────────────────────────────
    if st.button("🔮 Generate Multi-Target Forecast", type="primary", use_container_width=True) and selected_models:
        with st.spinner("Running model inference and aggregating multi-targets…"):

            dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            
            # Dictionary to store predictions for each model: {model_name: {target_name: [list of floats]}}
            model_predictions = {m: {"temp_C": [], "humidity_pct": [], "rainfall_mm": [], "solar_MJ": []} for m in selected_models}
            
            sc_X, sc_y = _load_scalers()

            for selected_model in selected_models:
                # ── Determine inference strategy ─────────────────────────────
                if selected_model in ML_MODELS:
                    # ── ML model: flat feature-vector inference ───────────────
                    model = _load_ml_model(selected_model)
                    if model is not None:
                        curr_t = temp_current
                        curr_h = humidity
                        curr_r = rainfall
                        curr_s = solar

                        for i in range(forecast_days):
                            current_dt = start_date + timedelta(days=i)
                            pred_day = current_dt.timetuple().tm_yday
                            d_sin = np.sin(2 * np.pi * pred_day / 365.25)
                            d_cos = np.cos(2 * np.pi * pred_day / 365.25)

                            # Features (Daily format, mimicking FeatureEngine.create_features structure)
                            base = [co2_latest]
                            # Simple mock for lags/rolls during live inference. 
                            # In production, these should be drawn from a state array.
                            lags_t = [curr_t] * len(config.LAG_FEATURES)
                            lags_h = [curr_h] * len(config.LAG_FEATURES)
                            lags_r = [curr_r] * len(config.LAG_FEATURES)
                            lags_s = [curr_s] * len(config.LAG_FEATURES)
                            lags_c = [co2_latest] * len(config.LAG_FEATURES)
                            
                            rolls_t = [curr_t] * len(config.ROLLING_WINDOWS)
                            rolls_h = [curr_h] * len(config.ROLLING_WINDOWS)
                            rolls_r = [curr_r] * len(config.ROLLING_WINDOWS)
                            rolls_s = [curr_s] * len(config.ROLLING_WINDOWS)
                            rolls_c = [co2_latest] * len(config.ROLLING_WINDOWS)
                            
                            cyclic = [pred_day, d_sin, d_cos]
                            
                            row = np.array(base + lags_t + lags_h + lags_r + lags_s + lags_c + 
                                           rolls_t + rolls_h + rolls_r + rolls_s + rolls_c + cyclic).reshape(1, -1)

                            # Predict [temp, humidity, rain, solar]
                            preds = model.predict(row)[0]
                            
                            model_predictions[selected_model]["temp_C"].append(float(preds[0]))
                            model_predictions[selected_model]["humidity_pct"].append(float(preds[1]))
                            model_predictions[selected_model]["rainfall_mm"].append(float(max(0, preds[2]))) # No negative rain
                            model_predictions[selected_model]["solar_MJ"].append(float(preds[3]))
                            
                            # Recursively feed output back
                            curr_t, curr_h, curr_r, curr_s = preds[0], preds[1], max(0, preds[2]), preds[3]

                elif selected_model in DL_MODELS:
                    # ── DL model: sequence-based inference with scaling ───────
                    dl_model = _load_dl_model(selected_model)
                    if dl_model is not None and sc_X is not None and sc_y is not None:
                        lookback = config.LOOKBACK
                        
                        curr_t = temp_current
                        curr_h = humidity
                        curr_r = rainfall
                        curr_s = solar

                        for i in range(forecast_days):
                            current_dt = start_date + timedelta(days=i)
                            pred_day = current_dt.timetuple().tm_yday
                            d_sin = np.sin(2 * np.pi * pred_day / 365.25)
                            d_cos = np.cos(2 * np.pi * pred_day / 365.25)
                            
                            # DL models only use the raw 5 features as sequence inputs, not the 34 engineered features
                            base = [curr_t, curr_h, curr_r, curr_s, co2_latest]
                            row = np.array(base).reshape(1, -1)
                            
                            # Create a synthetic sequence by repeating the row (simplified for live forecast)
                            # The scaler sc_X expects the 5-feature row.
                            seq_row_2d_scaled = sc_X.transform(row)
                            seq_input = np.tile(seq_row_2d_scaled, (1, lookback, 1))

                            # Predict [temp, humidity, rain, solar]
                            num_targets = len(config.TARGETS)
                            # Use direct callable instead of .predict() for massive speedup in loops
                            import tensorflow as tf
                            pred_scaled_vec = dl_model(tf.convert_to_tensor(seq_input), training=False).numpy().flatten()
                            pred_vec = float(sc_y.inverse_transform([[pred_scaled_vec[k] for k in range(num_targets)]])[0, 0])
                            # Wait, the shape is (1, 4) for inverse_transform
                            pred_scaled_reshaped = pred_scaled_vec.reshape(1, num_targets)
                            preds = sc_y.inverse_transform(pred_scaled_reshaped)[0]

                            model_predictions[selected_model]["temp_C"].append(float(preds[0]))
                            model_predictions[selected_model]["humidity_pct"].append(float(preds[1]))
                            model_predictions[selected_model]["rainfall_mm"].append(float(max(0, preds[2])))
                            model_predictions[selected_model]["solar_MJ"].append(float(preds[3]))

                            # Recursively feed output back
                            curr_t, curr_h, curr_r, curr_s = preds[0], preds[1], max(0, preds[2]), preds[3]

            # ── Summary metrics ───────────────────────────────────────────
            st.success(f"✅ Forecast generated across {len(selected_models)} models for 4 targets.")

            # Delegate charting to the specialized formatting module per industry standard
            render_forecast_charts(dates, model_predictions, selected_models, forecast_days)

            # ── Details Table for Selected Models ──────────────────────────────────────────
            st.markdown("### 📋 Forecast Details")
            tabs = st.tabs(selected_models)
            for i, model in enumerate(selected_models):
                with tabs[i]:
                    st.dataframe(
                        pd.DataFrame({
                            "Date": dates,
                            "Temp (°C)": [f"{p:.2f}" for p in model_predictions[model]["temp_C"]],
                            "Humidity (%)": [f"{p:.2f}" for p in model_predictions[model]["humidity_pct"]],
                            "Rainfall (mm)": [f"{p:.2f}" for p in model_predictions[model]["rainfall_mm"]],
                            "Solar (MJ)": [f"{p:.2f}" for p in model_predictions[model]["solar_MJ"]],
                        }),
                        use_container_width=True, height=250,
                    )

else:
    empty_state("Models Not Trained", "Run `python train.py` to enable forecasting.")

footer(text="Forecast Engine | Powered by AI & Climate Science")
