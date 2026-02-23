"""
Page 4 — Climate Forecast Engine.

Loads **actual trained models** and generates real predictions
using the project's feature-engineering pipeline.  Supports ML
models (.pkl), DL models (.keras), and live ensemble inference.

Data-source inputs: manual, IoT, CSV, Open-Meteo API.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    """Compute latest-month averages from the historical CSV."""
    try:
        df = pd.read_csv(DATA_DIR / "pune_climate_with_co2.csv", parse_dates=["date"])
        last_date = df["date"].max()
        month_mask = (df["date"].dt.month == last_date.month) & (df["date"].dt.year == last_date.year)
        monthly = df.loc[month_mask]
        return {
            "temp_C": float(monthly["temp_C"].mean()),
            "humidity_pct": float(monthly["humidity_pct"].mean()),
            "rainfall_mm": float(monthly["rainfall_mm"].sum()),
            "solar_MJ": float(monthly["solar_MJ"].mean()),
        }
    except Exception:
        return None


def _fetch_openmeteo() -> dict | None:
    """Fetch last 31 days from Open-Meteo and return monthly aggregates."""
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 18.5204, "longitude": 73.8567,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "relative_humidity_2m_max,relative_humidity_2m_min,shortwave_radiation_sum",
                "past_days": 31, "timezone": "Asia/Kolkata",
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
            "temp_C": float(np.mean(temps)),
            "humidity_pct": float(np.mean(hum)),
            "rainfall_mm": float(np.sum(rain)),
            "solar_MJ": float(np.mean(solar)),
        }
    except Exception:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────

# Model type classification
ML_MODELS = {"XGBoost", "Random Forest"}
DL_MODELS = {"CNN-LSTM", "Transformer"}
ENSEMBLE_MAP = {
    "ML Ensemble": ["XGBoost", "Random Forest"],
    "DL Ensemble": ["CNN-LSTM", "Transformer"],
}

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
    # Ensembles have no file — they need their constituents
    if name in ENSEMBLE_MAP:
        return all(_model_file_exists(m) for m in ENSEMBLE_MAP[name])
    return False


@st.cache_resource(show_spinner="Loading ML model…")
def _load_ml_model(name: str):
    """Load a joblib-persisted scikit-learn / XGBoost model."""
    import joblib

    path = MODELS_DIR / _MODEL_FILE_MAP[name]
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource(show_spinner="Loading DL model…")
def _load_dl_model(name: str):
    """Load a Keras .keras checkpoint."""
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

hero_section(
    title="🔮 Climate Forecast Engine",
    subtitle="Predict future climate patterns with trained AI models",
    variant="hero-red",
)

st.markdown(
    "<div class='info-box'><p>"
    "<strong>🎯 Data Sources:</strong> Choose your starting conditions.<br>"
    "• <strong>Manual</strong> — set values yourself<br>"
    "• <strong>IoT</strong> — fetch from sensor API<br>"
    "• <strong>CSV</strong> — latest month from historical data<br>"
    "• <strong>Open-Meteo</strong> — real-time monthly averages for Pune"
    "</p></div>",
    unsafe_allow_html=True,
)

if not results.empty:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("## 🎯 Forecast Configuration")

    # Offer all models whose artefacts exist on disk (ML, DL, ensembles)
    available = [m for m in results["Model"].tolist() if _model_file_exists(m)]
    if not available:
        available = results["Model"].tolist()  # fallback for display

    # Group labels for clarity
    def _label(m: str) -> str:
        if m in ML_MODELS:
            return f"🌲 {m} (ML)"
        if m in DL_MODELS:
            return f"🧠 {m} (Deep Learning)"
        if m in ENSEMBLE_MAP:
            return f"🔗 {m}"
        return m

    selected_model = st.sidebar.selectbox(
        "Select Model", available, format_func=_label,
    )
    model_rmse = float(results.loc[results["Model"] == selected_model, "RMSE"].values[0])
    st.sidebar.metric("RMSE", f"{model_rmse:.4f}°C")

    # Show model type info
    if selected_model in ENSEMBLE_MAP:
        members = ", ".join(ENSEMBLE_MAP[selected_model])
        st.sidebar.info(f"Ensemble = average of {members}")

    # ── Data source selector ──────────────────────────────────────────────────
    st.markdown("#### 🗂️ Data Source")
    data_source = st.radio(
        "Set input values using:",
        ["Manual Entry", "IoT Sensor (REST)", "CSV: Latest Month", "Open-Meteo API (Live)"],
        horizontal=True,
    )

    prefill: dict = {}
    if data_source == "IoT Sensor (REST)":
        prefill = _fetch_iot_sensor() or {}
        st.info("IoT data loaded." if prefill else "No IoT data available.")
    elif data_source == "CSV: Latest Month":
        prefill = _fetch_latest_csv() or {}
        st.info("CSV data loaded." if prefill else "Could not read CSV.")
    elif data_source == "Open-Meteo API (Live)":
        prefill = _fetch_openmeteo() or {}
        st.info("Live data loaded." if prefill else "Could not fetch live weather.")

    # ── Input sliders ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Input Parameters")
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='forecast-card'><h3>🌡️ Current Conditions</h3>", unsafe_allow_html=True)
        temp_current = st.slider("Avg Monthly Temperature (°C)", 15.0, 35.0,
                                 float(round(prefill.get("temp_C", 24.5), 2)), 0.5)
        humidity = st.slider("Avg Monthly Humidity (%)", 20.0, 95.0,
                             float(round(prefill.get("humidity_pct", 65.0), 2)), 1.0)
        rainfall = st.slider("Monthly Rainfall (mm)", 0.0, 500.0,
                             float(round(prefill.get("rainfall_mm", 50.0), 1)), 10.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='forecast-card'><h3>☀️ Forecast Settings</h3>", unsafe_allow_html=True)
        solar = st.slider("Avg Solar Radiation (MJ/m²)", 10.0, 25.0,
                          float(round(prefill.get("solar_MJ", 18.0), 2)), 0.5)
        forecast_months = st.slider("Forecast Horizon (months)", 1, 12, 6)
        start_month = st.selectbox(
            "Starting Month", list(range(1, 13)),
            index=datetime.now().month - 1,
            format_func=lambda m: datetime(2024, m, 1).strftime("%B"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Generate forecast ─────────────────────────────────────────────────────
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Running model inference…"):

            predictions: list[float] = []
            dates: list[str] = []
            inference_label = selected_model
            model_loaded = False

            # ── Determine inference strategy ─────────────────────────────
            if selected_model in ML_MODELS:
                # ── ML model: flat feature-vector inference ───────────────
                model = _load_ml_model(selected_model)
                if model is not None:
                    model_loaded = True
                    co2_latest = 420.0
                    current_temp = temp_current

                    for i in range(forecast_months):
                        pred_month = (start_month + i - 1) % 12 + 1
                        m_sin = np.sin(2 * np.pi * pred_month / 12)
                        m_cos = np.cos(2 * np.pi * pred_month / 12)

                        base = [humidity, rainfall, solar, co2_latest]
                        lags = [current_temp] * 3 + [humidity] * 3 + [rainfall] * 3 + [solar] * 3 + [co2_latest] * 3
                        rolls = [current_temp] * 3 + [humidity] * 3 + [rainfall] * 3 + [solar] * 3 + [co2_latest] * 3
                        cyclic = [pred_month, m_sin, m_cos]
                        row = np.array(base + lags + rolls + cyclic).reshape(1, -1)

                        pred = float(model.predict(row)[0])
                        predictions.append(pred)
                        dates.append((datetime.now() + timedelta(days=30 * i)).strftime("%b %Y"))
                        current_temp = pred

            elif selected_model in DL_MODELS:
                # ── DL model: sequence-based inference with scaling ───────
                dl_model = _load_dl_model(selected_model)
                sc_X, sc_y = _load_scalers()

                if dl_model is not None and sc_X is not None and sc_y is not None:
                    model_loaded = True
                    co2_latest = 420.0
                    lookback = 24  # must match config.LOOKBACK

                    # Build a synthetic history sequence of `lookback` steps
                    current_temp = temp_current
                    for i in range(forecast_months):
                        pred_month = (start_month + i - 1) % 12 + 1
                        # Features: [temp_C, humidity_pct, rainfall_mm, solar_MJ, co2_ppm]
                        row = np.array([current_temp, humidity, rainfall, solar, co2_latest])
                        seq = np.tile(row, (lookback, 1))  # repeat as lookback window
                        seq_scaled = sc_X.transform(seq)
                        seq_input = seq_scaled.reshape(1, lookback, -1)

                        pred_scaled = dl_model.predict(seq_input, verbose=0).flatten()[0]
                        pred = float(sc_y.inverse_transform([[pred_scaled]])[0, 0])
                        predictions.append(pred)
                        dates.append((datetime.now() + timedelta(days=30 * i)).strftime("%b %Y"))
                        current_temp = pred

            elif selected_model in ENSEMBLE_MAP:
                # ── Ensemble: run members and average ─────────────────────
                members = ENSEMBLE_MAP[selected_model]
                member_preds: list[list[float]] = []
                all_members_loaded = True

                for member in members:
                    m_preds: list[float] = []
                    if member in ML_MODELS:
                        model = _load_ml_model(member)
                        if model is None:
                            all_members_loaded = False
                            break
                        co2_latest = 420.0
                        current_temp = temp_current
                        for i in range(forecast_months):
                            pred_month = (start_month + i - 1) % 12 + 1
                            m_sin = np.sin(2 * np.pi * pred_month / 12)
                            m_cos = np.cos(2 * np.pi * pred_month / 12)
                            base = [humidity, rainfall, solar, co2_latest]
                            lags = [current_temp] * 3 + [humidity] * 3 + [rainfall] * 3 + [solar] * 3 + [co2_latest] * 3
                            rolls = [current_temp] * 3 + [humidity] * 3 + [rainfall] * 3 + [solar] * 3 + [co2_latest] * 3
                            cyclic = [pred_month, m_sin, m_cos]
                            row = np.array(base + lags + rolls + cyclic).reshape(1, -1)
                            pred = float(model.predict(row)[0])
                            m_preds.append(pred)
                            current_temp = pred
                    elif member in DL_MODELS:
                        dl_model = _load_dl_model(member)
                        sc_X, sc_y = _load_scalers()
                        if dl_model is None or sc_X is None or sc_y is None:
                            all_members_loaded = False
                            break
                        co2_latest = 420.0
                        lookback = 24
                        current_temp = temp_current
                        for i in range(forecast_months):
                            row = np.array([current_temp, humidity, rainfall, solar, co2_latest])
                            seq = np.tile(row, (lookback, 1))
                            seq_scaled = sc_X.transform(seq)
                            seq_input = seq_scaled.reshape(1, lookback, -1)
                            pred_scaled = dl_model.predict(seq_input, verbose=0).flatten()[0]
                            pred = float(sc_y.inverse_transform([[pred_scaled]])[0, 0])
                            m_preds.append(pred)
                            current_temp = pred
                    member_preds.append(m_preds)

                if all_members_loaded and member_preds:
                    model_loaded = True
                    predictions = list(np.mean(member_preds, axis=0))
                    dates = [(datetime.now() + timedelta(days=30 * i)).strftime("%b %Y")
                             for i in range(forecast_months)]
                    inference_label = f"{selected_model} ({' + '.join(members)})"

            # ── Handle results / fallback ─────────────────────────────────
            if model_loaded and predictions:
                st.success(f"✅ Forecast generated using **{inference_label}** (trained model).")
            else:
                st.warning(
                    f"⚠️ Saved artefacts for *{selected_model}* not found. "
                    "Showing analytical estimate. Run `python train.py` to save models."
                )
                current_temp = temp_current
                for i in range(forecast_months):
                    pred_month = (start_month + i - 1) % 12 + 1
                    seasonal = np.sin(2 * np.pi * pred_month / 12) * 3
                    trend = i * 0.02
                    pred = current_temp + seasonal + trend
                    predictions.append(float(pred))
                    dates.append((datetime.now() + timedelta(days=30 * i)).strftime("%b %Y"))

            # ── Confidence intervals ──────────────────────────────────────
            ci = 1.96 * model_rmse
            lower = [p - ci for p in predictions]
            upper = [p + ci for p in predictions]

            # ── Summary metrics ───────────────────────────────────────────
            st.markdown("### 📈 Forecast Results")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Next Month", f"{predictions[0]:.2f}°C",
                          delta=f"{predictions[0] - temp_current:+.2f}°C")
            with m2:
                st.metric(f"{forecast_months}-Month Avg", f"{np.mean(predictions):.2f}°C")
            with m3:
                st.metric("95% CI Width", f"±{ci:.2f}°C")

            # ── Chart ─────────────────────────────────────────────────────
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(102,126,234,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=predictions,
                mode="lines+markers", name="Predicted Temperature",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=10),
            ))
            fig.update_layout(
                template="plotly_white", height=500,
                title=f"Temperature Forecast — Next {forecast_months} Months",
                xaxis_title="Month", yaxis_title="Temperature (°C)",
                hovermode="x unified",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Table ─────────────────────────────────────────────────────
            st.markdown("### 📋 Forecast Table")
            st.dataframe(
                pd.DataFrame({
                    "Month": dates,
                    "Predicted (°C)": [f"{p:.2f}" for p in predictions],
                    "Lower 95% (°C)": [f"{l:.2f}" for l in lower],
                    "Upper 95% (°C)": [f"{u:.2f}" for u in upper],
                }),
                use_container_width=True, height=300,
            )

else:
    empty_state("Models Not Trained", "Run `python train.py` to enable forecasting.")

footer(text="Forecast Engine | Powered by AI & Climate Science")
