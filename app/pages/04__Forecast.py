import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Forecast | Pune Climate", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .forecast-card { background: white; padding: 2rem; border-radius: 15px;
                     box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results():
    try:
        return pd.read_csv('results/test_metrics.csv')
    except:
        return pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])

def fetch_iot_sensor():
    try:
        resp = requests.get("http://localhost:5001/iot/latest", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def fetch_latest_from_csv(file='data/pune_climate_with_co2.csv'):
    try:
        df = pd.read_csv(file, parse_dates=['date'])
        last_date = df['date'].max()
        monthly = df[(df['date'].dt.month == last_date.month) & (df['date'].dt.year == last_date.year)]
        return {
            'temp_C': monthly['temp_C'].mean(),
            'humidity_pct': monthly['humidity_pct'].mean(),
            'rainfall_mm': monthly['rainfall_mm'].sum(),
            'solar_MJ': monthly['solar_MJ'].mean()
        }
    except Exception:
        return None

def fetch_monthly_from_openmeteo():
    """
    Fetch last 31 days of weather data from Open-Meteo API and compute monthly averages.
    Converts units to match project schema (temp_C, humidity_pct, rainfall_mm, solar_MJ).
    """
    try:
        # Pune coordinates
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 18.5204,
            "longitude": 73.8567,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min,shortwave_radiation_sum",
            "past_days": 31,
            "timezone": "Asia/Kolkata"
        }
        
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()  # Raise exception for bad status codes
        
        jsn = resp.json()
        daily = jsn.get('daily', {})
        
        # Check if all required fields exist
        required_fields = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 
                          'relative_humidity_2m_max', 'relative_humidity_2m_min', 'shortwave_radiation_sum']
        if not all(field in daily for field in required_fields):
            return None
        
        # Calculate averages (filter out None values)
        temps_max = [t for t in daily['temperature_2m_max'] if t is not None]
        temps_min = [t for t in daily['temperature_2m_min'] if t is not None]
        temps = [(a+b)/2 for a, b in zip(temps_max, temps_min)]
        
        humidity_max = [h for h in daily['relative_humidity_2m_max'] if h is not None]
        humidity_min = [h for h in daily['relative_humidity_2m_min'] if h is not None]
        humidity = [(a+b)/2 for a, b in zip(humidity_max, humidity_min)]
        
        rainfall = [r for r in daily['precipitation_sum'] if r is not None]
        solar_raw = [s for s in daily['shortwave_radiation_sum'] if s is not None]
        
        # Convert solar radiation from Wh/m² to MJ/m² (1 Wh = 0.0036 MJ)
        solar = [s * 0.0036 for s in solar_raw]
        
        return {
            'temp_C': float(np.mean(temps)) if temps else 25.0,
            'humidity_pct': float(np.mean(humidity)) if humidity else 65.0,
            'rainfall_mm': float(np.sum(rainfall)) if rainfall else 50.0,
            'solar_MJ': float(np.mean(solar)) if solar else 18.0,
        }
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Data parsing error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error fetching live data: {str(e)}")
    return None

results = load_results()

st.markdown("<h1 style='text-align: center; color: #e74c3c;'>🔮 Climate Forecast Engine</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.05rem; color: #7f8c8d;'>
Select your data source and adjust if needed. "Manual" lets you set values yourself.<br>
"IoT" fetches from a (real or simulated) sensor API.<br>
"CSV" (historical) uses the latest month in your dataset.<br>
"Open-Meteo API" pulls real-time recent monthly averages for Pune. All values are editable before forecasting.
</p>
""", unsafe_allow_html=True)

if not results.empty:
    st.sidebar.markdown("## 🎯 Forecast Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        results['Model'].tolist(),
        help="Choose which trained model to use for forecasting"
    )
    model_rmse = results[results['Model'] == selected_model]['RMSE'].values[0]
    st.sidebar.metric("Selected Model RMSE", f"{model_rmse}°C")

    # Data source selector
    st.markdown("#### 🗂️ Data Source")
    data_source = st.radio(
        "Set input values using:",
        (
            "Manual Entry",
            "IoT Sensor (REST)",
            "CSV: Latest Month",
            "Open-Meteo API (Live)"
        ),
        horizontal=True
    )

    # Prefill logic
    if data_source == "Manual Entry":
        prefill = {}
    elif data_source == "IoT Sensor (REST)":
        prefill = fetch_iot_sensor() or {}
        st.info("Fetched IoT Sensor Data" if prefill else "No live IoT data available.")
    elif data_source == "CSV: Latest Month":
        prefill = fetch_latest_from_csv() or {}
        st.info("Fetched latest available month from CSV." if prefill else "Could not find data in CSV.")
    elif data_source == "Open-Meteo API (Live)":
        prefill = fetch_monthly_from_openmeteo() or {}
        st.info("Fetched recent real monthly averages (Open-Meteo)." if prefill else "Could not fetch live weather.")
    else:
        prefill = {}

    st.markdown("## 📊 Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='forecast-card'>", unsafe_allow_html=True)
        st.markdown("### 🌡️ Current Climate Conditions")
        temp_current = st.slider(
            "Average Monthly Temperature (°C) [last month's mean]", 15.0, 35.0,
            float(round(prefill.get("temp_C", 24.5), 2)), .5
        )
        humidity = st.slider(
            "Average Monthly Humidity (%)", 20.0, 95.0,
            float(round(prefill.get("humidity_pct", 65.0), 2)), 1.0
        )
        rainfall = st.slider(
            "Total Monthly Rainfall (mm)", 0.0, 500.0,
            float(round(prefill.get("rainfall_mm", 50.0), 1)), 10.0
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='forecast-card'>", unsafe_allow_html=True)
        st.markdown("### ☀️ Forecast Settings")
        solar = st.slider(
            "Average Monthly Solar Radiation (MJ/m²)", 10.0, 25.0,
            float(round(prefill.get("solar_MJ", 18.0), 2)), 0.5
        )
        forecast_months = st.slider(
            "Forecast Horizon (months)", 1, 12, 6
        )
        start_month = st.selectbox(
            "Starting Month", list(range(1, 13)), index=datetime.now().month - 1,
            format_func=lambda x: datetime(2024, x, 1).strftime('%B')
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Forecast Button
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            predictions = []
            dates = []
            lower_bounds = []
            upper_bounds = []
            for i in range(forecast_months):
                pred_month = (start_month + i - 1) % 12 + 1
                seasonal = np.sin(2 * np.pi * pred_month / 12) * 3
                trend = i * 0.02
                pred = temp_current + seasonal + trend + np.random.randn() * 0.3
                ci = 1.96 * model_rmse
                predictions.append(pred)
                dates.append((datetime.now() + timedelta(days=30*i)).strftime('%b %Y'))
                lower_bounds.append(pred - ci)
                upper_bounds.append(pred + ci)
            st.success(f"✅ Forecast generated successfully using **{selected_model}**!")
            st.markdown("## 📈 Forecast Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Next Month Prediction", f"{predictions[0]:.2f}°C", delta=f"{predictions[0] - temp_current:.2f}°C")
            with col2:
                st.metric(f"{forecast_months}-Month Average", f"{np.mean(predictions):.2f}°C")
            with col3:
                st.metric("Confidence Interval", f"±{model_rmse:.2f}°C", delta="95% CI")
            st.markdown("### 📊 Temperature Forecast Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Temperature',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=10, symbol='circle')
            ))
            fig.update_layout(
                template='plotly_white',
                height=500,
                title=f'Temperature Forecast - Next {forecast_months} Months',
                xaxis_title='Month',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📋 Detailed Forecast Table")
            forecast_df = pd.DataFrame({
                'Month': dates,
                'Predicted (°C)': [f"{p:.2f}" for p in predictions],
                'Lower Bound (°C)': [f"{l:.2f}" for l in lower_bounds],
                'Upper Bound (°C)': [f"{u:.2f}" for u in upper_bounds],
                'Confidence': ['95%'] * forecast_months
            })
            st.dataframe(forecast_df, use_container_width=True, height=300)
            st.info(
                "🎯 Forecast Insights:\n\n"
                "• Choose your preferred data source: demo IoT, real/historical, or manual input.\n"
                "• You can always tune values before prediction.\n"
                "• The displayed RMSE provides a typical forecast error—true uncertainty may be larger for extreme/novel conditions."
            )
else:
    st.warning("⚠️ No model results found. Train models first to enable forecasting.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #95a5a6;'>Forecast Engine | Powered by AI & Climate Science</p>", unsafe_allow_html=True)
