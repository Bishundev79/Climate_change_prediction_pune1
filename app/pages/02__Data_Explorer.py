import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer | Pune Climate", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_climate_data():
    try:
        return pd.read_csv('data/pune_climate_with_co2.csv', parse_dates=['date'])
    except:
        return None


df = load_climate_data()

st.markdown("<h1 style='text-align: center; color: #3498db;'>🔍 Interactive Data Explorer</h1>", unsafe_allow_html=True)

if df is not None:
    # Filters
    st.sidebar.markdown("## 🎛️ Filter Data")

    year_range = st.sidebar.slider(
        "Select Year Range",
        int(df['date'].dt.year.min()),
        int(df['date'].dt.year.max()),
        (int(df['date'].dt.year.min()), int(df['date'].dt.year.max()))
    )

    variable = st.sidebar.selectbox(
        "Select Variable",
        ["temp_C", "humidity_pct", "rainfall_mm", "solar_MJ", "co2_ppm"],
        format_func=lambda x: {
            "temp_C": "🌡️ Temperature",
            "humidity_pct": "💧 Humidity",
            "rainfall_mm": "🌧️ Rainfall",
            "solar_MJ": "☀️ Solar Radiation",
            "co2_ppm": "🌫️ CO₂"
        }[x]
    )

    aggregation = st.sidebar.radio("Aggregation", ["Daily", "Monthly", "Yearly"])

    # Filter data
    df_filtered = df[(df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])]

    # Aggregate
    if aggregation == "Monthly":
        df_plot = df_filtered.set_index('date').resample('MS').mean().reset_index()
    elif aggregation == "Yearly":
        df_plot = df_filtered.set_index('date').resample('YS').mean().reset_index()
    else:
        df_plot = df_filtered

    # Time series plot
    st.markdown(f"### 📊 {variable.replace('_', ' ').title()} Over Time")

    fig = px.line(
        df_plot, x='date', y=variable,
        title=f'{aggregation} {variable.replace("_", " ").title()}',
        labels={'date': 'Date', variable: variable.replace('_', ' ').title()}
    )

    fig.update_traces(line=dict(color='#3498db', width=2))
    fig.update_layout(template='plotly_white', height=450, hovermode='x unified')

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df_filtered[variable].mean():.2f}")
    with col2:
        st.metric("Std Dev", f"{df_filtered[variable].std():.2f}")
    with col3:
        st.metric("Min", f"{df_filtered[variable].min():.2f}")
    with col4:
        st.metric("Max", f"{df_filtered[variable].max():.2f}")

    # Distribution
    st.markdown("### 📊 Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df_filtered, x=variable, nbins=50, title='Histogram')
        fig.update_traces(marker_color='#9b59b6')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df_filtered, y=variable, title='Box Plot')
        fig.update_traces(marker_color='#16a085')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.markdown("### 🔗 Variable Correlations")
    corr = df_filtered[['temp_C', 'humidity_pct', 'rainfall_mm', 'solar_MJ']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ Data file not found. Please ensure `pune_climate_with_co2.csv` is in the `data/` folder.")
