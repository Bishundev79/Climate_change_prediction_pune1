"""
Page 2 — Interactive Data Explorer.

Filterable time-series plots, statistical summaries, distribution
analysis, and correlation heatmaps.
"""

import streamlit as st
import plotly.express as px

from utils.shared import load_css, load_climate_data, hero_section, gradient_card, footer, empty_state

st.set_page_config(page_title="Data Explorer | Pune Climate", page_icon="🔍", layout="wide")
load_css()

df = load_climate_data()

hero_section(
    title="🔍 Interactive Data Explorer",
    subtitle="Dive deep into 73 years of climate data",
    variant="hero-purple",
)

VARIABLE_LABELS = {
    "temp_C": "🌡️ Temperature",
    "humidity_pct": "💧 Humidity",
    "rainfall_mm": "🌧️ Rainfall",
    "solar_MJ": "☀️ Solar Radiation",
    "co2_ppm": "🌫️ CO₂",
}

if df is not None:
    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("## 🎛️ Filter Data")

    year_range = st.sidebar.slider(
        "Year Range",
        int(df["date"].dt.year.min()),
        int(df["date"].dt.year.max()),
        (int(df["date"].dt.year.min()), int(df["date"].dt.year.max())),
    )

    variable = st.sidebar.selectbox(
        "Variable",
        list(VARIABLE_LABELS.keys()),
        format_func=lambda k: VARIABLE_LABELS[k],
    )

    aggregation = st.sidebar.radio("Aggregation", ["Daily", "Monthly", "Yearly"])

    # ── Filter & aggregate ────────────────────────────────────────────────────
    mask = df["date"].dt.year.between(*year_range)
    df_filtered = df.loc[mask]

    resample_map = {"Monthly": "MS", "Yearly": "YS"}
    if aggregation in resample_map:
        df_plot = df_filtered.set_index("date").resample(resample_map[aggregation]).mean().reset_index()
    else:
        df_plot = df_filtered

    # ── Time-series chart ─────────────────────────────────────────────────────
    st.markdown(f"### 📊 {VARIABLE_LABELS[variable]} Over Time")
    fig = px.line(df_plot, x="date", y=variable, title=f"{aggregation} {variable.replace('_', ' ').title()}")
    fig.update_traces(line=dict(color="#3498db", width=2))
    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary statistics ────────────────────────────────────────────────────
    st.markdown("### 📊 Statistical Summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(gradient_card("Mean", f"{df_filtered[variable].mean():.2f}", variant="purple"), unsafe_allow_html=True)
    with s2:
        st.markdown(gradient_card("Std Dev", f"{df_filtered[variable].std():.2f}", variant="pink"), unsafe_allow_html=True)
    with s3:
        st.markdown(gradient_card("Min", f"{df_filtered[variable].min():.2f}", variant="blue"), unsafe_allow_html=True)
    with s4:
        st.markdown(gradient_card("Max", f"{df_filtered[variable].max():.2f}", variant="orange"), unsafe_allow_html=True)

    # ── Distribution analysis ─────────────────────────────────────────────────
    st.markdown("### 📊 Distribution Analysis")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(df_filtered, x=variable, nbins=50, title="Histogram")
        fig.update_traces(marker_color="#9b59b6")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.box(df_filtered, y=variable, title="Box Plot")
        fig.update_traces(marker_color="#16a085")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────
    st.markdown("### 🔗 Variable Correlations")
    num_cols = [c for c in ["temp_C", "humidity_pct", "rainfall_mm", "solar_MJ"] if c in df_filtered.columns]
    corr = df_filtered[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

else:
    empty_state("Data Not Available", "Ensure `data/pune_climate_with_co2.csv` exists.")

footer(text="Data Explorer | Filter · Visualize · Analyze")
