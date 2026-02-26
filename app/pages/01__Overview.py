"""
Page 1 — Project Overview.

Dataset statistics (computed dynamically), long-term climate trends,
seasonal analysis, and model performance summary.
"""

import streamlit as st
import plotly.graph_objects as go

from utils.shared import (
    load_css,
    load_results,
    load_climate_data,
    hero_section,
    gradient_card,
    footer,
    empty_state,
)

st.set_page_config(page_title="Climate Data Overview | Pune Climate", page_icon="📊", layout="wide")
load_css()

df = load_climate_data()
results = load_results()

# ── Hero Banner ───────────────────────────────────────────────────────────────
hero_section(
    title="📊 Climate Data Overview",
    subtitle="Historical Trends and Dataset Analysis",
    detail="Exploring 73 years of temperature, rainfall, and climate shifts",
    variant="hero-purple",
)

# ── Dataset Insights ─────────────────────────────────────────────────────────
if df is not None:
    st.markdown("### 📁 Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            gradient_card("📅 Records", f"{len(df):,}", "Daily measurements", "purple"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            gradient_card("🌡️ Avg Temp", f"{df['temp_C'].mean():.1f}°C",
                          f"Range: {df['temp_C'].min():.1f} – {df['temp_C'].max():.1f}°C", "pink"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            gradient_card("💧 Avg Humidity", f"{df['humidity_pct'].mean():.1f}%",
                          "Relative humidity", "blue"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            gradient_card("🌧️ Avg Rainfall", f"{df['rainfall_mm'].mean():.1f} mm",
                          "Daily average", "emerald"),
            unsafe_allow_html=True,
        )

    # ── Temporal Coverage & Key Insights ──────────────────────────────────────
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### ⏰ Temporal Coverage")
        n_years = (df["date"].max() - df["date"].min()).days // 365
        st.metric("Start Date", df["date"].min().strftime("%B %d, %Y"))
        st.metric("End Date", df["date"].max().strftime("%B %d, %Y"))
        st.metric("Total Duration", f"{n_years} Years")

    with right:
        st.markdown("#### 🔍 Key Insights")
        temp_trend = "Increasing 📈" if df["temp_C"].iloc[-365:].mean() > df["temp_C"].iloc[:365].mean() else "Stable"
        co2_max = df["co2_ppm"].max() if "co2_ppm" in df.columns else 0
        st.metric("Temperature Trend", temp_trend)
        st.metric("Peak Temperature", f"{df['temp_C'].max():.1f}°C")
        st.metric("Max CO₂ Level", f"{co2_max:.0f} ppm")

    # ── Long-term Climate Trends ──────────────────────────────────────────────
    st.markdown("### 📈 Long-term Climate Trends")
    st.info("📊 **Insight:** Yearly aggregated climate patterns over seven decades.")

    yearly = df.set_index("date").resample("YS").mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["date"], y=yearly["temp_C"],
        mode="lines+markers", name="Temperature",
        line=dict(color="#e74c3c", width=3),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=yearly["date"], y=yearly["humidity_pct"],
        mode="lines+markers", name="Humidity",
        line=dict(color="#3498db", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark", height=500,
        title="Yearly Averages: Temperature & Humidity",
        xaxis_title="Year",
        yaxis=dict(title="Temperature (°C)", title_font=dict(color="#e74c3c")),
        yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", title_font=dict(color="#3498db")),
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal Analysis ─────────────────────────────────────────────────────
    st.markdown("### 🌤️ Seasonal Analysis")
    s1, s2, s3 = st.columns(3)
    summer = df[df["date"].dt.month.isin([3, 4, 5])]["temp_C"].mean()
    monsoon = df[df["date"].dt.month.isin([6, 7, 8, 9])]["rainfall_mm"].mean()
    winter = df[df["date"].dt.month.isin([12, 1, 2])]["temp_C"].mean()

    with s1:
        st.markdown(gradient_card("☀️ Summer", f"{summer:.1f}°C", "Mar – May Average", "orange"), unsafe_allow_html=True)
    with s2:
        st.markdown(gradient_card("🌧️ Monsoon", f"{monsoon:.1f} mm", "Jun – Sep Rainfall", "blue"), unsafe_allow_html=True)
    with s3:
        st.markdown(gradient_card("❄️ Winter", f"{winter:.1f}°C", "Dec – Feb Average", "purple"), unsafe_allow_html=True)



elif df is None:
    empty_state("Data Not Available", "Ensure `data/pune_climate_with_co2.csv` exists.")

# ── Footer ────────────────────────────────────────────────────────────────────
footer(
    text="Built with Streamlit & Plotly | Climate Intelligence Platform",
    sub="Powered by Advanced Machine Learning & Data Science",
)
