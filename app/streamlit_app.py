"""
Home page — Pune Climate Intelligence Platform.

Acts as the landing page and application shell (sidebar nav, quick
stats, climate trend preview).
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
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Pune Climate Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# ── Data ──────────────────────────────────────────────────────────────────────
results = load_results()
climate_df = load_climate_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ Climate Dashboard")
    st.markdown("---")

    if not results.empty:
        best = results.iloc[0]
        st.metric("Best Model", best["Model"])
        st.metric("Best RMSE", f"{best['RMSE']}°C")
        st.metric("Best R²", f"{best['R2']}")

    st.markdown("---")
    st.markdown("### 🎯 Navigation")
    st.page_link("streamlit_app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/01__Overview.py", label="📊 Overview", icon="📊")
    st.page_link("pages/02__Data_Explorer.py", label="🔍 Data Explorer", icon="🔍")
    st.page_link("pages/03__Model_Arena.py", label="🤖 Model Arena", icon="🤖")
    st.page_link("pages/04__Forecast.py", label="🔮 Forecast", icon="🔮")
    st.page_link("pages/05__Benchmark.py", label="⚖️ Benchmark", icon="⚖️")

# ── Hero ──────────────────────────────────────────────────────────────────────
hero_section(
    title="🌍 Pune Climate Intelligence",
    subtitle="Advanced Machine Learning for Climate Forecasting | 1951 – 2024",
)

# ── Key Metrics (dynamically computed) ────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

if climate_df is not None:
    n_years = (climate_df["date"].max() - climate_df["date"].min()).days // 365
    n_records = f"{len(climate_df):,}"
else:
    n_years = "N/A"
    n_records = "N/A"

best_rmse = f"{results.iloc[0]['RMSE']}°C" if not results.empty else "—"
n_models = str(len(results)) if not results.empty else "0"

with c1:
    st.markdown(gradient_card("📅 Data Span", f"{n_years} Years", "1951 – 2024", "gc-purple"), unsafe_allow_html=True)
with c2:
    st.markdown(gradient_card("🤖 Models", f"{n_models} AI Models", "ML + DL + Ensembles", "gc-pink"), unsafe_allow_html=True)
with c3:
    st.markdown(gradient_card("🎯 Best RMSE", best_rmse, "Top model accuracy", "gc-blue"), unsafe_allow_html=True)
with c4:
    st.markdown(gradient_card("📊 Records", n_records, "Daily data points", "gc-green"), unsafe_allow_html=True)

# ── Model Leaderboard ────────────────────────────────────────────────────────
st.markdown("### 🏆 Model Performance Leaderboard")

if not results.empty:
    sorted_res = results.sort_values("RMSE")
    colors = ["#27ae60", "#3498db", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sorted_res["Model"],
            y=sorted_res["RMSE"],
            marker=dict(color=colors[: len(sorted_res)]),
            text=sorted_res["RMSE"].round(4),
            textposition="outside",
            name="RMSE (°C)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        title="Model RMSE Comparison (lower is better)",
        xaxis_title="Model",
        yaxis_title="RMSE (°C)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Detailed Metrics")
    st.dataframe(
        sorted_res.style.background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE"])
        .background_gradient(cmap="RdYlGn", subset=["R2"])
        .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}),
        use_container_width=True,
        height=220,
    )
else:
    st.info("⚠️ No model results found. Train models first: `python train.py`")

# ── Temperature Trend ─────────────────────────────────────────────────────────
if climate_df is not None:
    st.markdown("### 🌡️ Climate Trends Overview")
    monthly = climate_df.set_index("date").resample("MS").mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["temp_C"],
            mode="lines",
            name="Temperature",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.1)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        title="Temperature Trend (1951 – 2024)",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
footer(
    text="Built with Streamlit & Plotly | Climate Intelligence Platform",
    sub="© 2026 Pune Climate Intelligence — Powered by ML & Data Science",
)
