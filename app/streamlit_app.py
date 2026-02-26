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
    st.markdown(gradient_card("📅 Data Span", f"{n_years} Years", "1951 – 2024", "emerald"), unsafe_allow_html=True)
with c2:
    st.markdown(gradient_card("🤖 Models", "4 AI Models", "XGBoost, RF, CNN-LSTM, Transformer", "purple"), unsafe_allow_html=True)
with c3:
    st.markdown(gradient_card("🎯 Best RMSE", best_rmse, "Top model accuracy", "blue"), unsafe_allow_html=True)
with c4:
    st.markdown(gradient_card("📊 Records", n_records, "Daily data points", "coral"), unsafe_allow_html=True)

# ── Platform Navigation ──────────────────────────────────────────────────────
st.markdown("### 🚀 Discover the Platform")

st.markdown("""
<style>
.nav-desc { color: #9ca3af; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px; margin-left: 35px; }
</style>
""", unsafe_allow_html=True)

n1, n2 = st.columns(2)

with n1:
    st.page_link("pages/01__Overview.py", label="**📊 Climate Data Overview**", icon="📊")
    st.markdown("<p class='nav-desc'>Explore 73 years of historical averages, seasonal changes, and warming trends.</p>", unsafe_allow_html=True)
    
    st.page_link("pages/02__Data_Explorer.py", label="**🔍 Interactive Data Explorer**", icon="🔍")
    st.markdown("<p class='nav-desc'>Filter raw data, view correlation heatmaps, and analyze distributions.</p>", unsafe_allow_html=True)
    
    st.page_link("pages/05__Benchmark.py", label="**⚖️ Industry Benchmarks**", icon="⚖️")
    st.markdown("<p class='nav-desc'>See how our AI models compare against global meteorological services.</p>", unsafe_allow_html=True)

with n2:
    st.page_link("pages/03__Model_Arena.py", label="**🤖 Model Arena**", icon="🤖")
    st.markdown("<p class='nav-desc'>Dive into architecture specs, head-to-head metrics, and feature importance.</p>", unsafe_allow_html=True)

    st.page_link("pages/04__Forecast.py", label="**🔮 Next-Gen Forecast**", icon="🔮")
    st.markdown("<p class='nav-desc'>Generate future climate predictions using our advanced Deep Learning models.</p>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
footer(
    text="Built with Streamlit & Plotly | Climate Intelligence Platform",
    sub="© 2026 Pune Climate Intelligence — Powered by ML & Data Science",
)
