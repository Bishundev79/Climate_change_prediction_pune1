"""
Page 3 — Model Arena.

Head-to-head comparison of all trained models with performance
charts, radar plots, architecture cards, and feature importance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.shared import load_css, load_results, hero_section, footer, RESULTS_DIR

st.set_page_config(page_title="Model Arena | Pune Climate", page_icon="🤖", layout="wide")
load_css()

results = load_results()

hero_section(
    title="🤖 Model Performance Arena",
    subtitle="Head-to-Head Model Comparison",
    variant="hero-purple",
)

if not results.empty:
    sorted_res = results.sort_values("RMSE")
    best = sorted_res.iloc[0]

    # ── Winner Banner ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='winner-banner gc-purple'>"
        f"<h2>🏆 Champion Model</h2>"
        f"<h1>{best['Model']}</h1>"
        f"<div class='stats'>"
        f"<div class='stat-item'><p>RMSE</p><h2>{best['RMSE']}°C</h2></div>"
        f"<div class='stat-item'><p>MAE</p><h2>{best['MAE']}°C</h2></div>"
        f"<div class='stat-item'><p>R²</p><h2>{best['R2']}</h2></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Metric tabs ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance Metrics Comparison")
    tab1, tab2, tab3 = st.tabs(["📉 RMSE", "📈 R² Score", "📊 Radar"])

    palette = ["#27ae60", "#3498db", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]
    colors = palette[: len(sorted_res)]

    with tab1:
        fig = go.Figure(go.Bar(
            x=sorted_res["Model"], y=sorted_res["RMSE"],
            marker_color=colors,
            text=sorted_res["RMSE"].round(4), textposition="outside",
        ))
        fig.update_layout(template="plotly_white", height=450, title="RMSE (lower is better)", yaxis_title="RMSE (°C)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("📉 Lower RMSE indicates better prediction accuracy.")

    with tab2:
        fig = go.Figure(go.Bar(
            x=sorted_res["Model"], y=sorted_res["R2"],
            marker_color=colors,
            text=sorted_res["R2"].round(4), textposition="outside",
        ))
        fig.update_layout(template="plotly_white", height=450, title="R² Score (higher is better)", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        st.info("📈 R² closer to 1 means the model explains more variance.")

    with tab3:
        fig = go.Figure()
        for _, row in sorted_res.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[1 - row["RMSE"] / 5, row["R2"], 1 - row["MAE"] / 5],
                theta=["RMSE", "R²", "MAE"],
                fill="toself", name=row["Model"],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white", height=500, title="Multi-Metric Radar",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed table ────────────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Performance Table")
    st.dataframe(
        sorted_res.style.background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE"])
        .background_gradient(cmap="RdYlGn", subset=["R2"])
        .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}),
        use_container_width=True, height=250,
    )

    # ── Architecture overview ─────────────────────────────────────────────────
    st.markdown("### 🏗️ Model Architecture Overview")
    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            "<div class='model-card'><h3>🌲 Random Forest</h3>"
            "<p><strong>Type:</strong> Ensemble Learning</p>"
            "<p><strong>Estimators:</strong> 400 trees · Max Depth: 15</p>"
            "<p><strong>Best for:</strong> Robust baseline with minimal tuning</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='model-card'><h3>🚀 XGBoost</h3>"
            "<p><strong>Type:</strong> Gradient Boosting</p>"
            "<p><strong>Estimators:</strong> 500 · LR: 0.05</p>"
            "<p><strong>Best for:</strong> Feature-engineered tabular data</p></div>",
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            "<div class='model-card'><h3>🔗 CNN-LSTM Hybrid</h3>"
            "<p><strong>Type:</strong> Deep Learning</p>"
            "<p><strong>Architecture:</strong> 3 Conv1D → 2 LSTM layers</p>"
            "<p><strong>Best for:</strong> Spatial feature extraction + temporal modelling</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='model-card'><h3>🎯 Transformer + Attention</h3>"
            "<p><strong>Type:</strong> Attention-based DL</p>"
            "<p><strong>Architecture:</strong> 4 layers · 8 heads · d=128</p>"
            "<p><strong>Best for:</strong> Long-range dependencies (larger datasets)</p></div>",
            unsafe_allow_html=True,
        )

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### 🔍 Feature Importance Analysis")
    fi_tab1, fi_tab2 = st.tabs(["🌲 Random Forest", "🚀 XGBoost"])

    for tab, prefix, cmap in [(fi_tab1, "rf", "Viridis"), (fi_tab2, "xgb", "Plasma")]:
        with tab:
            fi_path = RESULTS_DIR / f"{prefix}_feature_importance.csv"
            if fi_path.exists():
                fi = pd.read_csv(fi_path).head(15)
                fig = go.Figure(go.Bar(
                    x=fi["importance"], y=fi["feature"], orientation="h",
                    marker=dict(color=fi["importance"], colorscale=cmap, showscale=True),
                    text=fi["importance"].round(3), textposition="auto",
                ))
                fig.update_layout(
                    template="plotly_white", height=500,
                    title="Top 15 Features", yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("📊 View all feature importances"):
                    st.dataframe(pd.read_csv(fi_path), use_container_width=True, height=300)
            else:
                st.warning(f"⚠️ `{fi_path.name}` not found. Run `python train.py` to generate it.")
else:
    st.warning("⚠️ No model results found. Train models first: `python train.py`")

footer(text="Model Arena | Compare · Analyze · Optimize")
