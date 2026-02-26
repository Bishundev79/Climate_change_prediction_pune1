"""
Page 3 — Model Arena.

Head-to-head comparison of all trained models with performance
charts, radar plots, architecture cards, and feature importance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.shared import load_css, load_results, hero_section, footer, RESULTS_DIR, gradient_card

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
    st.markdown("### 🏆 Champion Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(gradient_card("🥇 Model", best['Model'], "Top Performer", "purple"), unsafe_allow_html=True)
    with c2:
        st.markdown(gradient_card("📉 RMSE", f"{best['RMSE']}°C", "Root Mean Sq. Error", "emerald"), unsafe_allow_html=True)
    with c3:
        st.markdown(gradient_card("📏 MAE", f"{best['MAE']}°C", "Mean Absolute Error", "blue"), unsafe_allow_html=True)
    with c4:
        st.markdown(gradient_card("📈 R²", f"{best['R2']}", "Variance Explained", "pink"), unsafe_allow_html=True)

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
        fig.update_layout(template="plotly_dark", height=450, title="RMSE (lower is better)", yaxis_title="RMSE (°C)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("📉 Lower RMSE indicates better prediction accuracy.")

    with tab2:
        fig = go.Figure(go.Bar(
            x=sorted_res["Model"], y=sorted_res["R2"],
            marker_color=colors,
            text=sorted_res["R2"].round(4), textposition="outside",
        ))
        fig.update_layout(template="plotly_dark", height=450, title="R² Score (higher is better)", yaxis=dict(range=[0, 1]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
            polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(0,0,0,0)"),
            template="plotly_dark", height=500, title="Multi-Metric Radar",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed table ────────────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Performance Table")
    st.dataframe(
        sorted_res.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}),
        use_container_width=True, height=250,
    )

    # ── Architecture overview ─────────────────────────────────────────────────
    st.markdown("### 🏗️ Trained Model Zoo")
    
    # Dynamic descriptions mapping based on model names
    MODEL_DESCRIPTIONS = {
        "Random Forest": {
            "icon": "🌲", "type": "Machine Learning", 
            "spec": "400 trees · Max Depth: 15", "best": "Robust baseline with minimal tuning"
        },
        "XGBoost": {
            "icon": "🚀", "type": "Gradient Boosting", 
            "spec": "500 estimators · LR: 0.05", "best": "Feature-engineered tabular data"
        },
        "CNN-LSTM": {
            "icon": "🔗", "type": "Deep Learning Hybrid", 
            "spec": "3 Conv1D → 2 LSTM layers", "best": "Spatial feature extraction + temporal modelling"
        },
        "Transformer": {
            "icon": "🎯", "type": "Attention-based DL", 
            "spec": "4 layers · 8 heads · d=128", "best": "Long-range dependencies"
        }
    }
    
    cols = st.columns(2)
    trained_models = sorted_res['Model'].tolist()
    
    for i, model_name in enumerate(trained_models):
        col = cols[i % 2]
        desc = MODEL_DESCRIPTIONS.get(model_name, {
            "icon": "⚙️", "type": "Custom Architecture", 
            "spec": "Dynamically loaded", "best": "Specific climate logic"
        })
        
        with col:
            st.markdown(
                f"<div class='model-card'><h3>{desc['icon']} {model_name}</h3>"
                f"<p><strong>Type:</strong> {desc['type']}</p>"
                f"<p><strong>Specs:</strong> {desc['spec']}</p>"
                f"<p><strong>Best for:</strong> {desc['best']}</p></div>",
                unsafe_allow_html=True,
            )

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### 🔍 Feature Importance Analysis")
    
    ml_models = [m for m in trained_models if m in ["Random Forest", "XGBoost"]]
    if ml_models:
        tabs = st.tabs(ml_models)
        for i, model_name in enumerate(ml_models):
            with tabs[i]:
                # Map model name back to prefix
                prefix = "rf" if "Random Forest" in model_name else "xgb" if "XGBoost" in model_name else None
                if prefix:
                    fi_path = RESULTS_DIR / f"{prefix}_feature_importance.csv"
                    if fi_path.exists():
                        fi = pd.read_csv(fi_path).head(15)
                        cmap = "Viridis" if prefix == "rf" else "Plasma"
                        fig = go.Figure(go.Bar(
                            x=fi["importance"], y=fi["feature"], orientation="h",
                            marker=dict(color=fi["importance"], colorscale=cmap, showscale=True),
                            text=fi["importance"].round(3), textposition="auto",
                        ))
                        fig.update_layout(
                            template="plotly_dark", height=500,
                            title=f"Top 15 Features ({model_name})", yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("📊 View all feature importances"):
                            st.dataframe(pd.read_csv(fi_path), use_container_width=True, height=300)
                    else:
                        st.warning(f"⚠️ `{fi_path.name}` not found.")
else:
    st.warning("⚠️ No model results found. Train models first: `python train.py`")

footer(text="Model Arena | Compare · Analyze · Optimize")
