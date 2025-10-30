import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Arena | Pune Climate", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .winner-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    try:
        return pd.read_csv('results/test_metrics.csv')
    except:
        return pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])


results = load_results()

st.markdown("<h1 style='text-align: center; color: #9b59b6;'>🤖 Model Performance Arena</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>Head-to-Head Model Comparison</p>",
            unsafe_allow_html=True)

if not results.empty:
    results_sorted = results.sort_values("RMSE")

    # Winner Announcement
    best = results_sorted.iloc[0]
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin: 2rem 0;'>
        <h2 style='margin: 0; color: white;'>🏆 Champion Model</h2>
        <h1 style='margin: 1rem 0; font-size: 3rem; color: white;'>{best['Model']}</h1>
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 1.5rem;'>
            <div>
                <p style='margin: 0; opacity: 0.9;'>RMSE</p>
                <h2 style='margin: 0.5rem 0; color: white;'>{best['RMSE']}°C</h2>
            </div>
            <div>
                <p style='margin: 0; opacity: 0.9;'>MAE</p>
                <h2 style='margin: 0.5rem 0; color: white;'>{best['MAE']}°C</h2>
            </div>
            <div>
                <p style='margin: 0; opacity: 0.9;'>R² Score</p>
                <h2 style='margin: 0.5rem 0; color: white;'>{best['R2']}</h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison Charts
    st.markdown("### 📊 Performance Metrics Comparison")

    tab1, tab2, tab3 = st.tabs(["📉 RMSE", "📈 R² Score", "📊 All Metrics"])

    with tab1:
        fig = go.Figure()
        colors = ['#27ae60' if i == 0 else '#3498db' if i == 1 else '#e67e22' if i == 2 else '#e74c3c'
                  for i in range(len(results_sorted))]

        fig.add_trace(go.Bar(
            x=results_sorted['Model'],
            y=results_sorted['RMSE'],
            marker=dict(color=colors),
            text=results_sorted['RMSE'].round(4),
            textposition='outside',
            textfont=dict(size=14, color='#2c3e50')
        ))

        fig.update_layout(
            template='plotly_white',
            height=450,
            title='RMSE Comparison (Lower is Better)',
            yaxis_title='RMSE (°C)',
            showlegend=False,
            font=dict(size=14)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("📉 Lower RMSE indicates better prediction accuracy")

    with tab2:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=results_sorted['Model'],
            y=results_sorted['R2'],
            marker=dict(color=colors),
            text=results_sorted['R2'].round(4),
            textposition='outside',
            textfont=dict(size=14, color='#2c3e50')
        ))

        fig.update_layout(
            template='plotly_white',
            height=450,
            title='R² Score Comparison (Higher is Better)',
            yaxis_title='R² Score',
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            font=dict(size=14)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("📈 R² closer to 1 indicates the model explains more variance")

    with tab3:
        # Radar chart
        fig = go.Figure()

        for idx, row in results_sorted.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[1 - row['RMSE'] / 5, row['R2'], 1 - row['MAE'] / 5],  # Normalize
                theta=['RMSE', 'R²', 'MAE'],
                fill='toself',
                name=row['Model']
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template='plotly_white',
            height=500,
            title='Multi-Metric Performance Radar'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.markdown("### 📋 Detailed Performance Table")

    st.dataframe(
        results_sorted.style.background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE'])
        .background_gradient(cmap='RdYlGn', subset=['R2'])
        .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R2': '{:.4f}'}),
        use_container_width=True,
        height=250
    )

    # Model Architecture Cards
    st.markdown("### 🏗️ Model Architecture Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='model-card'>
            <h3 style='color: #27ae60;'>🌲 Random Forest</h3>
            <p><strong>Type:</strong> Ensemble Learning</p>
            <p><strong>Estimators:</strong> 400 trees</p>
            <p><strong>Max Depth:</strong> 15</p>
            <p><strong>Best for:</strong> Robust baseline with minimal tuning</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='model-card'>
            <h3 style='color: #3498db;'>🚀 XGBoost</h3>
            <p><strong>Type:</strong> Gradient Boosting</p>
            <p><strong>Estimators:</strong> 500 trees</p>
            <p><strong>Learning Rate:</strong> 0.05</p>
            <p><strong>Best for:</strong> Feature-engineered tabular data</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='model-card'>
            <h3 style='color: #e67e22;'>🔗 CNN-LSTM Hybrid</h3>
            <p><strong>Type:</strong> Deep Learning</p>
            <p><strong>Architecture:</strong> 3 Conv1D + 2 LSTM layers</p>
            <p><strong>Parameters:</strong> ~450K</p>
            <p><strong>Best for:</strong> Pattern extraction + temporal modeling</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='model-card'>
            <h3 style='color: #9b59b6;'>🎯 Transformer</h3>
            <p><strong>Type:</strong> Attention-based DL</p>
            <p><strong>Architecture:</strong> 4 layers, 8 attention heads</p>
            <p><strong>Parameters:</strong> ~380K</p>
            <p><strong>Best for:</strong> Long-range dependencies (larger datasets)</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⚠️ No model results found. Please train models first.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #95a5a6;'>Model Arena | Compare, Analyze, Optimize</p>",
            unsafe_allow_html=True)
