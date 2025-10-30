import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="🌍 Pune Climate Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, nature-inspired design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .hero { background: linear-gradient(120deg, #2ecc71 0%, #27ae60 100%);
            padding: 3rem 2rem; border-radius: 20px; color: white;
            text-align: center; margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);}
    .hero h1 { font-size: 3.5rem; font-weight: 700;
        margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.15);}
    .metric-card {background: white; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.09); transition: transform 0.3s;}
    .metric-card:hover {transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.11);}
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton > button { background: linear-gradient(90deg, #56ab2f, #a8e063);
        color: white; border:none; border-radius:25px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results():
    try:
        return pd.read_csv('results/test_metrics.csv')
    except:
        return pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])

@st.cache_data
def load_climate_data():
    try:
        df = pd.read_csv('data/pune_climate_with_co2.csv', parse_dates=['date'])
        return df
    except:
        return None

results = load_results()
climate_df = load_climate_data()

# Sidebar: All navigation and quick metrics here
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>🌡️ Climate Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if not results.empty:
        best_model = results.iloc[0]
        st.metric("Best Model", best_model['Model'])
        st.metric("Best RMSE", f"{best_model['RMSE']}°C")
        st.metric("Best R²", f"{best_model['R2']}")
    st.markdown("---")
    st.markdown("### 🎯 Navigation")
    st.page_link("streamlit_app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/01__Overview.py", label="📊 Overview", icon="📊")
    st.page_link("pages/02__Data_Explorer.py", label="🔍 Data Explorer", icon="🔍")
    st.page_link("pages/03__Model_Arena.py", label="🤖 Model Arena", icon="🤖")
    st.page_link("pages/04__Forecast.py", label="🔮 Forecast", icon="🔮")

# Hero Section
st.markdown("""
<div class='hero'>
    <h1>🌍 Pune Climate Intelligence</h1>
    <p>Advanced Machine Learning for Climate Forecasting | 1951-2024</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #27ae60; margin:0'>📅 Data Span</h3>
        <h2 style='margin:0.5rem 0'>73 Years</h2>
        <p style='color: #7f8c8d; margin:0'>1951 - 2024</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #3498db; margin:0'>🤖 Models</h3>
        <h2 style='margin:0.5rem 0'>4 AI Models</h2>
        <p style='color: #7f8c8d; margin:0'>2 ML + 2 DL</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #e74c3c; margin:0'>🎯 Accuracy</h3>
        <h2 style='margin:0.5rem 0'>0.80°C</h2>
        <p style='color: #7f8c8d; margin:0'>Best RMSE</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #9b59b6; margin:0'>📊 Records</h3>
        <h2 style='margin:0.5rem 0'>876K+</h2>
        <p style='color: #7f8c8d; margin:0'>Data Points</p>
    </div>
    """, unsafe_allow_html=True)

# Model Performance Section
st.markdown("## 🏆 Model Performance Leaderboard")
if not results.empty:
    results_sorted = results.sort_values("RMSE")
    colors = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_sorted['Model'],
        y=results_sorted['RMSE'],
        marker=dict(color=colors[:len(results_sorted)], line=dict(color='rgb(8,48,107)', width=1.5)),
        text=results_sorted['RMSE'].round(2),
        textposition='outside',
        name='RMSE (°C)'
    ))
    fig.update_layout(
        template='plotly_white',
        height=400,
        title=dict(text='Model RMSE Comparison', font=dict(size=20, color='#2c3e50')),
        xaxis=dict(title='Model', tickfont=dict(size=12)),
        yaxis=dict(title='RMSE (°C)', tickfont=dict(size=12)),
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📋 Detailed Metrics")
    st.dataframe(
        results_sorted.style.background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE'])
                           .background_gradient(cmap='RdYlGn', subset=['R2'])
                           .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R2': '{:.4f}'}),
        use_container_width=True,
        height=200
    )
else:
    st.info("⚠️ No model results found. Please train models first using `python train.py`")

if climate_df is not None:
    st.markdown("## 🌡️ Climate Trends Overview")
    monthly = climate_df.set_index('date').resample('MS').mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index,
        y=monthly['temp_C'],
        mode='lines',
        name='Temperature',
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    fig.update_layout(
        template='plotly_white',
        height=400,
        title='Temperature Trend (1951-2024)',
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p style='font-size: 0.9rem;'>🌱 Built with Machine Learning & Climate Science | 📊 Real-time Data Analytics</p>
    <p style='font-size: 0.8rem;'>© 2024 Pune Climate Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
