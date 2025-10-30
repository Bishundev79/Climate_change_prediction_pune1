import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Overview | Pune Climate", page_icon="📊", layout="wide")

# Custom CSS (same premium styling)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .section-header {
        color: #2c3e50;
        font-weight: 700;
        font-size: 2rem;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #27ae60;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/pune_climate_with_co2.csv', parse_dates=['date'])
        results = pd.read_csv('results/test_metrics.csv')
        return df, results
    except:
        return None, None


df, results = load_data()

st.markdown("<h1 style='text-align: center; color: #27ae60;'>📊 Project Overview</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>Comprehensive Analysis of 73 Years of Climate Data</p>",
    unsafe_allow_html=True)

if df is not None:
    # Dataset Summary
    st.markdown("<div class='section-header'>📁 Dataset Information</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <h3 style='color: #3498db; margin: 0;'>📅</h3>
            <h2 style='margin: 0.5rem 0;'>{len(df):,}</h2>
            <p style='color: #7f8c8d; margin: 0;'>Total Records</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <h3 style='color: #e74c3c; margin: 0;'>🌡️</h3>
            <h2 style='margin: 0.5rem 0;'>{df['temp_C'].mean():.1f}°C</h2>
            <p style='color: #7f8c8d; margin: 0;'>Avg Temperature</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <h3 style='color: #9b59b6; margin: 0;'>💧</h3>
            <h2 style='margin: 0.5rem 0;'>{df['humidity_pct'].mean():.1f}%</h2>
            <p style='color: #7f8c8d; margin: 0;'>Avg Humidity</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <h3 style='color: #16a085; margin: 0;'>🌧️</h3>
            <h2 style='margin: 0.5rem 0;'>{df['rainfall_mm'].mean():.1f}mm</h2>
            <p style='color: #7f8c8d; margin: 0;'>Avg Rainfall</p>
        </div>
        """, unsafe_allow_html=True)

    # Time Range
    st.markdown("<div class='section-header'>⏰ Temporal Coverage</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric-card'>
        <p style='font-size: 1.1rem; margin: 0;'>
            <strong>Start Date:</strong> {df['date'].min().strftime('%B %d, %Y')} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>End Date:</strong> {df['date'].max().strftime('%B %d, %Y')} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Duration:</strong> {(df['date'].max() - df['date'].min()).days // 365} Years
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Climate Trends
    st.markdown("<div class='section-header'>📈 Long-term Climate Trends</div>", unsafe_allow_html=True)

    monthly = df.set_index('date').resample('YS').mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['temp_C'],
        mode='lines+markers', name='Temperature',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['humidity_pct'],
        mode='lines+markers', name='Humidity',
        line=dict(color='#3498db', width=3),
        marker=dict(size=6),
        yaxis='y2'
    ))

    fig.update_layout(
        template='plotly_white',
        height=500,
        title='Yearly Averages: Temperature & Humidity',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Temperature (°C)', titlefont=dict(color='#e74c3c')),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right', titlefont=dict(color='#3498db')),
        hovermode='x unified',
        legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center')
    )

    st.plotly_chart(fig, use_container_width=True)

# Model Performance
if results is not None and not results.empty:
    st.markdown("<div class='section-header'>🏆 Model Performance Summary</div>", unsafe_allow_html=True)

    best = results.iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
            <h3 style='margin: 0; color: white;'>🥇 Best Model</h3>
            <h1 style='margin: 1rem 0; color: white;'>{best['Model']}</h1>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>RMSE: <strong>{best['RMSE']}°C</strong></p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>R²: <strong>{best['R2']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        colors = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']

        fig.add_trace(go.Bar(
            x=results['Model'],
            y=results['R2'],
            marker=dict(color=colors[:len(results)]),
            text=results['R2'].round(3),
            textposition='outside'
        ))

        fig.update_layout(
            template='plotly_white',
            height=300,
            title='Model R² Score Comparison',
            yaxis=dict(title='R² Score', range=[0, 1]),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #95a5a6;'>Built with Streamlit & Plotly | Climate Intelligence Platform</p>",
    unsafe_allow_html=True)
