import plotly.graph_objects as go
import streamlit as st

def render_forecast_charts(dates: list, model_predictions: dict, selected_models: list, forecast_days: int) -> None:
    """
    Renders 4 separate stacked charts for Multivariate climate targets,
    following industry standard presentation practices.
    """
    st.markdown(f"### Multivariate Climate Forecast — Next {forecast_days} Days")
    
    targets = [
        ("Temperature Forecast", "temp_C", "°C", "🌡️"),
        ("Humidity Forecast", "humidity_pct", "%", "💧"),
        ("Rainfall Forecast", "rainfall_mm", "mm", "🌧️"),
        ("Solar Radiation Forecast", "solar_MJ", "MJ/m²", "☀️")
    ]
    
    # Assign a consistent, bright climate-inspired color to each model so the legend works across all targets.
    # Emerald, Sunset Orange, Ocean Blue, Violet, Sun Yellow
    palette = ["#10b981", "#f97316", "#0ea5e9", "#8b5cf6", "#facc15"]
    
    for idx_t, (title, key, ylabel, icon) in enumerate(targets):
        fig = go.Figure()
        
        # Only show legend on the top chart to prevent 4 duplicate legends
        show_legend = (idx_t == 0)
        
        for idx_m, model_name in enumerate(selected_models):
            color = palette[idx_m % len(palette)]
            fig.add_trace(go.Scatter(
                x=dates, 
                y=model_predictions[model_name][key],
                mode="lines+markers", 
                name=model_name,
                legendgroup=model_name,
                showlegend=show_legend,
                line=dict(color=color, width=3, shape='spline', smoothing=0.8),
                marker=dict(size=6, color="#000000", line=dict(width=2, color=color)),
                hoverinfo="y+name"
            ))
            
        fig.update_layout(
            template="plotly_dark",
            font=dict(family="Space Grotesk, sans-serif", color="#a3a3a3"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=70, b=20),
            title=dict(
                text=f"<b>{icon} {title}</b>",
                x=0.01,
                y=0.95,
                font=dict(size=18, color="#ffffff")
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#050505",
                font_size=13,
                font_family="Inter, sans-serif",
                bordercolor="#262626"
            ),
            yaxis_title=dict(text=ylabel, font=dict(color="#737373")),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                font=dict(color="#ffffff")
            ) if show_legend else None,
            showlegend=show_legend
        )
        
        fig.update_xaxes(showgrid=False, tickfont=dict(color="#737373"))
        fig.update_yaxes(showgrid=True, gridcolor='#1f1f1f', zeroline=False, tickfont=dict(color="#737373"))
        
        st.plotly_chart(fig, use_container_width=True)
