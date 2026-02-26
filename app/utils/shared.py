"""
Shared utilities for the Streamlit application.

Provides centralised CSS loading, data loading with caching,
and reusable HTML component helpers.  All styling is loaded from
``app/static/styles.css`` — no inline CSS lives in Python code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Project paths (resolve once) ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "app" / "static"


def load_css() -> None:
    """Inject the external stylesheet into the Streamlit page.

    Reads ``app/static/styles.css`` once and injects it via a single
    ``<style>`` block rather than scattering inline styles across pages.
    """
    css_path = STATIC_DIR / "styles.css"
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ styles.css not found — UI may look unstyled.")


def load_results() -> pd.DataFrame:
    """Load model performance metrics from ``results/test_metrics.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: Model, RMSE, MAE, R2.  Empty DataFrame on error.
    """
    path = RESULTS_DIR / "test_metrics.csv"
    empty = pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning("⚠️ Results file not found. Train models first: `python train.py`")
        return empty
    except Exception as exc:
        st.error(f"Error loading results: {exc}")
        return empty


@st.cache_data(show_spinner=False)
def load_climate_data() -> Optional[pd.DataFrame]:
    """Load the Pune climate dataset from ``data/pune_climate_with_co2.csv``.

    Returns
    -------
    pd.DataFrame or None
        Climate data with ``date`` parsed as datetime, or *None* on error.
    """
    path = DATA_DIR / "pune_climate_with_co2.csv"
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        st.error("❌ Data file not found — ensure `pune_climate_with_co2.csv` is in `data/`.")
        return None
    except Exception as exc:
        st.error(f"Error loading climate data: {exc}")
        return None


# ── Reusable HTML components (CSS class–based, no inline styles) ──────────────

def hero_section(title: str, subtitle: str, detail: str = "", variant: str = "") -> None:
    """Render a hero banner using CSS classes from styles.css.

    Parameters
    ----------
    title : str
        Main heading (rendered as ``<h1>``).
    subtitle : str
        Sub-heading (rendered as ``<p>``).
    detail : str, optional
        Smaller detail line.
    variant : str, optional
        Extra CSS class for colour variants (e.g. ``hero-purple``).
    """
    detail_html = f"<p>{detail}</p>" if detail else ""
    st.markdown(
        f"<div class='hero {variant} fade-in'>"
        f"<h1>{title}</h1><p>{subtitle}</p>{detail_html}</div>",
        unsafe_allow_html=True,
    )


def gradient_card(title: str, value: str, subtitle: str = "", variant: str = "") -> str:
    """Return HTML for a gradient stat card. Now mapped to the premium metric-card CSS.

    Parameters
    ----------
    title : str
        Card heading.
    value : str
        Large value text.
    subtitle : str, optional
        Small description below the value.
    variant : str
        Passed string to determine accent color.
    """
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    
    # Map old variants to corresponding modern climate accent hex colors
    color = "#3b82f6"  # Default Blue
    if "purple" in variant: color = "#8b5cf6"
    elif "pink" in variant: color = "#ec4899"
    elif "red" in variant: color = "#ef4444"
    elif "green" in variant or "teal" in variant: color = "#10b981"
    elif "orange" in variant: color = "#f97316"
    elif "ocean" in variant: color = "#0ea5e9"
    
    return (
        f"<div class='metric-card' style='border-top-color: {color};'>"
        f"<h3>{title}</h3><h2>{value}</h2>{sub}</div>"
    )


def footer(text: str = "Climate Intelligence Platform", sub: str = "") -> None:
    """Render a page footer."""
    sub_html = f"<p class='sub'>{sub}</p>" if sub else ""
    st.markdown(
        f"<div class='footer hero-purple'>"
        f"<p>🌱 {text}</p>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str) -> None:
    """Show a styled empty / error state."""
    st.markdown(
        f"<div class='empty-state gc-red'>"
        f"<h2>{title}</h2><p>{message}</p></div>",
        unsafe_allow_html=True,
    )
