"""
Feature engineering for climate time-series data.

Creates lag features, rolling-window statistics, and cyclical
month encodings.  All window sizes and lag values are driven by
``config.yaml`` via :pydata:`src.config.config`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import config


class FeatureEngine:
    """Stateless feature transformer for monthly climate data.

    All parameters (lag orders, rolling windows, feature columns) are
    read from the global :class:`Config` instance so that every
    experiment is reproducible without code edits.
    """

    def create_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Augment *df* with engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Monthly resampled climate data that **must** include a
            ``config.DATE_COL`` column and all ``config.FEATURES``.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            ``(df_augmented, feature_column_names)``
        """
        df_feat = df.copy()

        # ── Lag features ──────────────────────────────────────────────
        for col in config.FEATURES:
            if col not in df_feat.columns:
                continue
            for lag in config.LAG_FEATURES:
                df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)

        # ── Rolling-window statistics ─────────────────────────────────
        for col in config.FEATURES:
            if col not in df_feat.columns:
                continue
            for window in config.ROLLING_WINDOWS:
                df_feat[f"{col}_roll{window}"] = (
                    df_feat[col].rolling(window, min_periods=1).mean()
                )

        # ── Cyclical month encoding ───────────────────────────────────
        df_feat["month"] = df_feat[config.DATE_COL].dt.month
        df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
        df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

        # Drop rows with NaN introduced by lags
        df_feat = df_feat.dropna()

        feature_cols = [
            c for c in df_feat.columns if c not in [config.DATE_COL, config.TARGET]
        ]
        return df_feat, feature_cols
