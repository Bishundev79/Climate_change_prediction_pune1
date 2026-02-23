"""
Data ingestion, cleaning, resampling, and train/val/test splitting.

This module provides the :class:`DataPipeline` class which owns the
entire data preparation flow—from raw CSV to analysis-ready splits.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.config import config


class DataPipeline:
    """End-to-end data preparation pipeline.

    Responsibilities
    ----------------
    * Load raw daily CSV data.
    * Resample to monthly frequency.
    * Impute missing values via linear interpolation + back-fill.
    * Split chronologically into train / validation / test.
    """

    AGGREGATION_MAP: dict[str, str] = {
        "temp_C": "mean",
        "humidity_pct": "mean",
        "rainfall_mm": "sum",
        "solar_MJ": "mean",
        "co2_ppm": "mean",
    }

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load CSV, resample monthly, impute, and split.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train, val, test)`` DataFrames sorted chronologically.
        """
        df = pd.read_csv(config.DATA_PATH)
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

        df_monthly = (
            df.set_index(config.DATE_COL)
            .resample("MS")
            .agg(self.AGGREGATION_MAP)
            .reset_index()
        )

        # Impute: linear interpolation then back-fill remaining edge NaNs
        df_clean = df_monthly.interpolate(method="linear").bfill()

        return self._split_data(df_clean)

    @staticmethod
    def _split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological train / val / test split (no shuffling).

        Uses ``config.TEST_SIZE`` and ``config.VAL_SIZE`` to compute
        split indices.
        """
        n = len(df)
        train_end = int(n * (1 - config.TEST_SIZE - config.VAL_SIZE))
        val_end = int(n * (1 - config.TEST_SIZE))
        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:val_end].copy(),
            df.iloc[val_end:].copy(),
        )
