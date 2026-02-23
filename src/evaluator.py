"""
Model evaluation utilities.

Computes standard regression metrics (RMSE, MAE, R²) and
logs them via the project logger.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .logger import logger


class Evaluator:
    """Stateless evaluator for regression models."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> Dict[str, float]:
        """Compute and log RMSE, MAE, and R² for *model_name*.

        Parameters
        ----------
        y_true : array-like
            Ground-truth target values.
        y_pred : array-like
            Model predictions (same length as *y_true*).
        model_name : str
            Human-readable model identifier for log output.

        Returns
        -------
        dict[str, float]
            ``{"rmse": …, "mae": …, "r2": …}``
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   RMSE: {rmse:.4f}°C")
        logger.info(f"   MAE:  {mae:.4f}°C")
        logger.info(f"   R²:   {r2:.4f}")

        return {"rmse": rmse, "mae": mae, "r2": r2}
