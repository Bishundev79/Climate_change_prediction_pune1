"""
Machine-learning model wrappers (XGBoost, Random Forest).

Hyperparameters are read from ``config.yaml`` via :pydata:`src.config.config`
so that experiments can be changed declaratively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.config import config


class XGBoostModel:
    """Gradient-boosted tree regressor (XGBoost).

    Hyperparameters are loaded from ``config.yaml`` under
    ``models.xgboost``.
    """

    def __init__(self) -> None:
        model_cfg = config.MODEL_PARAMS.get("xgboost", {})
        self.model = xgb.XGBRegressor(
            n_estimators=model_cfg.get("n_estimators", 500),
            max_depth=model_cfg.get("max_depth", 8),
            learning_rate=model_cfg.get("learning_rate", 0.05),
            subsample=model_cfg.get("subsample", 0.8),
            random_state=config.RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )
        self.name = "XGBoost"

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit the model with early-stopping on *eval_set*."""
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Persist model to disk via ``joblib``."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.name = "XGBoost"
        return instance


class RandomForestModel:
    """Ensemble of decision trees (scikit-learn).

    Hyperparameters are loaded from ``config.yaml`` under
    ``models.random_forest``.
    """

    def __init__(self) -> None:
        model_cfg = config.MODEL_PARAMS.get("random_forest", {})
        self.model = RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 400),
            max_depth=model_cfg.get("max_depth", 15),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        self.name = "Random Forest"

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model.  *X_val*/*y_val* kept for API symmetry."""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Persist model to disk via ``joblib``."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.name = "Random Forest"
        return instance
