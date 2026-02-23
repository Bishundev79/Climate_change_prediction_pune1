"""
Deep-learning model wrappers (CNN-LSTM Hybrid, Transformer).

Hyperparameters are read from ``config.yaml`` via :pydata:`src.config.config`
so that architecture changes don't require touching code.

Requires TensorFlow ≥ 2.12.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.config import config

# TensorFlow is imported lazily so that ML-only workflows still func­tion
# on machines without a GPU or without TF installed.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SequenceGenerator:
    """Convert tabular arrays into sliding-window sequences for DL models.

    Parameters
    ----------
    lookback : int
        Number of past time-steps to include in each input sample.
    """

    def __init__(self, lookback: int = config.LOOKBACK) -> None:
        self.lookback = lookback

    def create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(X, y)`` arrays with shape ``(n, lookback, features)``."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback : i])
            y.append(target[i])
        return np.array(X), np.array(y)


class CNNLSTMModel:
    """1-D Convolution → LSTM hybrid for time-series regression.

    Architecture parameters are read from ``config.yaml`` under
    ``models.cnn_lstm``.
    """

    def __init__(self) -> None:
        self.model: keras.Model | None = None
        self.name = "CNN-LSTM Hybrid"
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Construct the Keras model from config-driven hyperparameters."""
        cfg = config.MODEL_PARAMS.get("cnn_lstm", {})
        filters = cfg.get("cnn_filters", [64, 128, 64])
        lstm_units = cfg.get("lstm_units", [128, 64])

        inputs = layers.Input(shape=input_shape)

        x = inputs
        for i, f in enumerate(filters):
            x = layers.Conv1D(f, 3, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            if i < len(filters) - 1:
                x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        for j, u in enumerate(lstm_units):
            return_seq = j < len(lstm_units) - 1
            x = layers.LSTM(u, return_sequences=return_seq)(x)
            x = layers.Dropout(0.3 if return_seq else 0.2)(x)

        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="mse",
            metrics=["mae"],
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train with early stopping and learning-rate reduction."""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=config.PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
        ]
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return flattened predictions."""
        return self.model.predict(X, verbose=0).flatten()


class TransformerModel:
    """Multi-head self-attention model for time-series regression.

    Architecture parameters are read from ``config.yaml`` under
    ``models.transformer``.
    """

    def __init__(self) -> None:
        self.model: keras.Model | None = None
        self.name = "Transformer + Attention"
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Construct the Keras model from config-driven hyperparameters."""
        cfg = config.MODEL_PARAMS.get("transformer", {})
        d_model = cfg.get("d_model", 128)
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_layers", 4)
        dff = cfg.get("dff", 512)
        dropout_rate = cfg.get("dropout", 0.2)

        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(d_model)(inputs)

        for _ in range(num_layers):
            x = self._transformer_block(x, d_model, num_heads, dff, dropout_rate)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(d_model, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="mse",
            metrics=["mae"],
        )

    @staticmethod
    def _transformer_block(
        x: tf.Tensor,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.2,
    ) -> tf.Tensor:
        """Single Transformer encoder block (multi-head attention + FFN)."""
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        out1 = layers.LayerNormalization()(x + attn_output)

        ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        ffn_output = layers.Dropout(dropout)(ffn(out1))
        return layers.LayerNormalization()(out1 + ffn_output)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train with early stopping and learning-rate reduction."""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=config.PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
        ]
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return flattened predictions."""
        return self.model.predict(X, verbose=0).flatten()
