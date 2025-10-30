import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import config

class SequenceGenerator:
    def __init__(self, lookback=config.LOOKBACK):
        self.lookback = lookback
    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

class CNNLSTMModel:
    def __init__(self):
        self.model = None
        self.name = "CNN-LSTM Hybrid"
        self.history = None
    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0005),
            loss='mse',
            metrics=['mae']
        )
    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

class TransformerModel:
    def __init__(self):
        self.model = None
        self.name = "Transformer + Attention"
        self.history = None
    def build_model(self, input_shape):
        def transformer_block(x, d_model, num_heads, dff, dropout=0.2):
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model//num_heads
            )(x, x)
            attn_output = layers.Dropout(dropout)(attn_output)
            out1 = layers.LayerNormalization()(x + attn_output)
            ffn = keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dense(d_model)
            ])
            ffn_output = ffn(out1)
            ffn_output = layers.Dropout(dropout)(ffn_output)
            return layers.LayerNormalization()(out1 + ffn_output)
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(128)(inputs)
        for _ in range(4):
            x = transformer_block(x, d_model=128, num_heads=8, dff=512, dropout=0.2)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0003),
            loss='mse',
            metrics=['mae']
        )
    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()
