import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from src.config import config

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=config.RANDOM_STATE,
            tree_method='hist',
            n_jobs=-1
        )
        self.name = "XGBoost"
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        joblib.dump(self.model, path)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=400,
            max_depth=15,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        self.name = "Random Forest"
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        joblib.dump(self.model, path)
