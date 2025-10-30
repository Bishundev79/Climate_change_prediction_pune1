import pandas as pd
import numpy as np
from src.config import config

class FeatureEngine:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()
        for col in config.FEATURES:
            if col in df_feat.columns:
                for lag in config.LAG_FEATURES:
                    df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
        for col in config.FEATURES:
            if col in df_feat.columns:
                for window in config.ROLLING_WINDOWS:
                    df_feat[f'{col}_roll{window}'] = df_feat[col].rolling(
                        window, min_periods=1).mean()
        df_feat['month'] = df_feat[config.DATE_COL].dt.month
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat = df_feat.dropna()
        feature_cols = [c for c in df_feat.columns
                        if c not in [config.DATE_COL, config.TARGET]]
        return df_feat, feature_cols
