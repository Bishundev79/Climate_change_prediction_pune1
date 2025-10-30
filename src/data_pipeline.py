import pandas as pd
import numpy as np
from typing import Tuple
from src.config import config

class DataPipeline:
    def __init__(self):
        self.scaler = None

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(config.DATA_PATH)
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
        df_monthly = df.set_index(config.DATE_COL).resample('MS').agg({
            'temp_C': 'mean',
            'humidity_pct': 'mean',
            'rainfall_mm': 'sum',
            'solar_MJ': 'mean',
            'co2_ppm': 'mean'
        }).reset_index()
        df_clean = df_monthly.interpolate(method='linear').fillna(method='bfill')
        return self._split_data(df_clean)

    def _split_data(self, df: pd.DataFrame) -> Tuple:
        n = len(df)
        train_end = int(n * (1 - config.TEST_SIZE - config.VAL_SIZE))
        val_end = int(n * (1 - config.TEST_SIZE))
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        return train, val, test
