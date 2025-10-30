import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_PATH: Path = BASE_DIR / "data" / "pune_climate_with_co2.csv"
    MODEL_DIR: Path = BASE_DIR / "models"
    TARGET: str = "temp_C"
    DATE_COL: str = "date"
    FEATURES: List[str] = None
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    LOOKBACK: int = 24
    BATCH_SIZE: int = 32
    EPOCHS: int = 200
    PATIENCE: int = 25
    LAG_FEATURES: List[int] = None
    ROLLING_WINDOWS: List[int] = None

    def __post_init__(self):
        if self.FEATURES is None:
            self.FEATURES = ["temp_C", "humidity_pct", "rainfall_mm", "solar_MJ", "co2_ppm"]
        if self.LAG_FEATURES is None:
            self.LAG_FEATURES = [1, 6, 12]
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [3, 6, 12]
        self.MODEL_DIR.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            TARGET=cfg['data']['target'],
            DATE_COL=cfg['data']['date_col'],
            FEATURES=cfg['data']['features'],
            TEST_SIZE=cfg['training']['test_size'],
            VAL_SIZE=cfg['training']['val_size'],
            LOOKBACK=cfg['training']['lookback'],
            BATCH_SIZE=cfg['training']['batch_size'],
            EPOCHS=cfg['training']['epochs'],
            PATIENCE=cfg['training']['patience'],
            LAG_FEATURES=cfg['training']['lag_features'],
            ROLLING_WINDOWS=cfg['training']['rolling_windows']
        )

config = Config()
