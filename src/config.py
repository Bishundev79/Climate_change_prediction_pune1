"""
Centralised configuration loaded from ``config.yaml``.

All hyper-parameters (data paths, training settings, **model
architectures**) are defined declaratively in YAML and exposed
via the module-level :pydata:`config` singleton.  Code never
hard-codes magic numbers — it reads them from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    """Project-wide configuration with sane defaults.

    Attributes are overridden by values in ``config.yaml`` when
    :meth:`from_yaml` is used.
    """

    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_PATH: Path = BASE_DIR / "data" / "pune_climate_with_co2.csv"
    MODEL_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"

    TARGETS: Optional[List[str]] = None
    DATE_COL: str = "date"
    FEATURES: Optional[List[str]] = None

    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    LOOKBACK: int = 24
    BATCH_SIZE: int = 32
    EPOCHS: int = 200
    PATIENCE: int = 25

    LAG_FEATURES: Optional[List[int]] = None
    ROLLING_WINDOWS: Optional[List[int]] = None

    # Model-specific hyperparams loaded from config.yaml → models section
    MODEL_PARAMS: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.TARGETS is None:
            self.TARGETS = ["temp_C", "humidity_pct", "rainfall_mm", "solar_MJ"]
        if self.FEATURES is None:
            self.FEATURES = ["temp_C", "humidity_pct", "rainfall_mm", "solar_MJ", "co2_ppm"]
        if self.LAG_FEATURES is None:
            self.LAG_FEATURES = [1, 7, 30]
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [7, 30, 90]
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str | Path] = None) -> "Config":
        """Load configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str, Path, or None
            Path to the YAML file.  Defaults to ``config.yaml`` in
            the project root.

        Returns
        -------
        Config
        """
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "config.yaml"

        resolved = Path(yaml_path)

        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        with open(resolved, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        return cls(
            TARGETS=cfg["data"].get("targets", [cfg["data"].get("target")]),
            DATE_COL=cfg["data"]["date_col"],
            FEATURES=cfg["data"]["features"],
            TEST_SIZE=cfg["training"]["test_size"],
            VAL_SIZE=cfg["training"]["val_size"],
            LOOKBACK=cfg["training"]["lookback"],
            BATCH_SIZE=cfg["training"]["batch_size"],
            EPOCHS=cfg["training"]["epochs"],
            PATIENCE=cfg["training"]["patience"],
            LAG_FEATURES=cfg["training"]["lag_features"],
            ROLLING_WINDOWS=cfg["training"]["rolling_windows"],
            MODEL_PARAMS=cfg.get("models", {}),
        )


# ── Module-level singleton ────────────────────────────────────────────────────
try:
    config = Config.from_yaml()
except (FileNotFoundError, KeyError, yaml.YAMLError) as exc:
    print(f"⚠️  Warning: Could not load config.yaml ({exc}). Using defaults.")
    config = Config()
