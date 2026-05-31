"""Configuration loader.

Reads config.yaml from the project root and exposes a typed Config
dataclass tree used throughout the codebase. Validation lives here so
the rest of the code can trust the values it receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Top level project settings."""

    random_seed: int
    log_level: str


@dataclass(frozen=True)
class DataConfig:
    """Where market data comes from and how it is cached."""

    tickers: list[str]
    start_date: str
    end_date: str
    cache_dir: str
    source: str
    auto_adjust: bool


@dataclass(frozen=True)
class FeaturesConfig:
    """Which features to build, plus any extra indicator parameters.

    Current indicators encode their window inside the feature name
    (for example sma_10), so `params` is empty by default and is
    reserved for future indicators that need richer configuration.
    """

    enabled: list[str]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetConfig:
    """How the supervised label is defined."""

    type: str
    horizon_days: int
    up_threshold: float


@dataclass(frozen=True)
class DatasetConfig:
    """Sequence length and chronological split fractions."""

    lookback_window: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    shuffle_train_batches: bool

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train, val, and test fractions must sum to 1.0, got {total}"
            )
        if self.lookback_window < 1:
            raise ValueError("lookback_window must be at least 1")


@dataclass(frozen=True)
class LayerConfig:
    """A single recurrent layer in the LSTM stack."""

    units: int
    dropout: float
    return_sequences: bool


@dataclass(frozen=True)
class ModelConfig:
    """LSTM architecture and compilation settings."""

    type: str
    layers: list[LayerConfig]
    dense_units: int
    output_activation: str
    loss: str
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop hyperparameters and where to save weights."""

    epochs: int
    batch_size: int
    early_stopping_patience: int
    models_dir: str


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds that map model probability to a trade signal."""

    buy_threshold: float
    sell_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.sell_threshold <= self.buy_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= sell_threshold <= buy_threshold <= 1"
            )


@dataclass(frozen=True)
class CostsConfig:
    """Trading frictions modeled in the backtest."""

    commission_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest engine inputs."""

    signal: SignalConfig
    costs: CostsConfig
    initial_cash: float
    reports_dir: str


@dataclass(frozen=True)
class PaperTradeConfig:
    """Inputs for the daily Alpaca paper trading run."""

    poll_ticker: str
    cash_allocation_fraction: float
    logs_dir: str


@dataclass(frozen=True)
class Config:
    """Fully typed view of config.yaml."""

    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    target: TargetConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    backtest: BacktestConfig
    paper_trade: PaperTradeConfig


def _project_root() -> Path:
    """Return the repository root (one level above src/)."""
    return Path(__file__).resolve().parent.parent


def _coerce_date(value: Any) -> str:
    """Accept either a YAML date object or a string and return ISO text."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config.

    Args:
        path: Optional explicit path to a config file. Defaults to
            <project_root>/config.yaml.

    Returns:
        A fully typed Config tree.

    Raises:
        FileNotFoundError: If the YAML file is missing.
        ValueError: If any validation rule fails.
    """
    config_path = Path(path) if path else _project_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with config_path.open("r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    data_raw = dict(raw["data"])
    data_raw["start_date"] = _coerce_date(data_raw["start_date"])
    data_raw["end_date"] = _coerce_date(data_raw["end_date"])

    model_raw = dict(raw["model"])
    layers_raw = model_raw.pop("layers")
    layers = [LayerConfig(**layer) for layer in layers_raw]
    model_cfg = ModelConfig(layers=layers, **model_raw)

    backtest_raw = raw["backtest"]
    backtest_cfg = BacktestConfig(
        signal=SignalConfig(**backtest_raw["signal"]),
        costs=CostsConfig(**backtest_raw["costs"]),
        initial_cash=backtest_raw["initial_cash"],
        reports_dir=backtest_raw["reports_dir"],
    )

    return Config(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(**data_raw),
        features=FeaturesConfig(**raw["features"]),
        target=TargetConfig(**raw["target"]),
        dataset=DatasetConfig(**raw["dataset"]),
        model=model_cfg,
        training=TrainingConfig(**raw["training"]),
        backtest=backtest_cfg,
        paper_trade=PaperTradeConfig(**raw["paper_trade"]),
    )
