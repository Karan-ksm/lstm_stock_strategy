"""Save and load a trained model bundle.

A "bundle" is everything paper trading needs to reproduce the exact
inputs the model saw during training:

    artifacts/models/AAPL/
    ├── model.keras    Keras native format
    ├── scaler.pkl     Pickled FeatureScaler fitted on train only
    └── spec.json      List of feature names, lookback, target settings

Tying these three together prevents the most insidious live trading
bug: features computed in a different order, with a different scaler,
or against a different lookback than at training time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

from src.dataset.scaler import FeatureScaler
from src.utils.paths import ensure_dir, resolve


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to feed a live bar through the model.

    Stored alongside the model and scaler. Updated when any of these
    fields would change so the bundle stays self consistent.
    """

    ticker: str
    features: list[str]
    lookback: int
    horizon_days: int
    up_threshold: float
    target_type: str
    n_features: int
    trained_on: str


def save_bundle(
    out_dir: str | Path,
    model: tf.keras.Model,
    scaler: FeatureScaler,
    spec: ModelSpec,
) -> Path:
    """Write model, scaler, and spec into a single directory.

    Args:
        out_dir: Directory to write into. Created if missing.
        model: Compiled and trained Keras model.
        scaler: Fitted FeatureScaler.
        spec: ModelSpec describing how features were built.

    Returns:
        The absolute path of the bundle directory.
    """
    if not scaler.fitted:
        raise ValueError("Refusing to save an unfitted scaler in the bundle")

    out = ensure_dir(out_dir)
    model.save(out / "model.keras")
    scaler.save(out / "scaler.pkl")
    with (out / "spec.json").open("w") as handle:
        json.dump(asdict(spec), handle, indent=2)
    return out


def load_bundle(
    in_dir: str | Path,
) -> tuple[tf.keras.Model, FeatureScaler, ModelSpec]:
    """Load model, scaler, and spec from a bundle directory.

    Args:
        in_dir: Directory previously written by save_bundle.

    Returns:
        (model, scaler, spec) tuple.

    Raises:
        FileNotFoundError: If any of the three bundle files is missing.
    """
    bundle = resolve(in_dir)
    model_path = bundle / "model.keras"
    scaler_path = bundle / "scaler.pkl"
    spec_path = bundle / "spec.json"
    for path in (model_path, scaler_path, spec_path):
        if not path.exists():
            raise FileNotFoundError(f"Bundle is missing {path.name} at {bundle}")

    model = tf.keras.models.load_model(model_path)
    scaler = FeatureScaler.load(scaler_path)
    with spec_path.open("r") as handle:
        spec_raw = json.load(handle)
    spec = ModelSpec(**spec_raw)
    return model, scaler, spec
