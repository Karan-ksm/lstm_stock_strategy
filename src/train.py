"""Training entry point.

Orchestrates the full offline pipeline for each ticker in config:

  1. Download or load OHLCV data.
  2. Build features with the same pipeline used at paper trading time.
  3. Compute the directional target on the close price.
  4. Chronological train, val, test split (no shuffling across time).
  5. Fit the scaler on the training split ONLY.
  6. Build (X, y) sequences per split independently so no window
     bleeds across a train, val, or test boundary.
  7. Train the LSTM with early stopping on validation loss.
  8. Save the trained model, the fitted scaler, and a ModelSpec
     describing exactly how features were built, as a single bundle.

Call as `python -m src.train` or via scripts/run_train.py.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.config import Config, load_config
from src.data.downloader import download_one
from src.dataset.scaler import FeatureScaler
from src.dataset.sequences import build_sequences, make_directional_target
from src.dataset.splits import chronological_split
from src.features.pipeline import build_features
from src.model.lstm import build_model
from src.model.persistence import ModelSpec, save_bundle
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir

_LOGGER = get_logger(__name__)

_TARGET_COL = "__target__"


def _set_random_seeds(seed: int) -> None:
    """Seed Python, numpy, and tensorflow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one_ticker(ticker: str, config: Config) -> Path:
    """Run the full training pipeline for one ticker and save the bundle.

    Returns:
        The directory where the model bundle was written.
    """
    _LOGGER.info("=== Training pipeline for %s ===", ticker)

    # 1. Data
    df = download_one(
        ticker=ticker,
        start=config.data.start_date,
        end=config.data.end_date,
        cache_dir=config.data.cache_dir,
        auto_adjust=config.data.auto_adjust,
    )
    _LOGGER.info("Loaded %d OHLCV rows for %s", len(df), ticker)

    # 2. Features
    features = build_features(
        df, config.features.enabled, drop_warmup=True
    )
    _LOGGER.info(
        "Built %d feature rows with %d columns",
        len(features),
        len(features.columns),
    )

    # 3. Target (close price aligned to the post warm up feature index)
    target = make_directional_target(
        df["close"],
        horizon=config.target.horizon_days,
        up_threshold=config.target.up_threshold,
    ).loc[features.index]

    # 4. Chronological split. Combining features and target so they stay
    #    aligned through the split call.
    combined = features.copy()
    combined[_TARGET_COL] = target
    train_df, val_df, test_df = chronological_split(
        combined,
        train_fraction=config.dataset.train_fraction,
        val_fraction=config.dataset.val_fraction,
        test_fraction=config.dataset.test_fraction,
    )
    _LOGGER.info(
        "Split sizes: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    feat_cols = list(features.columns)
    train_features = train_df[feat_cols]
    val_features = val_df[feat_cols]
    test_features = test_df[feat_cols]
    train_target = train_df[_TARGET_COL]
    val_target = val_df[_TARGET_COL]
    test_target = test_df[_TARGET_COL]

    # 5. Scale. fit() on train only; transform() on every split.
    scaler = FeatureScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    # 6. Sequences per split. No window crosses a split boundary.
    lookback = config.dataset.lookback_window
    X_train, y_train, _ = build_sequences(train_scaled, train_target, lookback)
    X_val, y_val, _ = build_sequences(val_scaled, val_target, lookback)
    X_test, y_test, _ = build_sequences(test_scaled, test_target, lookback)
    _LOGGER.info(
        "Sequence shapes: X_train=%s X_val=%s X_test=%s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # 7. Model
    model = build_model(
        config.model,
        lookback=lookback,
        n_features=X_train.shape[2],
    )
    _LOGGER.info("Model built; parameter count: %d", model.count_params())

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        shuffle=config.dataset.shuffle_train_batches,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    _LOGGER.info("Test metrics for %s: %s", ticker, test_metrics)

    # 8. Save bundle
    out_dir = ensure_dir(Path(config.training.models_dir) / ticker.upper())
    spec = ModelSpec(
        ticker=ticker.upper(),
        features=feat_cols,
        lookback=lookback,
        horizon_days=config.target.horizon_days,
        up_threshold=config.target.up_threshold,
        target_type=config.target.type,
        n_features=len(feat_cols),
        trained_on=datetime.now(timezone.utc).isoformat(),
    )
    save_bundle(out_dir, model, scaler, spec)
    _LOGGER.info("Saved model bundle for %s to %s", ticker, out_dir)
    return out_dir


def main(config_path: str | None = None) -> None:
    """Run training for every ticker in the config."""
    config = load_config(config_path)
    _set_random_seeds(config.project.random_seed)
    for ticker in config.data.tickers:
        train_one_ticker(ticker, config)


if __name__ == "__main__":
    main()
