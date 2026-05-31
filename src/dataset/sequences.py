"""Build supervised LSTM training arrays.

Two responsibilities:
  1. Compute a binary up or down label for each date in a price series.
  2. Slice features and labels into sliding windows of length `lookback`.

Two invariants the rest of the code relies on:
  * No window crosses the start of the data; the first `lookback` - 1
    rows can never be used as a sample.
  * The only place future information enters is the label, which looks
    forward exactly `horizon` days into the price series. Features
    inside a window are strictly historical with respect to the
    decision date returned alongside the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def make_directional_target(
    close: pd.Series,
    horizon: int,
    up_threshold: float,
) -> pd.Series:
    """Compute a binary up or down label for each date.

    Args:
        close: Close price series indexed by date.
        horizon: How many trading days to look ahead when computing
            the future return.
        up_threshold: Label is 1 when log(close[t+horizon] / close[t])
            is strictly greater than this value, otherwise 0.

    Returns:
        Float Series indexed identically to `close`. The last `horizon`
        rows are NaN because the future return needed to label them
        does not exist yet.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    future_log_return = np.log(close.shift(-horizon) / close)
    label = (future_log_return > up_threshold).astype(float)
    return label.where(future_log_return.notna(), np.nan)


def build_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window of `lookback` rows across the input and emit (X, y, dates).

    For a given sample i, the window spans rows [i, ..., i + lookback - 1]
    of the joined (features and target) frame after NaN rows have been
    dropped. The label and the date both come from row i + lookback - 1,
    which is the decision time at which the model would make a
    prediction in live trading.

    Args:
        features: One row per date, already scaled.
        target: Aligned label series with the same DatetimeIndex.
        lookback: Number of trailing rows that make up one input window.

    Returns:
        A tuple (X, y, dates) with shapes
          X     : (n_samples, lookback, n_features) float32
          y     : (n_samples,) float32
          dates : DatetimeIndex of length n_samples; the decision time
                  for each sample.
    """
    if not features.index.equals(target.index):
        raise ValueError("features and target must share the same index")
    if lookback < 1:
        raise ValueError("lookback must be at least 1")

    # Drop rows where either side is NaN. After this, both sides are
    # dense and the window builder does not need to skip gaps.
    joined = features.copy()
    joined["__target__"] = target
    joined = joined.dropna()

    n = len(joined)
    if n < lookback:
        raise ValueError(
            f"Not enough rows after dropping NaN ({n}) for lookback={lookback}"
        )

    feat_cols = [c for c in joined.columns if c != "__target__"]
    feat = joined[feat_cols].to_numpy(dtype=np.float32)
    targ = joined["__target__"].to_numpy(dtype=np.float32)

    # sliding_window_view returns shape (n_samples, n_features, lookback).
    # Transpose the last two axes to land at (n_samples, lookback, n_features).
    windows = sliding_window_view(feat, window_shape=lookback, axis=0)
    X = np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)

    y = targ[lookback - 1 :]
    sample_dates = joined.index[lookback - 1 :]

    return X, y, sample_dates
