"""Technical indicator functions.

Each indicator is a pure function: it takes the OHLCV DataFrame and
returns a single Series aligned to the same DatetimeIndex. No side
effects, no global state, deterministic given the input.

All functions share the signature `(df, window=None)` so the pipeline
can dispatch uniformly across indicators that need a window and those
that do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_return(df: pd.DataFrame, **_: object) -> pd.Series:
    """One day percent change of close. Extra keywords are ignored."""
    return df["close"].pct_change(fill_method=None)


def log_return(df: pd.DataFrame, **_: object) -> pd.Series:
    """One day log return of close. Extra keywords are ignored."""
    return np.log(df["close"] / df["close"].shift(1))


def sma(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Simple moving average of close over the given window."""
    if window is None or window < 1:
        raise ValueError("sma requires a positive window, for example sma_10")
    return df["close"].rolling(window).mean()


def rsi(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Wilder's Relative Strength Index over the given window.

    Returns values in the closed interval [0, 100]. Yields NaN until
    enough history accumulates to populate the smoothing average.
    """
    if window is None or window < 1:
        raise ValueError("rsi requires a positive window, for example rsi_14")
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing is an exponential moving average with alpha = 1/n.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    # When avg_loss is 0, rs becomes inf and the formula collapses to 100,
    # which is the correct Wilder behavior for a pure uptrend window.
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def volatility(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Rolling standard deviation of one day log returns.

    Not annualized; left as a raw rolling standard deviation so the
    model sees the same units in training and in live use.
    """
    if window is None or window < 1:
        raise ValueError(
            "volatility requires a positive window, for example volatility_20"
        )
    ret = np.log(df["close"] / df["close"].shift(1))
    return ret.rolling(window).std()


def volume_change(df: pd.DataFrame, **_: object) -> pd.Series:
    """One day percent change of volume. Extra keywords are ignored."""
    return df["volume"].pct_change(fill_method=None)
