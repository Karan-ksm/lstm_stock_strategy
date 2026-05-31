"""Spot check indicator math and the feature pipeline dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import indicators
from src.features.pipeline import _parse, build_features


def _ohlcv(n: int = 100, start: float = 100.0) -> pd.DataFrame:
    """Build a synthetic, monotonically rising OHLCV frame for tests."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(start, start * 1.5, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_log_return_first_value_is_nan() -> None:
    df = _ohlcv()
    result = indicators.log_return(df)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].notna().all()


def test_sma_warmup_count_matches_window() -> None:
    df = _ohlcv()
    window = 10
    result = indicators.sma(df, window=window)
    assert result.iloc[: window - 1].isna().all()
    assert result.iloc[window - 1 :].notna().all()


def test_rsi_stays_in_zero_to_hundred() -> None:
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {"close": close, "volume": np.ones(n)}, index=idx
    )
    result = indicators.rsi(df, window=14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_parser_extracts_window_or_returns_none() -> None:
    assert _parse("sma_10") == ("sma", 10)
    assert _parse("rsi_14") == ("rsi", 14)
    assert _parse("volatility_20") == ("volatility", 20)
    assert _parse("log_return") == ("log_return", None)
    assert _parse("volume_change") == ("volume_change", None)


def test_build_features_drops_warmup_and_preserves_column_order() -> None:
    df = _ohlcv(n=200)
    enabled = ["log_return", "sma_10", "rsi_14"]
    features = build_features(df, enabled, drop_warmup=True)
    assert list(features.columns) == enabled
    assert features.isna().sum().sum() == 0
    # 14 is the longest required warm up.
    assert len(features) >= len(df) - 14
