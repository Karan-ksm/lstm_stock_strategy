"""Read cached OHLCV data from disk.

Every downstream module imports from this file to obtain price data.
Loader assumes the downloader has already populated the cache and
validates that the cached frame meets pipeline assumptions before
handing it back, so callers do not have to defensively check shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import resolve

_LOGGER = get_logger(__name__)

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str, cache_dir: str | Path) -> Path:
    """Return the Parquet path for a given ticker inside cache_dir."""
    return resolve(cache_dir) / f"{ticker.upper()}.parquet"


def _validate(frame: pd.DataFrame, ticker: str) -> None:
    """Raise ValueError if the cached frame violates pipeline assumptions."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: cache index is not a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError(f"{ticker}: cache has duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: cache index is not sorted ascending")
    missing = [c for c in _OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{ticker}: cache is missing columns {missing}")
    if frame[_OHLCV_COLUMNS].isna().any().any():
        raise ValueError(f"{ticker}: cache contains NaN OHLCV values")


def load(ticker: str, cache_dir: str | Path) -> pd.DataFrame:
    """Load a single cached ticker.

    Args:
        ticker: Symbol to load (case insensitive; stored uppercase).
        cache_dir: Directory containing the Parquet cache.

    Returns:
        DataFrame indexed by date with open, high, low, close, volume.

    Raises:
        FileNotFoundError: If the cache file does not exist.
        ValueError: If validation fails on the cached data.
    """
    path = _cache_path(ticker, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for {ticker} at {path}. " "Run the downloader first."
        )
    frame = pd.read_parquet(path)
    _validate(frame, ticker.upper())
    _LOGGER.debug("Loaded %d rows for %s from cache", len(frame), ticker)
    return frame


def load_many(tickers: list[str], cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load a batch of tickers from cache. See load() for details."""
    return {t.upper(): load(t, cache_dir) for t in tickers}
