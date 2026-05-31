"""Historical OHLCV downloader.

Pulls daily bars from yfinance and writes one Parquet file per ticker
into the configured cache directory. Reuses the cache on subsequent
runs unless the requested date range falls outside what is cached or
the caller passes force_refresh=True.

With auto_adjust=True (the default in config.yaml), prices are split
and dividend adjusted, so we do not need to track corporate actions
separately downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, resolve

_LOGGER = get_logger(__name__)

# Canonical lowercase column schema used everywhere downstream.
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Tolerance for weekend and holiday gaps when judging cache freshness.
_BOUNDARY_TOLERANCE = pd.Timedelta(days=5)


def _cache_path(ticker: str, cache_dir: str | Path) -> Path:
    """Return the Parquet path for a given ticker inside cache_dir."""
    return resolve(cache_dir) / f"{ticker.upper()}.parquet"


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to a single level.

    yfinance sometimes returns a MultiIndex (price field, ticker) even
    for a single ticker request. We keep only the price field level.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame.

    Steps:
      1. Flatten MultiIndex columns if present.
      2. Lowercase column names.
      3. Keep only the OHLCV columns we need.
      4. Force a sorted, deduplicated DatetimeIndex with no timezone.
      5. Drop any rows with NaN values and log how many were dropped.
    """
    df = _flatten_columns(frame).copy()
    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in _OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")

    df = df[_OHLCV_COLUMNS]

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    nan_rows = int(df.isna().any(axis=1).sum())
    if nan_rows:
        _LOGGER.warning("Dropping %d rows containing NaN OHLCV values", nan_rows)
        df = df.dropna()

    return df


def _is_cache_fresh(cached: pd.DataFrame, start: str, end: str) -> bool:
    """Return True when the cache fully covers the requested range.

    A small tolerance is allowed at each boundary because markets are
    closed on weekends and holidays, so the actual first or last
    trading day inside the request may be a few days inside it.

    If `end` is in the future, we clamp the comparison to today since
    no data source can have tomorrow's bars yet.
    """
    if cached.empty:
        return False
    requested_start = pd.Timestamp(start)
    today = pd.Timestamp.today().normalize()
    requested_end = min(pd.Timestamp(end), today)
    have_start = cached.index.min()
    have_end = cached.index.max()
    return (
        have_start <= requested_start + _BOUNDARY_TOLERANCE
        and have_end >= requested_end - _BOUNDARY_TOLERANCE
    )


def download_one(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    auto_adjust: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download or load a single ticker's daily OHLCV history.

    Args:
        ticker: Ticker symbol, for example "AAPL".
        start: ISO date string for the inclusive start.
        end: ISO date string for the inclusive end.
        cache_dir: Where to read or write the Parquet cache.
        auto_adjust: When True, prices are split and dividend adjusted.
        force_refresh: If True, always fetch fresh data and overwrite
            the cache.

    Returns:
        DataFrame indexed by date with columns open, high, low, close,
        and volume, sliced to the requested range.
    """
    ensure_dir(cache_dir)
    path = _cache_path(ticker, cache_dir)

    if path.exists() and not force_refresh:
        cached = pd.read_parquet(path)
        if _is_cache_fresh(cached, start, end):
            _LOGGER.info("Cache hit for %s at %s", ticker, path)
            return cached.loc[start:end]
        _LOGGER.info("Cache for %s does not cover request; fetching fresh", ticker)

    _LOGGER.info("Fetching %s from yfinance: %s to %s", ticker, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        actions=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker} between {start} and {end}"
        )

    frame = _normalize(raw)
    frame.to_parquet(path)
    _LOGGER.info("Wrote %d rows for %s to %s", len(frame), ticker, path)
    return frame.loc[start:end]


def download_many(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    auto_adjust: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers.

    Returns:
        A dict keyed by uppercased ticker symbol; each value is the
        same DataFrame shape returned by download_one.
    """
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        out[t.upper()] = download_one(
            t,
            start,
            end,
            cache_dir,
            auto_adjust=auto_adjust,
            force_refresh=force_refresh,
        )
    return out
