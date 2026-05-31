"""Feature pipeline.

Resolves the list of enabled feature names from config into concrete
indicator function calls and assembles a DataFrame of features.

Naming convention:
  * Bare name for indicators with no parameter: `log_return`,
    `simple_return`, `volume_change`.
  * `name_window` for indicators that take a window length:
    `sma_10`, `rsi_14`, `volatility_20`. The integer suffix is the
    rolling window in days.

The same code path runs at training time and during live paper
trading, so features computed on a live bar match features computed
offline byte for byte.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd

from src.features import indicators
from src.utils.logging import get_logger

_LOGGER = get_logger(__name__)

# Maps the bare indicator name to its implementation function.
_REGISTRY: dict[str, Callable[..., pd.Series]] = {
    "simple_return": indicators.simple_return,
    "log_return": indicators.log_return,
    "sma": indicators.sma,
    "rsi": indicators.rsi,
    "volatility": indicators.volatility,
    "volume_change": indicators.volume_change,
}

# Matches names like "sma_10": a lowercase prefix, an underscore, then digits.
_WINDOW_PATTERN = re.compile(r"^(?P<name>[a-z][a-z_]*?)_(?P<window>\d+)$")


def _parse(feature_name: str) -> tuple[str, int | None]:
    """Split a feature name into (base_name, window).

    The window suffix only counts when the prefix is itself a known
    indicator, so a bare name like `volume_change` is not mistaken
    for `volume` with a `change` window.

    Examples:
        sma_10        -> ("sma", 10)
        rsi_14        -> ("rsi", 14)
        volatility_20 -> ("volatility", 20)
        log_return    -> ("log_return", None)
        volume_change -> ("volume_change", None)
    """
    match = _WINDOW_PATTERN.match(feature_name)
    if match and match.group("name") in _REGISTRY:
        return match.group("name"), int(match.group("window"))
    return feature_name, None


def compute_one(df: pd.DataFrame, feature_name: str) -> pd.Series:
    """Compute a single feature column by name and return it as a Series."""
    base, window = _parse(feature_name)
    if base not in _REGISTRY:
        raise KeyError(
            f"Unknown feature '{feature_name}'. "
            f"Known indicators: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[base](df, window=window).rename(feature_name)


def build_features(
    df: pd.DataFrame,
    enabled: list[str],
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Assemble a feature DataFrame from an OHLCV frame.

    Args:
        df: Validated OHLCV frame; see src.data.loader for what counts
            as validated.
        enabled: Ordered list of feature names from config. The output
            columns appear in this exact order.
        drop_warmup: When True (default), drop leading rows where any
            feature is still NaN due to a rolling window warm up.

    Returns:
        DataFrame indexed by date with one column per name in `enabled`.
    """
    if not enabled:
        raise ValueError("No features enabled in config")

    series_list = [compute_one(df, name) for name in enabled]
    features = pd.concat(series_list, axis=1)

    if drop_warmup:
        n_before = len(features)
        features = features.dropna()
        n_dropped = n_before - len(features)
        if n_dropped:
            _LOGGER.debug(
                "Dropped %d warm up rows that still contained NaN features",
                n_dropped,
            )

    return features
