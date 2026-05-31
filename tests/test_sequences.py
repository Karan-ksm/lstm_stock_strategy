"""Assert sequence builder has correct shapes and no future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset.sequences import build_sequences, make_directional_target


def _rising_close(n: int, start: float = 100.0, end: float = 200.0) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(start, end, n), index=idx)


def test_target_tail_has_nan_equal_to_horizon() -> None:
    close = _rising_close(100)
    label = make_directional_target(close, horizon=3, up_threshold=0.0)
    assert label.iloc[-3:].isna().all()
    assert label.iloc[:-3].notna().all()


def test_target_is_one_on_a_monotonically_rising_series() -> None:
    close = _rising_close(50)
    label = make_directional_target(close, horizon=1, up_threshold=0.0)
    assert (label.dropna() == 1.0).all()


def test_target_is_zero_on_a_monotonically_falling_series() -> None:
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(200.0, 100.0, n), index=idx)
    label = make_directional_target(close, horizon=1, up_threshold=0.0)
    assert (label.dropna() == 0.0).all()


def test_build_sequences_has_expected_shape() -> None:
    n_rows = 100
    n_features = 4
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    features = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n_rows, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    target = pd.Series(np.zeros(n_rows), index=idx)

    X, y, dates = build_sequences(features, target, lookback=10)
    assert X.shape == (n_rows - 10 + 1, 10, n_features)
    assert y.shape == (n_rows - 10 + 1,)
    assert len(dates) == n_rows - 10 + 1


def test_each_window_ends_on_its_decision_date() -> None:
    """The last row of X[i] must come from features.loc[dates[i]]."""
    n_rows = 50
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    # Encode each row's position so violations are easy to spot.
    positions = np.arange(n_rows, dtype=np.float32).reshape(-1, 1)
    features = pd.DataFrame(
        np.tile(positions, (1, 2)),
        index=idx,
        columns=["f0", "f1"],
    )
    target = pd.Series(np.zeros(n_rows), index=idx)

    lookback = 5
    X, _, dates = build_sequences(features, target, lookback=lookback)

    for i in range(len(dates)):
        last_row_value = X[i, -1, 0]
        # The position of dates[i] in the original index.
        expected_position = float(idx.get_loc(dates[i]))
        assert last_row_value == expected_position


def test_window_contents_match_consecutive_feature_rows() -> None:
    """Within a single window, rows must be consecutive in the input."""
    n_rows = 30
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    positions = np.arange(n_rows, dtype=np.float32).reshape(-1, 1)
    features = pd.DataFrame(positions, index=idx, columns=["f0"])
    target = pd.Series(np.zeros(n_rows), index=idx)

    X, _, _ = build_sequences(features, target, lookback=4)
    # X[0] must be [0, 1, 2, 3], X[1] must be [1, 2, 3, 4], etc.
    for i in range(X.shape[0]):
        expected = np.arange(i, i + 4, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(X[i], expected)


def test_rejects_misaligned_index() -> None:
    idx1 = pd.date_range("2020-01-01", periods=50, freq="B")
    idx2 = pd.date_range("2020-02-01", periods=50, freq="B")
    features = pd.DataFrame({"f0": range(50)}, index=idx1)
    target = pd.Series(range(50), index=idx2)
    with pytest.raises(ValueError):
        build_sequences(features, target, lookback=5)


def test_rejects_too_few_rows_for_lookback() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    features = pd.DataFrame({"f0": [1.0, 2.0, 3.0]}, index=idx)
    target = pd.Series([0.0, 1.0, 0.0], index=idx)
    with pytest.raises(ValueError):
        build_sequences(features, target, lookback=10)
