"""Assert FeatureScaler obeys the fit only on train invariant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset.scaler import FeatureScaler


def _frame(n_rows: int = 100, n_cols: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), index=idx, columns=cols)


def test_starts_unfitted() -> None:
    assert FeatureScaler().fitted is False


def test_refuses_transform_before_fit() -> None:
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(_frame())


def test_refuses_to_refit() -> None:
    scaler = FeatureScaler()
    scaler.fit(_frame(seed=0))
    # A second fit on val or test data would leak statistics. Forbid it.
    with pytest.raises(RuntimeError):
        scaler.fit(_frame(seed=1))


def test_transform_uses_train_statistics_only() -> None:
    train = _frame(n_rows=200, seed=1)
    val = _frame(n_rows=50, seed=2)

    scaler = FeatureScaler().fit(train)

    # scikit learn StandardScaler uses biased std (ddof=0).
    expected_mean = train.mean()
    expected_std = train.std(ddof=0)

    scaled_val = scaler.transform(val)
    expected_val = (val[expected_mean.index] - expected_mean) / expected_std

    np.testing.assert_allclose(
        scaled_val.to_numpy(),
        expected_val.to_numpy(),
        rtol=1e-6,
    )


def test_transform_reorders_columns_to_fit_time_order() -> None:
    train = _frame(seed=3)
    scaler = FeatureScaler().fit(train)

    # Reverse the column order before calling transform.
    reordered = train[list(reversed(train.columns))]
    out = scaler.transform(reordered)

    # Output column order must match the fit time order, regardless of
    # what we passed in.
    assert list(out.columns) == list(train.columns)


def test_transform_rejects_missing_columns() -> None:
    train = _frame(seed=4)
    scaler = FeatureScaler().fit(train)
    with pytest.raises(ValueError):
        scaler.transform(train.drop(columns=["f0"]))


def test_save_and_load_round_trip(tmp_path) -> None:
    train = _frame(seed=5)
    scaler = FeatureScaler().fit(train)

    target_path = tmp_path / "scaler.pkl"
    scaler.save(target_path)

    loaded = FeatureScaler.load(target_path)
    assert loaded.fitted is True
    np.testing.assert_allclose(
        loaded.transform(train).to_numpy(),
        scaler.transform(train).to_numpy(),
    )
