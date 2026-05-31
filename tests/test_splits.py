"""Assert chronological_split is leak proof and sized correctly."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dataset.splits import chronological_split


def _make_frame(n_rows: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return pd.DataFrame({"x": range(n_rows)}, index=idx)


def test_sizes_round_to_floor_and_test_gets_the_remainder() -> None:
    frame = _make_frame(1000)
    train, val, test = chronological_split(frame, 0.7, 0.15, 0.15)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    assert len(train) + len(val) + len(test) == len(frame)


def test_no_overlap_between_any_two_splits() -> None:
    frame = _make_frame(1000)
    train, val, test = chronological_split(frame, 0.7, 0.15, 0.15)
    assert train.index.intersection(val.index).empty
    assert train.index.intersection(test.index).empty
    assert val.index.intersection(test.index).empty


def test_train_is_strictly_before_val_is_strictly_before_test() -> None:
    frame = _make_frame(1000)
    train, val, test = chronological_split(frame, 0.7, 0.15, 0.15)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_rejects_fractions_that_do_not_sum_to_one() -> None:
    frame = _make_frame(100)
    with pytest.raises(ValueError):
        chronological_split(frame, 0.5, 0.5, 0.5)


def test_rejects_descending_index() -> None:
    frame = _make_frame(100).sort_index(ascending=False)
    with pytest.raises(ValueError):
        chronological_split(frame, 0.7, 0.15, 0.15)


def test_rejects_non_datetime_index() -> None:
    frame = pd.DataFrame({"x": range(100)})
    with pytest.raises(ValueError):
        chronological_split(frame, 0.7, 0.15, 0.15)
