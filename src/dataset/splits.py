"""Chronological train, validation, and test splits.

The only acceptable kind of split for this project. No shuffling, no
random subsampling, no peeking across boundaries. Train is the earliest
slice of the data, validation comes next, test is the most recent.

If you ever find yourself wanting to shuffle here, the answer is no.
The whole point of an out of sample backtest is that the test slice
represents data the model has never seen and never could have seen.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into chronological train / val / test slices.

    Args:
        frame: DataFrame indexed by a sorted, ascending DatetimeIndex.
        train_fraction: Share of rows that go to train (earliest).
        val_fraction: Share of rows that go to validation.
        test_fraction: Share of rows that go to test (latest).

    Returns:
        A tuple (train_df, val_df, test_df). The slices are disjoint,
        their union equals the input frame, and every date in train is
        strictly earlier than every date in val, and likewise for val
        and test.

    Raises:
        ValueError: If the index is not a sorted ascending DatetimeIndex,
            or if the three fractions do not sum to 1.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("chronological_split requires a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("chronological_split requires a sorted ascending index")

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    n = len(frame)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    # Test gets whatever is left so floor rounding never loses a row.
    train = frame.iloc[:n_train]
    val = frame.iloc[n_train : n_train + n_val]
    test = frame.iloc[n_train + n_val :]
    return train, val, test
