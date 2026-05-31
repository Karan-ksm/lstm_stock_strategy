"""Feature scaling with the 'fit only on train' invariant baked in.

Wraps a scikit learn StandardScaler. The wrapper refuses to be fit
twice, so there is no path by which validation or test statistics
can ever influence the scaling parameters used by the model. The
practical effect is that any attempt to leak future information
through the scaler raises an exception instead of silently inflating
backtest metrics.

Usage:
    scaler = FeatureScaler()
    scaler.fit(train_features)            # one time only
    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

The whole scaler is pickled to disk alongside the trained model so
paper trading applies identical normalization on live bars.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Train fit only standard scaler with strict refit protection."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def fitted(self) -> bool:
        """True once fit has been called successfully."""
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        """Column order seen at fit time. Empty until fitted."""
        return list(self._feature_names)

    def fit(self, train_features: pd.DataFrame) -> "FeatureScaler":
        """Fit on training data. Raises if called more than once.

        Args:
            train_features: DataFrame whose rows are training samples
                and whose columns are features. The column order seen
                here defines the order applied at transform time.
        """
        if self._fitted:
            raise RuntimeError(
                "FeatureScaler is already fitted. Refitting would leak "
                "validation or test statistics into the scaling step. "
                "Construct a new FeatureScaler instead."
            )
        self._scaler.fit(train_features.to_numpy())
        self._feature_names = list(train_features.columns)
        self._fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the train time mean and standard deviation to `features`.

        Reorders columns to match the order seen at fit time so the
        caller cannot accidentally swap columns between splits.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler must be fitted on training data before "
                "transform can be called."
            )
        missing = set(self._feature_names) - set(features.columns)
        if missing:
            raise ValueError(
                f"Features frame is missing columns seen at fit time: {missing}"
            )
        ordered = features[self._feature_names]
        scaled = self._scaler.transform(ordered.to_numpy())
        return pd.DataFrame(
            scaled, index=features.index, columns=self._feature_names
        )

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on train and return the transformed train frame."""
        self.fit(train_features)
        return self.transform(train_features)

    def save(self, path: str | Path) -> None:
        """Pickle the entire scaler to disk, creating parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        """Load a previously saved scaler from disk."""
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object at {path} is not a FeatureScaler"
            )
        return obj
