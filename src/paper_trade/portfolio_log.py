"""CSV based decision log for paper trading.

One row per daily run. Append only. The same file feeds both the
audit log (what did we decide, why, what did we trade) and the
portfolio value time series used for the equity chart.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_dir

_LOG_FILENAME = "paper_trade_log.csv"


@dataclass(frozen=True)
class LogRow:
    """One row in the paper trade decision log."""

    timestamp_utc: str
    ticker: str
    decision_date: str
    probability_up: float
    current_position: int
    target_position: int
    action: str
    qty_traded: float
    last_price: float
    order_id: str
    cash: float
    portfolio_value: float
    equity: float


def _log_path(logs_dir: str | Path) -> Path:
    """Return the canonical CSV path inside `logs_dir`, creating dirs."""
    return ensure_dir(logs_dir) / _LOG_FILENAME


def append_row(row: LogRow, logs_dir: str | Path) -> Path:
    """Append a row to the log. Writes the header the first time."""
    path = _log_path(logs_dir)
    is_new = not path.exists()
    header = [f.name for f in fields(LogRow)]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(row))
    return path


def read_log(logs_dir: str | Path) -> pd.DataFrame:
    """Load the paper trade log as a DataFrame for inspection or plotting."""
    path = _log_path(logs_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp_utc", "decision_date"])
