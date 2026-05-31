"""Paper trading chart: portfolio value over time.

Reads the CSV produced by portfolio_log.append_row and saves a PNG
into the same logs directory. Designed to be safe to call whether
the log is empty, has one row, or has thousands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

# Non interactive backend so plots save cleanly inside cron / launchd.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.paper_trade.portfolio_log import read_log  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

_LOGGER = get_logger(__name__)


def plot_portfolio_value(logs_dir: str | Path) -> Optional[Path]:
    """Save the portfolio value curve as a PNG. Return its path, or None.

    Returns None when the log is empty so the caller can decide what
    to do (typically just skip and move on).
    """
    df = read_log(logs_dir)
    if df.empty:
        _LOGGER.info("Paper trade log is empty; no chart produced")
        return None

    df = df.sort_values("timestamp_utc")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df["timestamp_utc"],
        df["portfolio_value"],
        label="Paper portfolio value",
        linewidth=1.5,
    )
    ax.set_title("Paper trading: portfolio value over time")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = ensure_dir(logs_dir)
    path = out_dir / "paper_portfolio_value.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
