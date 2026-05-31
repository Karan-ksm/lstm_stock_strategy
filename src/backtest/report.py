"""Backtest report: console table, plots, and saved artifacts.

Reads a BacktestResult and a metrics dict, writes plots and CSVs into
the reports directory, and prints a side by side comparison table to
stdout. Matplotlib is imported eagerly here because every backtest
run produces plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use the non interactive Agg backend so saving figures works in
# headless environments and we never pop a window during a script run.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import pandas as pd  # noqa: E402

from src.backtest.engine import BacktestResult  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

_LOGGER = get_logger(__name__)


def _fmt(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "int":
        return f"{int(value):d}"
    return f"{value:.4f}"


def print_summary(
    strategy_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    ticker: str,
) -> None:
    """Print a strategy vs buy and hold table to stdout."""
    rows: list[tuple[str, str]] = [
        ("total_return", "pct"),
        ("annualized_return", "pct"),
        ("sharpe_ratio", "ratio"),
        ("max_drawdown", "pct"),
        ("win_rate", "pct"),
        ("n_trades", "int"),
        ("n_days", "int"),
    ]
    header = f"{'Metric':<20} {'Strategy':>15} {'Baseline':>15}"
    print()
    print(f"=== {ticker} ===")
    print(header)
    print("-" * len(header))
    for key, kind in rows:
        s = _fmt(strategy_metrics.get(key, 0.0), kind)
        b = _fmt(baseline_metrics.get(key, 0.0), kind)
        print(f"{key:<20} {s:>15} {b:>15}")
    print()


def plot_equity_curve(
    result: BacktestResult, ticker: str, out_dir: Path
) -> Path:
    """Save a strategy vs buy and hold equity curve PNG. Return its path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        result.equity_curve.index,
        result.equity_curve.to_numpy(),
        label="Strategy",
        linewidth=1.5,
    )
    ax.plot(
        result.baseline_curve.index,
        result.baseline_curve.to_numpy(),
        label="Buy and hold",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(f"{ticker}: strategy vs buy and hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{ticker}_equity_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown(
    result: BacktestResult, ticker: str, out_dir: Path
) -> Path:
    """Save an underwater drawdown chart PNG. Return its path."""
    eq = result.equity_curve
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color="red", alpha=0.4
    )
    ax.set_title(f"{ticker}: strategy drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{ticker}_drawdown.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    ticker: str,
    result: BacktestResult,
    strategy_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    reports_dir: str | Path,
) -> Path:
    """Save plots, a metrics CSV, and a trades CSV under reports_dir/<TICKER>/.

    Returns:
        The absolute path of the per ticker report directory.
    """
    out_dir = ensure_dir(Path(reports_dir) / ticker.upper())
    plot_equity_curve(result, ticker, out_dir)
    plot_drawdown(result, ticker, out_dir)

    metrics_df = pd.DataFrame(
        {"strategy": pd.Series(strategy_metrics), "baseline": pd.Series(baseline_metrics)}
    )
    metrics_df.to_csv(out_dir / "metrics.csv")

    if not result.trades.empty:
        result.trades.to_csv(out_dir / "trades.csv", index=False)

    _LOGGER.info("Wrote report for %s to %s", ticker, out_dir)
    return out_dir
