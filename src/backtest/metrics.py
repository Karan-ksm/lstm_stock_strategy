"""Performance metrics on a backtest result.

Every metric is a pure function of (equity_curve, daily_returns,
initial_cash, n_trades). No side effects, no plotting, no I/O. The
report module pulls these in and formats them; this file just does
the math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per year. Used for annualization.
_TRADING_DAYS_PER_YEAR = 252


def total_return(equity: pd.Series, initial_cash: float) -> float:
    """Total return from initial cash to the last equity value, as a fraction."""
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / initial_cash - 1.0)


def annualized_return(equity: pd.Series, initial_cash: float) -> float:
    """Compound annual growth rate, assuming 252 trading days per year."""
    if equity.empty:
        return 0.0
    n_days = len(equity)
    if n_days < 2:
        return 0.0
    total = total_return(equity, initial_cash)
    years = n_days / _TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Annualized Sharpe with a zero risk free rate assumption.

    sharpe = mean(r) / std(r) * sqrt(252)

    Returns 0.0 when there is no variance to divide by.
    """
    r = daily_returns.dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(r.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak to trough decline as a (negative) fraction."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def win_rate(daily_returns: pd.Series) -> float:
    """Fraction of strictly positive return days, ignoring zero days.

    Zero days are excluded because in this strategy a flat day produces
    exactly zero, and including them would dilute the win rate of the
    days when the strategy actually had an open position.
    """
    non_zero = daily_returns[daily_returns != 0.0].dropna()
    if non_zero.empty:
        return 0.0
    return float((non_zero > 0).mean())


def summarize(
    equity: pd.Series,
    daily_returns: pd.Series,
    initial_cash: float,
    n_trades: int,
) -> dict[str, float]:
    """Return a dict of all metrics in one call."""
    return {
        "total_return": total_return(equity, initial_cash),
        "annualized_return": annualized_return(equity, initial_cash),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(daily_returns),
        "n_trades": float(n_trades),
        "n_days": float(len(equity)),
    }
