"""Day by day backtest simulation.

Two pieces:
  * `run_backtest`: a pure function that takes a decision DataFrame and
    cost parameters, walks it bar by bar, and returns equity, returns,
    and a trade log.
  * `backtest_from_bundle`: the orchestrator that pulls a saved model
    bundle, replays features on the test split with the same code path
    as training, generates predictions, and feeds them into run_backtest.

Timing convention (no lookahead):
  At each decision date t, we observe:
    * close[t]: the last known closing price.
    * prediction[t]: the model output, computed from features up to
      and including close[t].
  We decide a position for the next bar. If the position changes, a
  cost of (commission_bps + slippage_bps) is charged at close[t]. The
  position is then held from close[t] to close[t+1], realizing the
  underlying's return for that bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.signals import classify_action, desired_position
from src.config import Config
from src.data.downloader import download_one
from src.dataset.sequences import build_sequences, make_directional_target
from src.dataset.splits import chronological_split
from src.features.pipeline import build_features
from src.model.persistence import load_bundle
from src.utils.logging import get_logger

_LOGGER = get_logger(__name__)

_TARGET_COL = "__target__"


@dataclass(frozen=True)
class BacktestResult:
    """Container for everything produced by a single backtest run."""

    equity_curve: pd.Series
    baseline_curve: pd.Series
    daily_returns: pd.Series
    baseline_returns: pd.Series
    trades: pd.DataFrame
    initial_cash: float


def assemble_decisions(
    close: pd.Series, predictions: pd.Series
) -> pd.DataFrame:
    """Build a (decision_date, close_t, close_t_plus_1, prediction) frame.

    Each row is one decision step. Rows where either the next bar's
    close or the prediction is missing are dropped, which keeps the
    walk pure: every row that survives has the data needed to realize
    a return.
    """
    frame = pd.DataFrame(
        {
            "close_t": close,
            "close_t_plus_1": close.shift(-1),
            "prediction": predictions,
        }
    )
    return frame.dropna()


def run_backtest(
    decisions: pd.DataFrame,
    initial_cash: float,
    buy_threshold: float,
    sell_threshold: float,
    commission_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    """Walk decision dates and simulate trading.

    Args:
        decisions: Frame from `assemble_decisions`.
        initial_cash: Starting portfolio value.
        buy_threshold: Probability required to enter a long position.
        sell_threshold: Probability below which we exit to flat.
        commission_bps: Per trade commission in basis points.
        slippage_bps: Per trade modeled slippage in basis points.

    Returns:
        A BacktestResult whose equity_curve and baseline_curve both
        start with `initial_cash` and have one row per decision date.
    """
    if decisions.empty:
        raise ValueError("No decisions to backtest")
    required = {"close_t", "close_t_plus_1", "prediction"}
    if not required.issubset(decisions.columns):
        raise ValueError(f"decisions frame missing columns: {required}")

    cost_per_change = (commission_bps + slippage_bps) / 10_000.0

    # Baseline holds whatever fractional shares initial_cash buys at
    # the first close_t, then marks to market on each subsequent bar.
    baseline_shares = initial_cash / float(decisions["close_t"].iloc[0])

    position = 0
    equity = initial_cash

    records: list[dict[str, float]] = []
    trades: list[dict[str, object]] = []

    for date, row in decisions.iterrows():
        prediction = float(row["prediction"])
        target = desired_position(
            prediction, position, buy_threshold, sell_threshold
        )
        changed = target != position

        ret = float(row["close_t_plus_1"] / row["close_t"] - 1.0)
        fee = cost_per_change if changed else 0.0

        # New equity: pay any fee at close_t, then hold target through close_t_plus_1.
        new_equity = equity * (1.0 - fee) * (1.0 + target * ret)
        daily_ret = new_equity / equity - 1.0

        if changed:
            trades.append(
                {
                    "date": date,
                    "action": classify_action(position, target).value,
                    "price": float(row["close_t"]),
                    "prediction": prediction,
                    "equity_before": equity,
                    "equity_after": new_equity,
                }
            )

        baseline_equity = baseline_shares * float(row["close_t_plus_1"])

        records.append(
            {
                "date": date,
                "position": float(target),
                "equity": new_equity,
                "daily_return": daily_ret,
                "baseline_equity": baseline_equity,
                "baseline_return": ret,
            }
        )

        equity = new_equity
        position = target

    log_df = pd.DataFrame(records).set_index("date")

    # Prepend the initial state so curves start at initial_cash and
    # drawdown picks up any first day decline.
    pre_index = decisions.index[0] - pd.tseries.offsets.BDay(1)
    initial_row = pd.DataFrame(
        {
            "position": [0.0],
            "equity": [initial_cash],
            "daily_return": [0.0],
            "baseline_equity": [initial_cash],
            "baseline_return": [0.0],
        },
        index=[pre_index],
    )
    log_df = pd.concat([initial_row, log_df])

    return BacktestResult(
        equity_curve=log_df["equity"],
        baseline_curve=log_df["baseline_equity"],
        daily_returns=log_df["daily_return"],
        baseline_returns=log_df["baseline_return"],
        trades=pd.DataFrame(trades),
        initial_cash=initial_cash,
    )


def backtest_from_bundle(
    ticker: str, config: Config
) -> tuple[BacktestResult, pd.Series]:
    """End to end backtest for one ticker using a saved model bundle.

    Reuses the exact training code paths to rebuild features, scale
    them, and form sequences. This is what guarantees the inputs the
    model evaluates here match what it saw during training.

    Args:
        ticker: Ticker symbol whose bundle should be loaded.
        config: Loaded config. Split fractions must match the values
            in use at training time; otherwise the test slice will
            cover different dates.

    Returns:
        (result, predictions). predictions is a Series of P(up) indexed
        by decision date for inspection or further analysis.
    """
    _LOGGER.info("=== Backtest for %s ===", ticker)

    bundle_dir = Path(config.training.models_dir) / ticker.upper()
    model, scaler, spec = load_bundle(bundle_dir)
    _LOGGER.info("Loaded bundle from %s (trained_on=%s)", bundle_dir, spec.trained_on)

    df = download_one(
        ticker=ticker,
        start=config.data.start_date,
        end=config.data.end_date,
        cache_dir=config.data.cache_dir,
        auto_adjust=config.data.auto_adjust,
    )

    # Use spec.features so the order and identity match what the model saw.
    features = build_features(df, spec.features, drop_warmup=True)

    target = make_directional_target(
        df["close"],
        horizon=spec.horizon_days,
        up_threshold=spec.up_threshold,
    ).loc[features.index]

    combined = features.copy()
    combined[_TARGET_COL] = target
    _train_df, _val_df, test_df = chronological_split(
        combined,
        train_fraction=config.dataset.train_fraction,
        val_fraction=config.dataset.val_fraction,
        test_fraction=config.dataset.test_fraction,
    )
    _LOGGER.info("Test slice: %d rows", len(test_df))

    test_features = test_df[spec.features]
    test_target = test_df[_TARGET_COL]

    test_scaled = scaler.transform(test_features)
    X_test, _y_test, decision_dates = build_sequences(
        test_scaled, test_target, lookback=spec.lookback
    )

    # Direct forward pass instead of model.predict(): predict() can
    # deadlock on its first call in a freshly loaded process on macOS.
    # Our test batches are small, so one eager call is fast and reliable.
    probabilities = np.asarray(model(X_test, training=False)).flatten()
    predictions = pd.Series(probabilities, index=decision_dates, name="prob_up")
    _LOGGER.info(
        "Generated %d predictions: mean=%.3f std=%.3f",
        len(predictions), float(predictions.mean()), float(predictions.std()),
    )

    # Close prices over the test span. assemble_decisions will trim to
    # rows where both close_t_plus_1 and prediction are non NaN.
    close_test = df["close"].loc[decision_dates[0]:]
    decisions = assemble_decisions(close_test, predictions)

    result = run_backtest(
        decisions=decisions,
        initial_cash=config.backtest.initial_cash,
        buy_threshold=config.backtest.signal.buy_threshold,
        sell_threshold=config.backtest.signal.sell_threshold,
        commission_bps=config.backtest.costs.commission_bps,
        slippage_bps=config.backtest.costs.slippage_bps,
    )

    _LOGGER.info(
        "Backtest finished: %d trades, final equity %.2f, baseline %.2f",
        len(result.trades),
        float(result.equity_curve.iloc[-1]),
        float(result.baseline_curve.iloc[-1]),
    )
    return result, predictions
