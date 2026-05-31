"""CLI entry point: backtest each ticker that has a saved model bundle.

Run from the project root after training:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config /path/to/other_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.engine import backtest_from_bundle  # noqa: E402
from src.backtest.metrics import summarize  # noqa: E402
from src.backtest.report import print_summary, write_report  # noqa: E402
from src.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml. Defaults to the project root config.",
    )
    return parser.parse_args()


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    for ticker in config.data.tickers:
        try:
            result, _predictions = backtest_from_bundle(ticker, config)
        except FileNotFoundError as exc:
            _LOGGER.warning(
                "Skipping %s: %s (train it first with scripts/run_train.py)",
                ticker, exc,
            )
            continue

        strategy_metrics = summarize(
            equity=result.equity_curve,
            daily_returns=result.daily_returns,
            initial_cash=result.initial_cash,
            n_trades=len(result.trades),
        )
        baseline_metrics = summarize(
            equity=result.baseline_curve,
            daily_returns=result.baseline_returns,
            initial_cash=result.initial_cash,
            n_trades=1,
        )

        print_summary(strategy_metrics, baseline_metrics, ticker)
        write_report(
            ticker=ticker,
            result=result,
            strategy_metrics=strategy_metrics,
            baseline_metrics=baseline_metrics,
            reports_dir=config.backtest.reports_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
