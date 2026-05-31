"""CLI entry point: one paper trading decision, plus the portfolio chart.

Intended for once per trading day execution after the market close.
Example launchd or cron entry (4:30pm Eastern, weekdays):

    30 16 * * 1-5  cd /path/to/lstm_stock_strategy && python scripts/run_paper_trade.py

Flags:
    --config PATH   Use a non default config.yaml.
    --plot-only     Skip the trade decision and only redraw the chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.paper_trade.plots import plot_portfolio_value  # noqa: E402
from src.paper_trade.runner import main as runner_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml. Defaults to the project root config.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the trade decision; only redraw the portfolio chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not args.plot_only:
        runner_main(args.config)
    chart = plot_portfolio_value(config.paper_trade.logs_dir)
    if chart is not None:
        print(f"Updated chart: {chart}")


if __name__ == "__main__":
    main()
