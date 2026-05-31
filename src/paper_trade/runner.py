"""Daily paper trading run.

Designed for once per trading day execution (cron or launchd). Each
run is self contained: load config, load model bundle, fetch the
latest bar, build features, predict, compare with the current Alpaca
position, place an order if the desired position differs, and append
one row to the audit log.

Live data is pulled through the project's own yfinance downloader so
features match training byte for byte. Alpaca is used only for the
trading API (orders, positions, account).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from alpaca.trading.enums import OrderSide

from src.backtest.signals import TradeAction, classify_action, desired_position
from src.config import Config, load_config
from src.data.downloader import download_one
from src.features.pipeline import build_features
from src.model.persistence import load_bundle
from src.paper_trade.client import PaperTradingClient
from src.paper_trade.portfolio_log import LogRow, append_row
from src.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def _load_env_if_present() -> None:
    """Load .env from the project root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _current_position_int(client: PaperTradingClient, ticker: str) -> int:
    """Return 1 if the account holds a long position in `ticker`, else 0."""
    snapshot = client.get_position(ticker)
    if snapshot is None or snapshot.qty <= 0:
        return 0
    return 1


def _whole_shares_to_buy(
    buying_power: float, allocation_fraction: float, price: float
) -> int:
    """Whole shares the allocation will buy at `price`, rounded down."""
    if price <= 0:
        return 0
    allocatable = buying_power * allocation_fraction
    return max(int(allocatable // price), 0)


def run_once(config: Config) -> None:
    """Execute one daily paper trading decision."""
    _load_env_if_present()
    ticker = config.paper_trade.poll_ticker.upper()
    _LOGGER.info("=== Paper trading run for %s ===", ticker)

    bundle_dir = Path(config.training.models_dir) / ticker
    model, scaler, spec = load_bundle(bundle_dir)
    _LOGGER.info("Loaded model bundle from %s", bundle_dir)

    client = PaperTradingClient()

    # Pull fresh daily bars. We always re fetch so today's last bar is
    # included. Cost is small (a few MB per ticker per day).
    today = datetime.now(timezone.utc).date()
    df = download_one(
        ticker=ticker,
        start=config.data.start_date,
        end=today.isoformat(),
        cache_dir=config.data.cache_dir,
        auto_adjust=config.data.auto_adjust,
        force_refresh=True,
    )

    features = build_features(df, spec.features, drop_warmup=True)
    if len(features) < spec.lookback:
        _LOGGER.error(
            "Only %d feature rows available; need at least %d for the lookback. Aborting.",
            len(features), spec.lookback,
        )
        return

    # Take the most recent lookback rows and run them through the saved scaler.
    window = features.iloc[-spec.lookback :]
    decision_date = window.index[-1]
    scaled = scaler.transform(window)
    X = scaled.to_numpy(dtype=np.float32).reshape(
        1, spec.lookback, len(spec.features)
    )
    # Direct forward pass instead of model.predict (see engine.py note).
    probability_up = float(np.asarray(model(X, training=False)).flatten()[0])
    _LOGGER.info(
        "Decision date %s: P(up) = %.4f",
        decision_date.date(), probability_up,
    )

    current_position = _current_position_int(client, ticker)
    target_position = desired_position(
        probability_up,
        current_position,
        buy_threshold=config.backtest.signal.buy_threshold,
        sell_threshold=config.backtest.signal.sell_threshold,
    )
    action = classify_action(current_position, target_position)
    _LOGGER.info(
        "current_position=%d target_position=%d action=%s",
        current_position, target_position, action.value,
    )

    last_price = float(df["close"].iloc[-1])
    order_id: Optional[str] = None
    qty_traded = 0.0

    if action is TradeAction.BUY:
        account = client.get_account()
        qty = _whole_shares_to_buy(
            buying_power=account.buying_power,
            allocation_fraction=config.paper_trade.cash_allocation_fraction,
            price=last_price,
        )
        if qty <= 0:
            _LOGGER.warning(
                "Buying power %.2f cannot afford one share at %.2f; skipping",
                account.buying_power, last_price,
            )
        else:
            order = client.submit_market_order(ticker, qty, OrderSide.BUY)
            order_id = str(getattr(order, "id", ""))
            qty_traded = float(qty)
            _LOGGER.info("Submitted BUY %d %s (order %s)", qty, ticker, order_id)

    elif action is TradeAction.SELL:
        snapshot = client.get_position(ticker)
        qty = float(snapshot.qty) if snapshot is not None else 0.0
        if qty <= 0:
            _LOGGER.warning("No open position to sell; skipping")
        else:
            order = client.submit_market_order(ticker, qty, OrderSide.SELL)
            order_id = str(getattr(order, "id", ""))
            qty_traded = qty
            _LOGGER.info("Submitted SELL %s %s (order %s)", qty, ticker, order_id)
    else:
        _LOGGER.info("No action required")

    # Snapshot the account after any order for the log.
    account = client.get_account()
    append_row(
        LogRow(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            decision_date=str(decision_date.date()),
            probability_up=probability_up,
            current_position=current_position,
            target_position=target_position,
            action=action.value,
            qty_traded=qty_traded,
            last_price=last_price,
            order_id=order_id or "",
            cash=account.cash,
            portfolio_value=account.portfolio_value,
            equity=account.equity,
        ),
        logs_dir=config.paper_trade.logs_dir,
    )
    _LOGGER.info(
        "Logged row. portfolio_value=%.2f cash=%.2f equity=%.2f",
        account.portfolio_value, account.cash, account.equity,
    )


def main(config_path: str | None = None) -> None:
    """Run a single paper trading decision for the configured ticker."""
    config = load_config(config_path)
    run_once(config)


if __name__ == "__main__":
    main()
