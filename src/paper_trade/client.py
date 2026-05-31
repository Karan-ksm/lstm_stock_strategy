"""Alpaca paper trading client.

Thin wrapper around alpaca-py covering only trading operations:
account info, current position, and market orders. Historical and
daily bars are fetched through the project's own data downloader
(yfinance) so live features match training features exactly.

Credentials come from environment variables. paper=True is the only
mode this wrapper ever constructs, so there is no path from this
codebase to the live trading endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

_ENV_API_KEY = "ALPACA_API_KEY"
_ENV_SECRET = "ALPACA_SECRET_KEY"


@dataclass(frozen=True)
class AccountSnapshot:
    """A view of the paper account at one point in time."""

    cash: float
    equity: float
    buying_power: float
    portfolio_value: float


@dataclass(frozen=True)
class PositionSnapshot:
    """A view of one open position."""

    ticker: str
    qty: float
    market_value: float
    avg_entry_price: float


class PaperTradingClient:
    """Trading wrapper, paper endpoint only."""

    def __init__(self) -> None:
        api_key = os.environ.get(_ENV_API_KEY)
        secret_key = os.environ.get(_ENV_SECRET)
        if not api_key or not secret_key:
            raise RuntimeError(
                f"Set {_ENV_API_KEY} and {_ENV_SECRET} in the environment "
                "(see .env.example). Refusing to operate without credentials."
            )
        # paper=True forces the paper endpoint. This wrapper does not
        # expose a way to construct a live TradingClient.
        self._trading = TradingClient(api_key, secret_key, paper=True)

    def get_account(self) -> AccountSnapshot:
        """Snapshot the current account."""
        acc = self._trading.get_account()
        return AccountSnapshot(
            cash=float(acc.cash),
            equity=float(acc.equity),
            buying_power=float(acc.buying_power),
            portfolio_value=float(acc.portfolio_value),
        )

    def get_position(self, ticker: str) -> Optional[PositionSnapshot]:
        """Return the open position for `ticker`, or None if flat.

        alpaca-py raises when no position exists; we catch that and
        return None so callers can branch cleanly.
        """
        try:
            pos = self._trading.get_open_position(ticker.upper())
        except Exception:  # noqa: BLE001  (alpaca raises a generic APIError)
            return None
        return PositionSnapshot(
            ticker=ticker.upper(),
            qty=float(pos.qty),
            market_value=float(pos.market_value),
            avg_entry_price=float(pos.avg_entry_price),
        )

    def submit_market_order(
        self, ticker: str, qty: float, side: OrderSide
    ) -> object:
        """Submit a day market order. Returns the alpaca-py Order object.

        Alpaca rejects fractional quantities on market orders by
        default, so callers must round to whole shares.
        """
        request = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        return self._trading.submit_order(request)
