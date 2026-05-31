"""Map model probabilities into a desired position.

Strategy is long only, all in or all out. The position state is a
single integer:

    0 = flat (holding cash)
    1 = long (fully invested)

Two thresholds with hysteresis: a higher buy_threshold to enter long
and a lower sell_threshold to exit. The gap between them prevents
whipsawing when the model's probability oscillates near a single
value.
"""

from __future__ import annotations

from enum import Enum


class TradeAction(str, Enum):
    """Action implied by a position change. Strings so logs stay readable."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def desired_position(
    probability_up: float,
    current_position: int,
    buy_threshold: float,
    sell_threshold: float,
) -> int:
    """Decide the position to hold next, given a model probability.

    Args:
        probability_up: Model output, expected in [0, 1].
        current_position: Current position. 0 = flat, 1 = long.
        buy_threshold: Enter long when probability_up exceeds this and
            we are currently flat.
        sell_threshold: Exit to flat when probability_up falls below
            this and we are currently long.

    Returns:
        The position to hold for the next bar (0 or 1).
    """
    if current_position not in (0, 1):
        raise ValueError(
            f"current_position must be 0 or 1, got {current_position}"
        )
    if current_position == 0:
        return 1 if probability_up > buy_threshold else 0
    return 0 if probability_up < sell_threshold else 1


def classify_action(prev_position: int, new_position: int) -> TradeAction:
    """Return the action label implied by going from prev to new position."""
    if prev_position == new_position:
        return TradeAction.HOLD
    return TradeAction.BUY if new_position > prev_position else TradeAction.SELL
