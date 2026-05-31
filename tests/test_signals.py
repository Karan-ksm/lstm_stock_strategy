"""Assert the signal logic matches the documented hysteresis rules."""

from __future__ import annotations

import pytest

from src.backtest.signals import TradeAction, classify_action, desired_position


def test_buys_when_flat_and_probability_strictly_above_buy_threshold() -> None:
    assert desired_position(0.60, 0, buy_threshold=0.55, sell_threshold=0.45) == 1


def test_stays_flat_when_probability_equals_buy_threshold() -> None:
    # Strictly greater than, not greater or equal; so 0.55 does not buy.
    assert desired_position(0.55, 0, buy_threshold=0.55, sell_threshold=0.45) == 0


def test_stays_flat_when_probability_below_buy_threshold() -> None:
    assert desired_position(0.50, 0, buy_threshold=0.55, sell_threshold=0.45) == 0


def test_sells_when_long_and_probability_strictly_below_sell_threshold() -> None:
    assert desired_position(0.40, 1, buy_threshold=0.55, sell_threshold=0.45) == 0


def test_holds_long_when_probability_in_hysteresis_band() -> None:
    # Between sell and buy; already long, so we keep the position.
    assert desired_position(0.50, 1, buy_threshold=0.55, sell_threshold=0.45) == 1


def test_holds_long_when_probability_equals_sell_threshold() -> None:
    # Strictly less than, so 0.45 does not sell.
    assert desired_position(0.45, 1, buy_threshold=0.55, sell_threshold=0.45) == 1


def test_rejects_invalid_current_position() -> None:
    with pytest.raises(ValueError):
        desired_position(0.7, current_position=2, buy_threshold=0.55, sell_threshold=0.45)


def test_classify_action_maps_position_transitions_correctly() -> None:
    assert classify_action(0, 1) == TradeAction.BUY
    assert classify_action(1, 0) == TradeAction.SELL
    assert classify_action(0, 0) == TradeAction.HOLD
    assert classify_action(1, 1) == TradeAction.HOLD
