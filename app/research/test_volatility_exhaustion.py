"""Tests for the frozen volatility-exhaustion baseline."""

import pandas as pd

from app.research.backtester import STRATEGIES
from app.research.simulation import simulate_strategy
from app.research.strategies.volatility_exhaustion import (
    VolatilityExhaustionStrategy,
)


def _candles(closes: list[float], lower: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="15min"),
            "open": closes,
            "high": [value * 1.004 for value in closes],
            "low": [value * 0.996 for value in closes],
            "close": closes,
            "volume": 1.0,
            "bb_lower": lower,
        }
    )


def test_entry_requires_prior_excursion_and_current_reentry() -> None:
    strategy = VolatilityExhaustionStrategy()

    no_excursion = strategy.generate_entries(_candles([101.0, 102.0], [100.0, 100.0]))
    still_below = strategy.generate_entries(_candles([99.0, 98.0], [100.0, 100.0]))
    reentry = strategy.generate_entries(_candles([99.0, 100.0], [100.0, 100.0]))

    assert no_excursion.tolist() == [False, False]
    assert still_below.tolist() == [False, False]
    assert reentry.tolist() == [False, True]


def test_entry_uses_no_future_candle() -> None:
    strategy = VolatilityExhaustionStrategy()
    original = _candles([99.0, 100.0, 500.0], [100.0, 100.0, 1.0])
    changed_future = original.copy()
    changed_future.loc[2, ["close", "bb_lower"]] = [0.5, 1000.0]

    assert strategy.generate_entries(original).iloc[:2].equals(
        strategy.generate_entries(changed_future).iloc[:2]
    )


def test_nan_warmup_is_false_and_index_is_preserved() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0, 101.0], [float("nan"), 100.0, 100.0])
    frame.index = pd.Index([10, 20, 40], name="candle")

    entries = strategy.generate_entries(frame)

    assert entries.index.equals(frame.index)
    assert entries.dtype == bool
    assert not entries.any()


def test_frozen_long_only_parameters_and_exits() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0], [100.0, 100.0])

    assert strategy.direction == "long_only"
    assert strategy.take_profit_pct() == 0.003
    assert strategy.stop_loss_pct() == 0.002
    assert strategy.max_holding_candles() == 25
    assert strategy.generate_exits(frame).eq(False).all()
    assert STRATEGIES["volatility_exhaustion"] is VolatilityExhaustionStrategy


def test_official_simulator_executes_reentry_at_close() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0, 100.4], [100.0, 100.0, 100.0])

    result = simulate_strategy(frame, strategy, fee_rate=0.0)

    assert result.metrics.total_trades == 1
    assert result.trades[0].entry_index == 1
    assert result.trades[0].entry_price == 100.0
    assert result.trades[0].exit_reason == "take_profit"
