"""Tests for the frozen Momentum Pullback / Trend Continuation baseline."""

import pandas as pd
import pytest

from app.research.backtester import STRATEGIES
from app.research.simulation import simulate_strategy
from app.research.strategies.momentum_pullback_continuation import (
    MomentumPullbackContinuationStrategy,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min"),
            "open": [105.0, 104.0, 104.0],
            "high": [106.0, 105.0, 107.0],
            "low": [103.0, 101.0, 103.0],
            "close": [104.0, 103.0, 106.0],
            "volume": [1.0, 1.0, 1.0],
            "ema20": [103.0, 102.0, 103.0],
            "ema50": [101.0, 101.0, 102.0],
            "ema200": [99.0, 99.0, 100.0],
        }
    )


def test_valid_trend_pullback_and_continuation_produces_entry() -> None:
    entries = MomentumPullbackContinuationStrategy().generate_entries(_valid_frame())
    assert entries.tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("ema20", 102.0),
        ("ema50", 100.0),
        ("low", 102.1),
        ("close", 101.0),
        ("open", 103.0),
    ],
)
def test_invalid_trend_or_pullback_prevents_entry(column: str, value: float) -> None:
    frame = _valid_frame()
    row = 2 if column in {"ema20", "ema50"} else 1
    frame.loc[row, column] = value
    assert not MomentumPullbackContinuationStrategy().generate_entries(frame).iloc[2]


@pytest.mark.parametrize(("column", "value"), [("open", 106.0), ("close", 105.0)])
def test_invalid_confirmation_prevents_entry(column: str, value: float) -> None:
    frame = _valid_frame()
    frame.loc[2, column] = value
    assert not MomentumPullbackContinuationStrategy().generate_entries(frame).iloc[2]


def test_signal_does_not_depend_on_future_rows() -> None:
    strategy = MomentumPullbackContinuationStrategy()
    frame = _valid_frame()
    future = frame.iloc[[2]].copy()
    future["timestamp"] += pd.Timedelta(minutes=15)
    future[["open", "high", "low", "close", "ema20", "ema50", "ema200"]] = -999.0
    extended = pd.concat([frame, future], ignore_index=True)
    assert strategy.generate_entries(frame).equals(strategy.generate_entries(extended).iloc[:3])


def test_frozen_parameters_registration_and_exits() -> None:
    strategy = MomentumPullbackContinuationStrategy()
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert strategy.generate_exits(_valid_frame()).eq(False).all()
    assert STRATEGIES["momentum_pullback_continuation"] is MomentumPullbackContinuationStrategy


def test_official_simulator_enters_at_confirmation_close() -> None:
    frame = _valid_frame()
    extra = frame.iloc[[2]].copy()
    extra["timestamp"] += pd.Timedelta(minutes=15)
    extra[["open", "close"]] = 106.0
    extra["high"] = 108.0
    extra["low"] = 105.0
    frame = pd.concat([frame, extra], ignore_index=True)

    result = simulate_strategy(frame, MomentumPullbackContinuationStrategy(), fee_rate=0.0)

    assert result.metrics.total_trades == 1
    assert result.trades[0].entry_index == 2
    assert result.trades[0].entry_price == 106.0
