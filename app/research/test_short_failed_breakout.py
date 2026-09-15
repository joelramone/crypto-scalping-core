"""Synthetic contract tests for HYP-SHORT-FAILED-BREAKOUT-001."""

from pathlib import Path

import pandas as pd
import pytest

from app.research.backtester import STRATEGIES
from app.research.optimizer.grid_search import (
    OPTIMIZER_STRATEGIES,
    expand_parameter_grid,
    load_grid_search_config,
)
from app.research.strategies.short_failed_breakout import ShortFailedBreakoutStrategy


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [99.0, 101.0, 100.0, 100.0, 103.0, 103.0],
            "high": [100.0, 102.0, 101.0, 101.0, 110.0, 104.0],
            "low": [98.0, 99.0, 99.0, 99.0, 102.0, 100.0],
            "close": [99.0, 101.0, 100.0, 100.0, 105.0, 101.0],
        },
        index=pd.Index([10, 20, 30, 40, 50, 60], name="candle"),
    )


def test_exact_rule_and_frozen_resistance_generate_expected_entry() -> None:
    frame = _valid_frame()
    signals = ShortFailedBreakoutStrategy().generate_entries(frame)
    assert signals.tolist() == [False, False, False, False, False, True]
    assert signals.index.equals(frame.index)
    assert signals.dtype == bool


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("high", 102.0),
        ("close", 102.0),
        ("close", 101.0),
    ],
)
def test_breakout_predicates_must_strictly_exceed_resistance(
    column: str, value: float
) -> None:
    frame = _valid_frame()
    frame.loc[50, column] = value
    assert not ShortFailedBreakoutStrategy().generate_entries(frame).loc[60]


def test_rejection_equality_does_not_signal() -> None:
    frame = _valid_frame()
    frame.loc[60, "close"] = 102.0
    assert not ShortFailedBreakoutStrategy().generate_entries(frame).loc[60]


def test_breakout_remaining_accepted_does_not_signal() -> None:
    frame = _valid_frame()
    frame.loc[60, "close"] = 103.0
    assert not ShortFailedBreakoutStrategy().generate_entries(frame).loc[60]


@pytest.mark.parametrize(
    ("row", "column"),
    [
        (10, "high"),
        (20, "high"),
        (30, "high"),
        (50, "high"),
        (50, "close"),
        (60, "close"),
    ],
)
def test_missing_required_values_do_not_signal(row: int, column: str) -> None:
    frame = _valid_frame()
    frame.loc[row, column] = float("nan")
    assert not ShortFailedBreakoutStrategy().generate_entries(frame).loc[60]


@pytest.mark.parametrize("row_count", [0, 1, 2, 3, 4])
def test_insufficient_history_never_signals(row_count: int) -> None:
    signals = ShortFailedBreakoutStrategy().generate_entries(
        _valid_frame().iloc[:row_count]
    )
    assert not signals.any()
    assert signals.dtype == bool


def test_future_candles_cannot_change_existing_signal() -> None:
    strategy = ShortFailedBreakoutStrategy()
    frame = _valid_frame()
    future = pd.DataFrame(
        {
            "open": [-1.0, 1_000_000.0],
            "high": [1_000_000.0, 1_000_001.0],
            "low": [-1_000_000.0, -2.0],
            "close": [-999_999.0, 999_999.0],
        },
        index=[70, 80],
    )
    extended = pd.concat([frame, future])
    assert (
        strategy.generate_entries(frame).loc[60]
        == strategy.generate_entries(extended).loc[60]
    )


def test_direction_frozen_defaults_and_no_exits() -> None:
    frame = _valid_frame()
    strategy = ShortFailedBreakoutStrategy()
    exits = strategy.generate_exits(frame)
    assert strategy.name() == "short_failed_breakout"
    assert strategy.direction() == "short"
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert exits.index.equals(frame.index)
    assert exits.dtype == bool
    assert not exits.any()


def test_only_ohlc_structure_is_required() -> None:
    frame = _valid_frame()
    assert set(frame.columns) == {"open", "high", "low", "close"}
    assert ShortFailedBreakoutStrategy().generate_entries(frame).loc[60]


def test_standard_runners_resolve_strategy() -> None:
    strategy_type = ShortFailedBreakoutStrategy
    assert STRATEGIES["short_failed_breakout"] is strategy_type
    assert OPTIMIZER_STRATEGIES["short_failed_breakout"] is strategy_type


def test_baseline_yaml_expands_to_exactly_one_configuration() -> None:
    config_path = Path(
        "research/optimization/grid_search/short_failed_breakout_baseline.yaml"
    )
    config = load_grid_search_config(config_path)
    assert config.strategy == "short_failed_breakout"
    assert config.data == Path("data/BTCUSDT_1m.csv")
    assert config.timeframe == "15m"
    assert config.period_start == "2025-01-01"
    assert config.period_end == "2025-12-31"
    assert config.hypothesis_id == "HYP-SHORT-FAILED-BREAKOUT-001"
    assert config.preregistered is True
    assert config.anti_tuning is True
    assert expand_parameter_grid(config.parameters) == [
        {
            "take_profit_pct": 0.012,
            "stop_loss_pct": 0.008,
            "max_holding_candles": 24,
        }
    ]
