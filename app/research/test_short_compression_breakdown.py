"""Synthetic contract tests for HYP-SHORT-COMPRESSION-BREAKDOWN-001."""

import pandas as pd
import pytest

from app.research.backtester import STRATEGIES
from app.research.optimizer.grid_search import OPTIMIZER_STRATEGIES, PARAMETER_GRIDS
from app.research.strategies.short_compression_breakdown import (
    ShortCompressionBreakdownStrategy,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.0],
            "high": [110.0, 108.0, 102.0],
            "low": [90.0, 92.0, 88.0],
            "close": [100.0, 100.0, 91.0],
        },
        index=pd.Index([10, 20, 30], name="candle"),
    )


def test_exact_rule_generates_only_expected_entry() -> None:
    frame = _valid_frame()
    signals = ShortCompressionBreakdownStrategy().generate_entries(frame)
    assert signals.tolist() == [False, False, True]
    assert signals.index.equals(frame.index)
    assert signals.dtype == bool


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [
        (20, "high", 110.0),
        (20, "low", 90.0),
        (30, "close", 92.0),
    ],
)
def test_strict_boundaries_do_not_signal(row: int, column: str, value: float) -> None:
    frame = _valid_frame()
    frame.loc[row, column] = value
    assert not ShortCompressionBreakdownStrategy().generate_entries(frame).loc[30]


@pytest.mark.parametrize(
    ("row", "column"),
    [(10, "high"), (10, "low"), (20, "high"), (20, "low"), (30, "close")],
)
def test_missing_required_values_do_not_signal(row: int, column: str) -> None:
    frame = _valid_frame()
    frame.loc[row, column] = float("nan")
    assert not ShortCompressionBreakdownStrategy().generate_entries(frame).loc[30]


@pytest.mark.parametrize("row_count", [0, 1, 2])
def test_fewer_than_three_rows_never_signal(row_count: int) -> None:
    signals = ShortCompressionBreakdownStrategy().generate_entries(_valid_frame().iloc[:row_count])
    assert not signals.any()
    assert signals.dtype == bool


def test_direction_frozen_defaults_and_no_exits() -> None:
    frame = _valid_frame()
    strategy = ShortCompressionBreakdownStrategy()
    exits = strategy.generate_exits(frame)
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
    assert ShortCompressionBreakdownStrategy().generate_entries(frame).loc[30]


def test_future_candle_cannot_change_existing_signal() -> None:
    strategy = ShortCompressionBreakdownStrategy()
    frame = _valid_frame()
    extended = pd.concat(
        [frame, pd.DataFrame({"open": [-1.0], "high": [-1.0], "low": [-1.0], "close": [-1.0]}, index=[40])]
    )
    assert strategy.generate_entries(frame).equals(strategy.generate_entries(extended).loc[frame.index])


def test_standard_runners_resolve_strategy_with_one_frozen_grid() -> None:
    strategy_type = ShortCompressionBreakdownStrategy
    assert STRATEGIES["short_compression_breakdown"] is strategy_type
    assert OPTIMIZER_STRATEGIES["short_compression_breakdown"] is strategy_type
    assert PARAMETER_GRIDS["short_compression_breakdown"] == {
        "take_profit_pct": [0.012],
        "stop_loss_pct": [0.008],
        "max_holding_candles": [24],
    }
