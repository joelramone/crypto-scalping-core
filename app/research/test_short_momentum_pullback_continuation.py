"""Synthetic contract tests for HYP-SHORT-PULLBACK-CONTINUATION-001."""

import pandas as pd
import pytest

from app.research.backtester import STRATEGIES
from app.research.optimizer.grid_search import OPTIMIZER_STRATEGIES, PARAMETER_GRIDS
from app.research.strategies.short_momentum_pullback_continuation import (
    ShortMomentumPullbackContinuationStrategy,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [102.0, 101.0, 101.0],
            "high": [103.0, 104.0, 102.0],
            "low": [100.0, 100.0, 98.0],
            "close": [101.0, 103.0, 99.0],
            "ema20": [104.0, 103.0, 102.0],
            "ema50": [106.0, 105.0, 104.0],
            "ema200": [108.0, 107.0, 106.0],
        },
        index=pd.Index([10, 20, 30], name="candle"),
    )


def test_all_predicates_true_generates_only_the_expected_entry() -> None:
    signals = ShortMomentumPullbackContinuationStrategy().generate_entries(_valid_frame())
    assert signals.tolist() == [False, False, True]
    assert signals.index.equals(_valid_frame().index)
    assert signals.dtype == bool


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [
        (30, "ema20", 104.0),
        (30, "ema50", 106.0),
        (20, "high", 102.0),
        (20, "close", 105.0),
        (20, "close", 101.0),
        (30, "close", 101.0),
        (30, "close", 100.0),
    ],
)
def test_each_failed_predicate_independently_prevents_entry(
    row: int, column: str, value: float
) -> None:
    frame = _valid_frame()
    frame.loc[row, column] = value
    assert not ShortMomentumPullbackContinuationStrategy().generate_entries(frame).loc[30]


def test_first_row_without_prior_candle_is_false() -> None:
    signals = ShortMomentumPullbackContinuationStrategy().generate_entries(
        _valid_frame().iloc[:1]
    )
    assert signals.tolist() == [False]


def test_direction_frozen_defaults_and_no_exits() -> None:
    strategy = ShortMomentumPullbackContinuationStrategy()
    exits = strategy.generate_exits(_valid_frame())
    assert strategy.direction() == "short"
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert exits.index.equals(_valid_frame().index)
    assert exits.dtype == bool
    assert not exits.any()


def test_no_regime_column_or_concept_is_required() -> None:
    frame = _valid_frame()
    assert "base_regime" not in frame.columns
    assert ShortMomentumPullbackContinuationStrategy().generate_entries(frame).loc[30]


def test_future_candle_cannot_change_an_existing_signal() -> None:
    strategy = ShortMomentumPullbackContinuationStrategy()
    frame = _valid_frame()
    extended = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    column: [-999.0]
                    for column in ["open", "high", "low", "close", "ema20", "ema50", "ema200"]
                },
                index=pd.Index([40], name="candle"),
            ),
        ]
    )
    assert strategy.generate_entries(frame).equals(strategy.generate_entries(extended).loc[frame.index])


def test_standard_runners_resolve_the_strategy_with_one_frozen_grid() -> None:
    strategy_type = ShortMomentumPullbackContinuationStrategy
    assert STRATEGIES["short_momentum_pullback_continuation"] is strategy_type
    assert OPTIMIZER_STRATEGIES["short_momentum_pullback_continuation"] is strategy_type
    assert PARAMETER_GRIDS["short_momentum_pullback_continuation"] == {
        "take_profit_pct": [0.012],
        "stop_loss_pct": [0.008],
        "max_holding_candles": [24],
    }
