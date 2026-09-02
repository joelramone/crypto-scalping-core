"""Contract tests for HYP-SHORT-DIRECTIONAL-001."""

import pandas as pd

from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.simulation import simulate_strategy
from app.research.strategies.regime_transition import RegimeTransitionStrategy
from app.research.strategies.short_regime_transition import ShortRegimeTransitionStrategy


def _regimes(values: list[str], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"base_regime": values}, index=index)


def test_exact_transition_matrix_and_index_alignment() -> None:
    strategy = ShortRegimeTransitionStrategy()
    frame = _regimes(
        [
            "TREND_UP",
            "TREND_DOWN",
            "TREND_DOWN",
            "TREND_UP",
            "RANGE",
            "TREND_DOWN",
            "TREND_UP",
            "NEUTRAL",
            "TREND_DOWN",
            "RANGE",
        ],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    signals = strategy.generate_entries(frame)
    assert signals.index.equals(frame.index)
    assert signals.tolist() == [False, True, False, False, False, True, False, False, True, False]


def test_non_entry_transitions() -> None:
    strategy = ShortRegimeTransitionStrategy()
    for current in ("TREND_UP", "RANGE"):
        assert not strategy.generate_entries(_regimes(["TREND_DOWN", current])).iloc[-1]


def test_no_future_row_dependency() -> None:
    strategy = ShortRegimeTransitionStrategy()
    original = _regimes(["RANGE", "TREND_DOWN", "TREND_UP"])
    changed_future = _regimes(["RANGE", "TREND_DOWN", "TREND_DOWN"])
    assert strategy.generate_entries(original).iloc[:2].equals(
        strategy.generate_entries(changed_future).iloc[:2]
    )


def test_direction_and_frozen_exits() -> None:
    strategy = ShortRegimeTransitionStrategy()
    frame = _regimes(["RANGE", "TREND_DOWN"])
    assert strategy.direction() == "short"
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert not strategy.generate_exits(frame).any()


def test_official_simulator_and_permanent_diagnostics_accept_short_trade() -> None:
    count = 27
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC"),
            "open": [100.0] * count,
            "high": [100.1] * count,
            "low": [99.9] * count,
            "close": [100.0] * count,
            "base_regime": ["RANGE", "TREND_DOWN", *(["TREND_DOWN"] * (count - 2))],
        }
    )
    result = simulate_strategy(frame, ShortRegimeTransitionStrategy())
    assert len(result.trades) == 1
    assert result.trades[0].side == "short"
    assert result.trades[0].holding_candles == 24
    diagnostics = calculate_trade_diagnostics(result.trades, [1])
    assert diagnostics.completed_trades == 1
    assert diagnostics.raw_entry_signals == 1


def test_existing_bullish_transition_behavior_remains_unchanged() -> None:
    frame = _regimes(["RANGE", "TREND_UP", "TREND_UP", "TREND_DOWN", "TREND_UP"])
    assert RegimeTransitionStrategy().generate_entries(frame).tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]
