"""Contract tests for HYP-REGIME-TRANSITION-001."""

import inspect

import pandas as pd

from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.backtester import add_official_base_regime
from app.research.regimes.classifier import classify_regimes
from app.research.simulation import simulate_strategy
from app.research.strategies.regime_transition import RegimeTransitionStrategy


def _regimes(values: list[str], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"base_regime": values}, index=index)


def test_exact_transition_matrix_and_index_alignment() -> None:
    strategy = RegimeTransitionStrategy()
    frame = _regimes(
        ["RANGE", "TREND_UP", "TREND_UP", "TREND_DOWN", "TREND_UP", "RANGE", "TREND_UP", "NEUTRAL", "TREND_UP", "RANGE"],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    signals = strategy.generate_entries(frame)
    assert signals.index.equals(frame.index)
    assert signals.tolist() == [False, True, False, False, True, False, True, False, True, False]


def test_no_future_row_dependency() -> None:
    strategy = RegimeTransitionStrategy()
    original = _regimes(["RANGE", "TREND_UP", "TREND_DOWN"])
    changed_future = _regimes(["RANGE", "TREND_UP", "TREND_UP"])
    assert strategy.generate_entries(original).iloc[:2].equals(
        strategy.generate_entries(changed_future).iloc[:2]
    )


def test_frozen_exits_and_no_strategy_exit() -> None:
    strategy = RegimeTransitionStrategy()
    frame = _regimes(["RANGE", "TREND_UP"])
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert not strategy.generate_exits(frame).any()


def test_official_simulator_compatibility_and_diagnostics() -> None:
    strategy = RegimeTransitionStrategy()
    count = 27
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC"),
            "open": [100.0] * count,
            "high": [100.1] * count,
            "low": [99.9] * count,
            "close": [100.0] * count,
            "base_regime": ["RANGE", "TREND_UP", *(["TREND_UP"] * (count - 2))],
        }
    )
    result = simulate_strategy(frame, strategy)
    assert len(result.trades) == 1
    assert result.trades[0].holding_candles == 24
    diagnostics = calculate_trade_diagnostics(result.trades, [1])
    assert diagnostics.completed_trades == 1
    assert diagnostics.raw_entry_signals == 1
    assert diagnostics.total_fees > 0.0
    assert diagnostics.monthly


def test_official_classifier_is_reused_not_duplicated() -> None:
    source = inspect.getsource(add_official_base_regime)
    assert "classify_regimes" in source
    assert "compute_regime_features" in source
    strategy_source = inspect.getsource(RegimeTransitionStrategy)
    assert "adx" not in strategy_source.lower()
    assert "ema" not in strategy_source.lower()
    assert callable(classify_regimes)
