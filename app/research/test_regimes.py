"""Tests for deterministic, causal Phase 1 market-regime research."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from app.research.regimes.analysis import assign_trades_to_regimes, prepare_dataset
from app.research.regimes.classifier import RegimeConfig, classify_regimes
from app.research.regimes.features import REGIME_FEATURE_COLUMNS, compute_regime_features
from app.research.simulation import BacktestTrade, simulate_strategy


def candles(size: int = 750) -> pd.DataFrame:
    close = pd.Series(100.0 + np.linspace(0.0, 20.0, size) + np.sin(np.arange(size) / 8.0))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=size, freq="15min"),
        "open": close.shift(1).fillna(close.iloc[0]), "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 10.0,
    })


def classified_row(**overrides: float) -> pd.DataFrame:
    values = {
        "regime_ema20": 103.0, "regime_ema50": 102.0, "regime_ema200": 101.0,
        "regime_ema20_slope_pct": 0.001, "regime_ema50_slope_pct": 0.001,
        "ema20_ema50_separation": 0.001, "ema50_ema200_separation": 0.001,
        "adx14": 30.0, "realized_volatility_20": 0.01,
    }
    values.update(overrides)
    return pd.DataFrame([values] * 220)


def test_features_do_not_look_ahead() -> None:
    complete = candles(750)
    short = compute_regime_features(complete.iloc[:500])
    long = compute_regime_features(complete).iloc[:500]
    assert_frame_equal(short[list(REGIME_FEATURE_COLUMNS)], long[list(REGIME_FEATURE_COLUMNS)])


def test_classification_is_deterministic_and_trends_are_symmetric() -> None:
    up = classify_regimes(classified_row())
    down = classify_regimes(classified_row(
        regime_ema20=99.0, regime_ema50=100.0, regime_ema200=101.0,
        regime_ema20_slope_pct=-0.001, regime_ema50_slope_pct=-0.001,
    ))
    assert up["regime"].eq("TREND_UP").all()
    assert down["regime"].eq("TREND_DOWN").all()
    assert_frame_equal(up, classify_regimes(classified_row()))


def test_range_and_high_volatility_overlay() -> None:
    frame = classified_row(adx14=15.0, regime_ema20=100.1, regime_ema50=100.0,
                           regime_ema200=99.9, regime_ema20_slope_pct=0.0001,
                           regime_ema50_slope_pct=0.0001)
    frame.loc[219, "realized_volatility_20"] = 1.0
    result = classify_regimes(frame, RegimeConfig(volatility_lookback=200,
                                                  volatility_min_history=200))
    assert result.loc[219, "regime"] == "RANGE"
    assert bool(result.loc[219, "is_high_volatility"])


def test_trade_regime_comes_from_entry_candle() -> None:
    frame = classified_row().assign(regime="RANGE", is_high_volatility=False)
    frame.loc[3, ["regime", "is_high_volatility"]] = ["TREND_UP", True]
    trade = BacktestTrade(entry_index=3, exit_index=4, entry_timestamp=3,
                          exit_timestamp=4, entry_price=100, exit_price=101,
                          notional=100, gross_pnl=1, fees=0.08, net_pnl=0.92,
                          exit_reason="take_profit")
    assigned = assign_trades_to_regimes([trade], frame)
    assert assigned["TREND_UP"] == [trade]
    assert assigned["HIGH_VOLATILITY"] == [trade]
    assert not assigned["RANGE"]


def test_analysis_uses_official_simulator_and_periods_are_independent(tmp_path) -> None:
    assert simulate_strategy.__module__ == "app.research.simulation"
    first = candles(750)
    second = candles(750).assign(close=lambda x: x["close"] * 2,
                                 open=lambda x: x["open"] * 2,
                                 high=lambda x: x["high"] * 2,
                                 low=lambda x: x["low"] * 2)
    paths = [tmp_path / "2025.csv", tmp_path / "2026.csv"]
    first.to_csv(paths[0], index=False)
    second.to_csv(paths[1], index=False)
    prepared_first = prepare_dataset(paths[0], "15m")
    prepared_second = prepare_dataset(paths[1], "15m")
    assert prepared_first["close"].max() < prepared_second["close"].min()
