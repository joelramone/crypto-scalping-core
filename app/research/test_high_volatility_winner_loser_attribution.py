"""Focused invariants for descriptive HIGH_VOLATILITY winner/loser attribution."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from app.research.regimes.high_volatility_winner_loser_attribution import (
    FEATURES,
    FROZEN_TIMEFRAME,
    analyze,
    entry_records,
    frozen_high_volatility_trades,
    summarize,
)
from app.research.simulation import BacktestTrade


def _trade(entry: int, net_pnl: float) -> BacktestTrade:
    return BacktestTrade(
        entry_index=entry,
        exit_index=entry + 1,
        entry_timestamp=entry,
        exit_timestamp=entry + 1,
        entry_price=100.0,
        exit_price=101.0,
        notional=100.0,
        gross_pnl=net_pnl + 0.08,
        fees=0.08,
        net_pnl=net_pnl,
        exit_reason="max_holding",
    )


def _frame(high: list[bool]) -> pd.DataFrame:
    count = len(high)
    data: dict[str, object] = {
        "is_high_volatility": high,
        "high_volatility_threshold": [0.01] * count,
    }
    for offset, feature in enumerate(FEATURES):
        if feature not in data and feature not in ("breakout_distance_pct", "volatility_threshold_ratio"):
            data[feature] = [float(index + offset + 1) for index in range(count)]
    return pd.DataFrame(data)


def test_only_frozen_094_and_high_volatility_official_trades_are_used(monkeypatch) -> None:
    frame = _frame([True, False, True])
    official = [_trade(0, 0.4), _trade(1, -0.2), _trade(2, -0.3)]
    captured: list[object] = []

    class Result:
        trades = official

    def fake_simulator(_: pd.DataFrame, strategy: object) -> Result:
        captured.append(strategy)
        strategy.last_breakout_distance_pct = pd.Series([0.1, 0.2, 0.3])
        return Result()

    monkeypatch.setattr(
        "app.research.regimes.high_volatility_winner_loser_attribution.simulate_strategy",
        fake_simulator,
    )
    selected, breakout = frozen_high_volatility_trades(frame)
    assert captured[0].min_close_location_filter == 0.94
    assert FROZEN_TIMEFRAME == "15m"
    assert captured[0].name() == "donchian_breakout"
    assert captured[0].lookback == 3
    assert captured[0].volume_ratio == 0.4
    assert captured[0].take_profit_pct() == 0.012
    assert captured[0].stop_loss_pct() == 0.008
    assert captured[0].max_holding_candles() == 24
    assert captured[0].min_quality_score == 0
    assert selected == [official[0], official[2]]
    assert all(isinstance(item, BacktestTrade) for item in selected)
    assert all(any(item is source for source in official) for item in selected)
    records = entry_records(frame, selected, breakout, "2025")
    assert records["period"].eq("2025").all()
    assert len(records) == 2


def test_entry_records_reject_non_high_volatility_trade() -> None:
    frame = _frame([False])
    with pytest.raises(AssertionError, match="non-HIGH_VOLATILITY"):
        entry_records(frame, [_trade(0, 0.2)], pd.Series([0.1]), "2025")


def test_periods_remain_independently_attributable(monkeypatch) -> None:
    calls: list[pd.DataFrame] = []

    def fake_extract(frame: pd.DataFrame) -> tuple[list[BacktestTrade], pd.Series]:
        calls.append(frame)
        return [_trade(0, float(frame.attrs["net_pnl"]))], pd.Series([0.1])

    monkeypatch.setattr(
        "app.research.regimes.high_volatility_winner_loser_attribution.frozen_high_volatility_trades",
        fake_extract,
    )
    first, second = _frame([True]), _frame([True])
    first.attrs["net_pnl"], second.attrs["net_pnl"] = 0.2, -0.2
    rows = analyze({"2025": first, "2026": second})
    assert calls == [first, second]
    assert {row.period for row in rows} == {"2025", "2026", "combined"}
    assert next(row for row in rows if row.period == "2025").winner_count == 1
    assert next(row for row in rows if row.period == "2026").loser_count == 1
    with pytest.raises(ValueError, match="independently"):
        analyze({"2025": first})


def test_summary_is_descriptive_and_performs_no_threshold_optimization() -> None:
    frame = pd.DataFrame({
        "period": ["2025"] * 4,
        "net_pnl": [0.4, 0.2, -0.1, -0.3],
        "is_winner": [True, True, False, False],
        **{feature: [4.0, 2.0, 1.0, 3.0] for feature in FEATURES},
    })
    row = summarize(frame, "2025")[0]
    assert row.winner_count == row.loser_count == 2
    assert row.winner_mean == 3.0
    assert row.loser_mean == 2.0
    assert row.difference_winner_minus_loser == 1.0
    source = inspect.getsource(summarize)
    assert "threshold" not in source
    assert "grid" not in source
    assert "optimiz" not in source
