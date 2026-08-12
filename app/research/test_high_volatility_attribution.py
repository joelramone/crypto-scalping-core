"""Focused invariants for HIGH_VOLATILITY attribution."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from app.research.regimes.classifier import RegimeConfig
from app.research.regimes.high_volatility_attribution import (
    FROZEN_CANDIDATE_FILTER,
    attribute,
    build_strategy,
    partition_entry_high_volatility,
)
from app.research.simulation import BacktestTrade


def trade(entry: int, exit_: int, net: float) -> BacktestTrade:
    return BacktestTrade(
        entry_index=entry, exit_index=exit_, entry_timestamp=entry,
        exit_timestamp=exit_, entry_price=100.0, exit_price=101.0,
        notional=100.0, gross_pnl=net + 0.08, fees=0.08, net_pnl=net,
        exit_reason="max_holding",
    )


def test_partitions_reuse_official_records_are_disjoint_and_reconcile_exactly() -> None:
    trades = [trade(0, 2, -0.18), trade(1, 3, 0.32), trade(2, 3, -0.07)]
    frame = pd.DataFrame({"is_high_volatility": [True, False, True, False]})
    high, non_high = partition_entry_high_volatility(trades, frame)
    assert len(high) + len(non_high) == len(trades)
    assert not {id(item) for item in high} & {id(item) for item in non_high}
    assert all(any(item is source for source in trades) for item in high + non_high)
    exact = lambda values: sum((Decimal(str(item.net_pnl)) for item in values), Decimal(0))
    assert exact(trades) == exact(high) + exact(non_high)


def test_partition_uses_entry_time_regime_only() -> None:
    official = trade(0, 2, -0.18)
    first = pd.DataFrame({"is_high_volatility": [False, True, True]})
    changed_exit = pd.DataFrame({"is_high_volatility": [False, True, False]})
    assert partition_entry_high_volatility([official], first)[1] == [official]
    assert partition_entry_high_volatility([official], changed_exit)[1] == [official]


def test_frozen_parameters_and_regime_thresholds_are_unchanged() -> None:
    default = build_strategy("phase1_default")
    candidate = build_strategy("frozen_0.94_candidate")
    fresh_default = type(default)()
    for name in (
        "lookback", "volume_ratio", "min_quality_score", "min_close_location_filter"
    ):
        assert getattr(default, name) == getattr(fresh_default, name)
    assert default.take_profit_pct() == fresh_default.take_profit_pct()
    assert default.stop_loss_pct() == fresh_default.stop_loss_pct()
    assert default.max_holding_candles() == fresh_default.max_holding_candles()
    assert candidate.lookback == 3
    assert candidate.volume_ratio == 0.4
    assert candidate.take_profit_pct() == 0.012
    assert candidate.stop_loss_pct() == 0.008
    assert candidate.max_holding_candles() == 24
    assert candidate.min_quality_score == 0
    assert candidate.min_close_location_filter == FROZEN_CANDIDATE_FILTER == 0.94
    assert RegimeConfig() == RegimeConfig(
        trend_adx_threshold=25.0, range_adx_threshold=20.0,
        range_max_ema_separation_pct=0.0025, range_max_slope_pct=0.0005,
        high_volatility_percentile=0.90, volatility_lookback=500,
        volatility_min_history=200,
    )


def test_attribute_runs_official_simulator_once_and_reuses_records(monkeypatch) -> None:
    records = [trade(0, 1, -0.18), trade(1, 2, 0.32)]
    calls: list[object] = []

    class Result:
        trades = records

    def fake_simulator(frame: pd.DataFrame, strategy: object) -> Result:
        calls.append(strategy)
        return Result()

    monkeypatch.setattr(
        "app.research.regimes.high_volatility_attribution.simulate_strategy",
        fake_simulator,
    )
    rows = attribute(
        pd.DataFrame({"is_high_volatility": [True, False, False]}),
        "2025", "phase1_default",
    )
    assert len(calls) == 1
    assert [row.metrics.trades for row in rows] == [2, 1, 1]
    assert rows[0].metrics.net_pnl == rows[1].metrics.net_pnl + rows[2].metrics.net_pnl
