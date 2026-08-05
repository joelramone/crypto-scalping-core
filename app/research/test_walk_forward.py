from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import app.research.walk_forward.close_location_walk_forward as walk_forward
from app.research.simulation import BacktestTrade


def _featured_candles() -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", "2026-01-01", freq="6h", inclusive="left", tz="UTC")
    close = pd.Series([100.0 + index * 0.1 for index in range(len(timestamp))])
    return pd.DataFrame({
        "timestamp": timestamp,
        "open": close - 0.05,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume_ratio": 1.0,
        "atr14": [1.0 + (index % 21) * 0.1 for index in range(len(timestamp))],
        "ema200": close - 1.0,
        "ema20_slope": 0.1,
        "body_to_range": 0.75,
        "close_location_value": [0.95 if index % 2 == 0 else 0.90 for index in range(len(timestamp))],
        "range_expansion_ratio": 1.0,
        "atr_expansion_ratio": 1.0,
        "ema20_slope_pct": 0.01,
        "ema_alignment_strength": 0.01,
    })


def test_calendar_windows_and_three_month_step_are_exact() -> None:
    windows = walk_forward.build_windows(pd.Timestamp("2025-01-17", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
    assert [(pd.Timestamp(w.train_start), pd.Timestamp(w.train_end), pd.Timestamp(w.test_start), pd.Timestamp(w.test_end)) for w in windows] == [
        tuple(pd.Timestamp(value, tz="UTC") for value in ("2025-01-01", "2025-07-01", "2025-07-01", "2025-10-01")),
        tuple(pd.Timestamp(value, tz="UTC") for value in ("2025-04-01", "2025-10-01", "2025-10-01", "2026-01-01")),
    ]
    assert all(w.train_end == w.test_start for w in windows)
    assert pd.Timestamp(windows[1].test_start) == pd.Timestamp(windows[0].test_start) + pd.DateOffset(months=3)


def test_baseline_parameters_are_exactly_fixed() -> None:
    assert walk_forward.FIXED_PARAMETERS == {
        "lookback": 3,
        "volume_ratio": 0.4,
        "take_profit_pct": 0.012,
        "stop_loss_pct": 0.008,
        "max_holding_candles": 24,
        "min_quality_score": 0,
    }
    strategy = walk_forward._TestIntervalStrategy(0.0, datetime(2025, 7, 1), datetime(2025, 10, 1))
    assert strategy.lookback == 3
    assert strategy.volume_ratio == 0.4
    assert strategy.take_profit_pct() == 0.012
    assert strategy.stop_loss_pct() == 0.008
    assert strategy.max_holding_candles() == 24
    assert strategy.min_quality_score == 0
    assert strategy.min_close_location_filter == 0.0


def test_filter_094_only_removes_baseline_entry_signals() -> None:
    data = _featured_candles()
    start, end = datetime(2025, 7, 1), datetime(2025, 10, 1)
    baseline = walk_forward._TestIntervalStrategy(0.0, start, end).generate_entries(data)
    filtered = walk_forward._TestIntervalStrategy(0.94, start, end).generate_entries(data)
    assert filtered.sum() > 0
    assert filtered.sum() < baseline.sum()
    assert not (filtered & ~baseline).any()


def test_official_simulator_is_used_for_both_variants_and_entries_are_oos(monkeypatch: pytest.MonkeyPatch) -> None:
    official = walk_forward.simulate_strategy
    calls: list[walk_forward.DonchianBreakoutStrategy] = []

    def spy(df: pd.DataFrame, strategy: walk_forward.DonchianBreakoutStrategy):
        calls.append(strategy)
        return official(df, strategy)

    monkeypatch.setattr(walk_forward, "simulate_strategy", spy)
    result = walk_forward.run_walk_forward(_featured_candles())
    assert len(calls) == len(result.windows) * 2
    assert all(isinstance(strategy, walk_forward.DonchianBreakoutStrategy) for strategy in calls)
    for strategy in calls:
        simulated = official(_featured_candles().loc[_featured_candles()["timestamp"] < strategy.test_end], strategy)
        assert all(strategy.test_start <= pd.Timestamp(t.entry_timestamp) < strategy.test_end for t in simulated.trades)


def _trade(index: int, net_pnl: float) -> BacktestTrade:
    timestamp = pd.Timestamp("2025-07-01", tz="UTC") + pd.Timedelta(minutes=index * 15)
    fees = 0.08
    return BacktestTrade(
        entry_index=index,
        exit_index=index + 1,
        entry_timestamp=timestamp,
        exit_timestamp=timestamp + pd.Timedelta(minutes=15),
        entry_price=100.0,
        exit_price=100.0 + net_pnl + fees,
        notional=100.0,
        gross_pnl=net_pnl + fees,
        fees=fees,
        net_pnl=net_pnl,
        exit_reason="max_holding",
    )


def test_aggregate_pf_and_expectancy_are_recomputed_not_averaged() -> None:
    first = [_trade(0, 4.0), _trade(2, -2.0)]  # PF 2
    second = [_trade(4, 2.0), _trade(6, -3.0)]  # PF 2/3
    aggregate = walk_forward._metrics(first + second, 4, baseline=True)
    assert aggregate.profit_factor == pytest.approx(6.0 / 5.0)
    assert aggregate.profit_factor != pytest.approx((2.0 + 2.0 / 3.0) / 2.0)
    assert aggregate.expectancy == pytest.approx(1.0 / 4.0)
    assert aggregate.net_pnl == pytest.approx(1.0)


def test_walk_forward_concatenates_window_trade_totals() -> None:
    result = walk_forward.run_walk_forward(_featured_candles())
    aggregates = {row.variant: row.metrics for row in result.aggregates}
    for variant in ("baseline", "filtered"):
        rows = [row.metrics for row in result.window_results if row.variant == variant]
        assert aggregates[variant].trades == sum(row.trades for row in rows)
        assert aggregates[variant].net_pnl == pytest.approx(sum(row.net_pnl for row in rows))
    assert aggregates["filtered"].filtered_trade_retention == pytest.approx(aggregates["filtered"].trades / aggregates["baseline"].trades)
