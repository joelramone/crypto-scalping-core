from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from app.research.simulation import calculate_metrics
from app.research.validation.external_oos_validation import (
    BASELINE_FILTER,
    CANDIDATE_FILTER,
    _metrics,
    build_strategy,
    run_external_oos,
)


def _featured_candles() -> pd.DataFrame:
    rows = 800
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series([100.0 + index * 0.08 for index in range(rows)])
    close_location = pd.Series([0.97 if index % 3 else 0.90 for index in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 0.06,
            "high": close + 0.02,
            "low": close - 0.10,
            "close": close,
            "volume_ratio": 1.0,
            "atr14": [1.0 + (index % 21) * 0.1 for index in range(rows)],
            "ema200": close - 1.0,
            "ema20_slope": 0.1,
            "body_to_range": 0.75,
            "close_location_value": close_location,
            "range_expansion_ratio": 1.0,
            "atr_expansion_ratio": 1.0,
            "ema20_slope_pct": 0.01,
            "ema_alignment_strength": 0.01,
        }
    )


def test_variants_are_frozen_and_candidate_signals_are_subset() -> None:
    data = _featured_candles()
    baseline = build_strategy(BASELINE_FILTER)
    candidate = build_strategy(CANDIDATE_FILTER)

    assert baseline.min_close_location_filter == 0.0
    assert candidate.min_close_location_filter == 0.94
    fixed_attributes = (
        "lookback",
        "volume_ratio",
        "_take_profit_pct",
        "_stop_loss_pct",
        "_max_holding_candles",
        "min_quality_score",
    )
    assert all(getattr(baseline, name) == getattr(candidate, name) for name in fixed_attributes)
    baseline_entries = baseline.generate_entries(data)
    candidate_entries = candidate.generate_entries(data)
    assert not (candidate_entries & ~baseline_entries).any()


def test_validation_uses_official_simulator_and_recomputes_trade_metrics() -> None:
    data = _featured_candles()
    from app.research.validation import external_oos_validation as validation

    official_simulator = validation.simulate_strategy
    with patch.object(validation, "simulate_strategy", wraps=official_simulator) as simulator:
        result = run_external_oos(data)

    assert simulator.call_count == 2
    for row, call in zip(result.aggregate_results, simulator.call_args_list):
        official_trades = official_simulator(*call.args, **call.kwargs).trades
        recomputed = _metrics(
            official_trades,
            result.aggregate_results[0].metrics.trades,
        )
        official = calculate_metrics(official_trades)
        assert row.metrics.model_dump() == recomputed.model_dump()
        assert row.metrics.net_pnl == official.net_pnl
        assert row.metrics.max_drawdown == official.max_drawdown


def test_months_use_only_utc_trade_entry_month_and_no_2025_data() -> None:
    result = run_external_oos(_featured_candles())

    assert result.source_first_timestamp.year == 2026
    assert result.source_last_timestamp.year == 2026
    assert {row.period for row in result.monthly_results} == {
        f"2026-{month:02d}" for month in range(1, 9)
    }
    assert all(row.period.startswith("2026-") for row in result.monthly_results)
    january = [row for row in result.monthly_results if row.period == "2026-01"]
    assert [row.metrics.trades for row in january] == [
        row.metrics.trades for row in result.aggregate_results
    ]


def test_2025_featured_data_is_rejected() -> None:
    data = _featured_candles()
    data["timestamp"] = data["timestamp"] - pd.DateOffset(years=1)

    try:
        run_external_oos(data)
    except ValueError as exc:
        assert "only 2026" in str(exc)
    else:
        raise AssertionError("2025 data must not enter external OOS validation")
