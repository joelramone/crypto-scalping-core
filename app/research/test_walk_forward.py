from __future__ import annotations

import pandas as pd

from app.research.walk_forward.close_location_walk_forward import (
    BASELINE_FILTER,
    FILTERED_FILTER,
    build_windows,
    run_walk_forward,
)


def _featured_candles() -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", "2026-01-01", freq="6h", inclusive="left", tz="UTC")
    close = pd.Series([100.0 + index * 0.1 for index in range(len(timestamp))])
    close_location = pd.Series([0.95 if index % 2 == 0 else 0.90 for index in range(len(timestamp))])
    return pd.DataFrame(
        {
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
            "close_location_value": close_location,
            "range_expansion_ratio": 1.0,
            "atr_expansion_ratio": 1.0,
            "ema20_slope_pct": 0.01,
            "ema_alignment_strength": 0.01,
        }
    )


def test_build_windows_uses_exact_rolling_calendar_boundaries() -> None:
    windows = build_windows(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))

    assert [window.model_dump() for window in windows] == [
        {
            "window": 1,
            "train_start": pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(),
            "train_end": pd.Timestamp("2025-07-01", tz="UTC").to_pydatetime(),
            "test_start": pd.Timestamp("2025-07-01", tz="UTC").to_pydatetime(),
            "test_end": pd.Timestamp("2025-10-01", tz="UTC").to_pydatetime(),
        },
        {
            "window": 2,
            "train_start": pd.Timestamp("2025-04-01", tz="UTC").to_pydatetime(),
            "train_end": pd.Timestamp("2025-10-01", tz="UTC").to_pydatetime(),
            "test_start": pd.Timestamp("2025-10-01", tz="UTC").to_pydatetime(),
            "test_end": pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        },
    ]


def test_walk_forward_runs_only_fixed_variants_and_recomputes_aggregates() -> None:
    result = run_walk_forward(_featured_candles())

    assert len(result.windows) == 2
    assert [(row.variant, row.min_close_location_filter) for row in result.window_results] == [
        ("baseline", BASELINE_FILTER),
        ("filtered", FILTERED_FILTER),
        ("baseline", BASELINE_FILTER),
        ("filtered", FILTERED_FILTER),
    ]
    aggregate = {row.variant: row.metrics for row in result.aggregates}
    assert aggregate["baseline"].trades == sum(
        row.metrics.trades for row in result.window_results if row.variant == "baseline"
    )
    assert aggregate["filtered"].trades == sum(
        row.metrics.trades for row in result.window_results if row.variant == "filtered"
    )
    assert aggregate["filtered"].filtered_trade_retention == (
        aggregate["filtered"].trades / aggregate["baseline"].trades
    )
    baseline_windows = [
        row.metrics for row in result.window_results if row.variant == "baseline"
    ]
    assert aggregate["baseline"].expectancy == (
        sum(row.net_pnl for row in baseline_windows) / aggregate["baseline"].trades
    )
    assert aggregate["baseline"].net_pnl == sum(row.net_pnl for row in baseline_windows)
