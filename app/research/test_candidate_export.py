from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.research.analysis.candidate_export import (
    ExportConfig,
    compute_base_candidates,
    compute_quality_score,
    load_market_data,
    enrich_features,
    simulate_candidate_outcomes,
)
from app.research.analysis.quality_analysis import build_report


def _sample_market_data() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    close = pd.Series([100 + (i * 0.2) for i in range(80)], index=index)
    data = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": [100 + (i % 10) * 15 for i in range(80)],
        },
        index=index,
    )
    return data


def test_candidate_export_preserves_base_subset_relationship() -> None:
    config = ExportConfig(
        strategy="donchian_breakout",
        data=Path("unused"),
        timeframe="15m",
        output=Path("unused"),
    )
    df = enrich_features(_sample_market_data(), config)
    base_candidates = compute_base_candidates(df, config)
    quality_score = compute_quality_score(df, config)
    filtered_candidates = base_candidates & (quality_score >= 5)

    assert filtered_candidates.sum() <= base_candidates.sum()
    assert ((filtered_candidates & ~base_candidates).sum()) == 0


def test_candidate_export_outputs_not_opened_rows_when_position_active() -> None:
    config = ExportConfig(
        strategy="donchian_breakout",
        data=Path("unused"),
        timeframe="15m",
        output=Path("unused"),
        max_holding_candles=5,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    exported = simulate_candidate_outcomes(_sample_market_data(), config)

    assert not exported.empty
    assert set(exported["trade_result"].unique()).issubset({"winner", "loser", "flat", "not_opened"})
    assert (exported["trade_opened"] == False).any()  # noqa: E712


def test_quality_analysis_report_mentions_top_features() -> None:
    df = pd.DataFrame(
        {
            "trade_opened": [True, True, True, True],
            "net_pnl": [0.02, -0.01, 0.03, -0.02],
            "quality_score": [6, 3, 7, 2],
            "body_to_range": [0.8, 0.3, 0.9, 0.2],
            "close_location_value": [0.9, 0.4, 0.95, 0.35],
            "range_expansion_ratio": [1.5, 0.8, 1.6, 0.7],
            "atr_expansion_ratio": [1.2, 0.9, 1.3, 0.85],
            "ema20_slope_pct": [0.01, -0.01, 0.02, -0.02],
            "ema_alignment_strength": [0.03, -0.01, 0.04, -0.02],
            "breakout_distance_pct": [0.01, 0.002, 0.012, 0.001],
            "volume_ratio": [1.8, 0.7, 2.0, 0.6],
            "rsi14": [62, 45, 66, 41],
            "atr14": [1.2, 0.8, 1.3, 0.7],
        }
    )

    report = build_report(df)

    assert "Potentially Useful Features" in report
    assert "Quality Score Performance" in report


def test_load_market_data_requires_timestamp_column(tmp_path) -> None:
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "open_time": ["2026-01-01T00:00:00Z"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    ).to_csv(csv_path, index=False)

    try:
        load_market_data(csv_path, "15m")
    except ValueError as exc:
        assert "Expected timestamp column." in str(exc)
        assert "open_time" in str(exc)
    else:
        raise AssertionError("Expected ValueError when timestamp column is missing")
