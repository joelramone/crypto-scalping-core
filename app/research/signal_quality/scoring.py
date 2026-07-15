"""Deterministic breakout quality scoring."""

from __future__ import annotations

import pandas as pd


def calculate_long_breakout_quality(
    df: pd.DataFrame,
    breakout_distance_pct: pd.Series,
    *,
    min_body_to_range: float,
    min_close_location: float,
    min_range_expansion: float,
    min_atr_expansion: float,
    min_ema20_slope_pct: float,
    min_ema_alignment_strength: float,
    min_breakout_distance_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Score each candle by counting satisfied long-breakout quality conditions."""
    components = pd.DataFrame(
        {
            "strong_body": df["body_to_range"] >= min_body_to_range,
            "close_near_high": df["close_location_value"] >= min_close_location,
            "range_expansion": df["range_expansion_ratio"] >= min_range_expansion,
            "atr_expansion": df["atr_expansion_ratio"] >= min_atr_expansion,
            "positive_trend_slope": df["ema20_slope_pct"] > min_ema20_slope_pct,
            "positive_ema_alignment": (
                df["ema_alignment_strength"] > min_ema_alignment_strength
            ),
            "breakout_distance": breakout_distance_pct >= min_breakout_distance_pct,
        },
        index=df.index,
    )
    quality_score = components.astype(int).sum(axis=1)
    return quality_score, components
