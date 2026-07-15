"""Feature helpers for deterministic signal-quality analysis."""

from __future__ import annotations

import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while treating zero denominators as zero outputs."""
    denominator_nonzero = denominator.where(denominator != 0.0)
    return numerator.div(denominator_nonzero).fillna(0.0)


def add_signal_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append candle-structure and expansion features used by quality filters."""
    features = df.copy()

    candle_range = features["high"] - features["low"]
    body_abs = (features["close"] - features["open"]).abs()
    upper_wick = features["high"] - features[["open", "close"]].max(axis=1)
    lower_wick = features[["open", "close"]].min(axis=1) - features["low"]

    features["body_abs"] = body_abs
    features["body_to_range"] = safe_divide(body_abs, candle_range)
    features["close_location_value"] = safe_divide(features["close"] - features["low"], candle_range)
    features["upper_wick_ratio"] = safe_divide(upper_wick, candle_range)
    features["lower_wick_ratio"] = safe_divide(lower_wick, candle_range)

    features["range_sma20"] = candle_range.rolling(window=20, min_periods=20).mean()
    features["range_expansion_ratio"] = safe_divide(candle_range, features["range_sma20"])

    features["atr_sma20"] = features["atr14"].rolling(window=20, min_periods=20).mean()
    features["atr_expansion_ratio"] = safe_divide(features["atr14"], features["atr_sma20"])

    features["ema20_slope_pct"] = safe_divide(features["ema20_slope"], features["close"])
    features["ema50_slope"] = features["ema50"].diff()
    features["ema50_slope_pct"] = safe_divide(features["ema50_slope"], features["close"])
    features["ema_alignment_strength"] = safe_divide(
        features["ema20"] - features["ema200"],
        features["close"],
    )

    return features
