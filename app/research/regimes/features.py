"""Causal feature calculations for market-regime research."""

from __future__ import annotations

import pandas as pd

REGIME_FEATURE_COLUMNS = (
    "regime_atr14",
    "atr_pct",
    "realized_volatility_20",
    "regime_ema20",
    "regime_ema50",
    "regime_ema200",
    "regime_ema20_slope_pct",
    "regime_ema50_slope_pct",
    "ema20_ema50_separation",
    "ema50_ema200_separation",
    "adx14",
    "return_4",
    "return_12",
    "return_24",
)


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic trailing features without future observations."""
    result = df.copy()
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        (high - low, (high - previous_close).abs(), (low - previous_close).abs()),
        axis=1,
    ).max(axis=1)
    result["regime_atr14"] = true_range.ewm(
        alpha=1.0 / 14.0, adjust=False, min_periods=14
    ).mean()
    result["atr_pct"] = result["regime_atr14"] / close

    one_candle_return = close.pct_change(fill_method=None)
    result["realized_volatility_20"] = one_candle_return.rolling(
        20, min_periods=20
    ).std()

    for period in (20, 50, 200):
        result[f"regime_ema{period}"] = close.ewm(
            span=period, adjust=False, min_periods=period
        ).mean()

    result["regime_ema20_slope_pct"] = result["regime_ema20"].pct_change(
        fill_method=None
    )
    result["regime_ema50_slope_pct"] = result["regime_ema50"].pct_change(
        fill_method=None
    )
    result["ema20_ema50_separation"] = (
        result["regime_ema20"] / result["regime_ema50"] - 1.0
    )
    result["ema50_ema200_separation"] = (
        result["regime_ema50"] / result["regime_ema200"] - 1.0
    )

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    smoothed_tr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / 14.0, adjust=False, min_periods=14
    ).mean() / smoothed_tr
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / 14.0, adjust=False, min_periods=14
    ).mean() / smoothed_tr
    denominator = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    result["adx14"] = dx.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    for period in (4, 12, 24):
        result[f"return_{period}"] = close.pct_change(period, fill_method=None)
    return result
