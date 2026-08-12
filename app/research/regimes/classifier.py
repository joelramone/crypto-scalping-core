"""Explicit deterministic market-regime classification rules."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

BASE_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "NEUTRAL")
HIGH_VOLATILITY = "HIGH_VOLATILITY"


class RegimeConfig(BaseModel):
    """Conventional Phase 1 thresholds, deliberately unrelated to strategy PnL."""

    trend_adx_threshold: float = Field(default=25.0, gt=0.0)
    range_adx_threshold: float = Field(default=20.0, gt=0.0)
    range_max_ema_separation_pct: float = Field(default=0.0025, gt=0.0)
    range_max_slope_pct: float = Field(default=0.0005, gt=0.0)
    high_volatility_percentile: float = Field(default=0.90, gt=0.5, lt=1.0)
    volatility_lookback: int = Field(default=500, ge=20)
    volatility_min_history: int = Field(default=200, ge=20)


def classify_regimes(
    df: pd.DataFrame, config: RegimeConfig | None = None
) -> pd.DataFrame:
    """Add an exclusive base regime and a separate high-volatility overlay."""
    settings = config or RegimeConfig()
    result = df.copy()
    ema20 = result["regime_ema20"]
    ema50 = result["regime_ema50"]
    ema200 = result["regime_ema200"]
    slope20 = result["regime_ema20_slope_pct"]
    slope50 = result["regime_ema50_slope_pct"]
    adx = result["adx14"]

    trend_up = (
        (ema20 > ema50)
        & (ema50 > ema200)
        & (slope20 > 0.0)
        & (slope50 > 0.0)
        & (adx >= settings.trend_adx_threshold)
    )
    trend_down = (
        (ema20 < ema50)
        & (ema50 < ema200)
        & (slope20 < 0.0)
        & (slope50 < 0.0)
        & (adx >= settings.trend_adx_threshold)
    )
    range_regime = (
        (adx <= settings.range_adx_threshold)
        & (result["ema20_ema50_separation"].abs() <= settings.range_max_ema_separation_pct)
        & (result["ema50_ema200_separation"].abs() <= settings.range_max_ema_separation_pct)
        & (slope20.abs() <= settings.range_max_slope_pct)
        & (slope50.abs() <= settings.range_max_slope_pct)
    )

    result["regime"] = "NEUTRAL"
    result.loc[range_regime, "regime"] = "RANGE"
    result.loc[trend_up, "regime"] = "TREND_UP"
    result.loc[trend_down, "regime"] = "TREND_DOWN"

    # Shift makes the threshold known before the classified candle; the current
    # observation is never used to decide whether it is unusually volatile.
    threshold = result["realized_volatility_20"].rolling(
        settings.volatility_lookback,
        min_periods=settings.volatility_min_history,
    ).quantile(settings.high_volatility_percentile).shift(1)
    result["high_volatility_threshold"] = threshold
    result["is_high_volatility"] = (
        result["realized_volatility_20"] > threshold
    ).fillna(False)
    return result
