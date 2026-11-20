"""Frozen HYP-VOLATILITY-EXPANSION-BREAKOUT-001 strategy.

See research/hypotheses/volatility_expansion_breakout_v1.md. Every constant here is
pre-registered; changing any of them after observing Dataset B is prohibited.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from app.research.strategies.base import BaseStrategy

SIGNAL_TIMEFRAME = "15min"
MINUTES_PER_CANDLE = 15
ATR_LOOKBACK = 96
EXPANSION_MULT = 2.0
LOC_HIGH = 0.75
LOC_LOW = 0.25
TAKE_PROFIT_PCT = 0.010
STOP_LOSS_PCT = 0.006
MAX_HOLDING_CANDLES = 32


def resample_to_15m(one_minute: pd.DataFrame) -> pd.DataFrame:
    """Causally aggregate 1m klines into complete 15m candles labeled by open time."""
    frame = one_minute.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    grouped = frame.set_index("timestamp").resample(SIGNAL_TIMEFRAME, label="left", closed="left")
    out = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                      close=("close", "last"), volume=("volume", "sum"), minutes=("open", "size"))
    out = out[out["minutes"] == MINUTES_PER_CANDLE].drop(columns="minutes")
    return out.reset_index()


def true_range(candles: pd.DataFrame) -> pd.Series:
    prev_close = candles["close"].shift(1)
    return pd.concat([candles["high"] - candles["low"], (candles["high"] - prev_close).abs(),
                      (candles["low"] - prev_close).abs()], axis=1).max(axis=1, skipna=False)


def atr_prev(candles: pd.DataFrame) -> pd.Series:
    """Mean TR of the previous ATR_LOOKBACK candles, excluding the current one."""
    return true_range(candles).shift(1).rolling(ATR_LOOKBACK, min_periods=ATR_LOOKBACK).mean()


def expansion(candles: pd.DataFrame) -> pd.Series:
    return (true_range(candles) >= EXPANSION_MULT * atr_prev(candles)).fillna(False).astype(bool)


def close_location(candles: pd.DataFrame) -> pd.Series:
    span = candles["high"] - candles["low"]
    return (candles["close"] - candles["low"]) / span.where(span > 0)


def signals(candles: pd.DataFrame, side: Literal["long", "short"]) -> pd.Series:
    loc = close_location(candles)
    if side == "long":
        directional = (loc >= LOC_HIGH) & (candles["close"] > candles["open"])
    else:
        directional = (loc <= LOC_LOW) & (candles["close"] < candles["open"])
    return (expansion(candles) & directional.fillna(False)).astype(bool)


class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    """Single-direction leg of the frozen volatility expansion breakout baseline."""

    def __init__(self, side: Literal["long", "short"]) -> None:
        if side not in ("long", "short"):
            raise ValueError(f"Unsupported side: {side!r}")
        self._side = side

    def direction(self) -> Literal["long", "short"]:
        return self._side

    def take_profit_pct(self) -> float:
        return TAKE_PROFIT_PCT

    def stop_loss_pct(self) -> float:
        return STOP_LOSS_PCT

    def max_holding_candles(self) -> int:
        return MAX_HOLDING_CANDLES

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        return signals(df, self._side)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    def name(self) -> str:
        return f"volatility_expansion_breakout_v1_{self._side}"
