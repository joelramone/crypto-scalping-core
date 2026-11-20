"""Frozen HYP-AGGRESSOR-FLOW-PERSISTENCE-001 strategy (kline taker-flow proxy).

See research/hypotheses/aggressor_flow_persistence_v1.md. Every constant here is
pre-registered; changing any of them after observing Dataset A is prohibited.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from app.research.strategies.base import BaseStrategy

THETA = 0.10
K = 3
SIGNAL_TIMEFRAME = "5min"
TAKE_PROFIT_PCT = 0.006
STOP_LOSS_PCT = 0.004
MAX_HOLDING_CANDLES = 12


def resample_to_5m(one_minute: pd.DataFrame) -> pd.DataFrame:
    """Causally aggregate 1m klines into complete 5m candles labeled by open time."""
    frame = one_minute.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    grouped = frame.set_index("timestamp").resample(SIGNAL_TIMEFRAME, label="left", closed="left")
    out = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), taker_buy=("taker_buy_base_volume", "sum"), minutes=("open", "size"),
    )
    out = out[out["minutes"] == 5].drop(columns="minutes")
    return out.reset_index()


def imbalance(candles: pd.DataFrame) -> pd.Series:
    """imb = (2 * taker_buy - volume) / volume; undefined (NaN) when volume is zero."""
    volume = candles["volume"].astype(float)
    taker = candles["taker_buy"].astype(float)
    return ((2.0 * taker - volume) / volume.where(volume != 0.0)).astype(float)


def persistence_state(candles: pd.DataFrame, side: Literal["long", "short"]) -> pd.Series:
    imb = imbalance(candles)
    hit = imb >= THETA if side == "long" else imb <= -THETA
    hit = hit & imb.notna()
    state = hit.astype(int).rolling(K, min_periods=K).sum() == K
    return state.fillna(False).astype(bool)


def onset_signals(candles: pd.DataFrame, side: Literal["long", "short"]) -> pd.Series:
    state = persistence_state(candles, side)
    previous = state.shift(1, fill_value=False).astype(bool)
    return (state & ~previous).astype(bool)


class AggressorFlowPersistenceStrategy(BaseStrategy):
    """Single-direction leg of the frozen aggressor flow persistence baseline."""

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
        return onset_signals(df, self._side)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    def name(self) -> str:
        return f"aggressor_flow_persistence_v1_{self._side}"
