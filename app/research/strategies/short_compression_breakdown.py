"""Pre-registered Family #7 short compression breakdown baseline."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

TAKE_PROFIT_PCT = 0.012
STOP_LOSS_PCT = 0.008
MAX_HOLDING_CANDLES = 24


class ShortCompressionBreakdownParameters(BaseModel):
    """Frozen simulator exit parameters for the Family #7 baseline."""

    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class ShortCompressionBreakdownStrategy(BaseStrategy):
    """Enter short after a strict inside bar closes through its lower boundary."""

    def __init__(
        self,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = ShortCompressionBreakdownParameters(
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def direction(self) -> Literal["short"]:
        return "short"

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return the exact causal inside-bar compression breakdown signal."""
        compression = (
            (df["high"].shift(1) < df["high"].shift(2))
            & (df["low"].shift(1) > df["low"].shift(2))
        )
        breakdown = df["close"] < df["low"].shift(1)
        return (compression & breakdown).fillna(False).astype(bool)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no strategy exits; the official simulator owns every exit."""
        return pd.Series(False, index=df.index, dtype=bool)

    def take_profit_pct(self) -> float:
        return self.parameters.take_profit_pct

    def stop_loss_pct(self) -> float:
        return self.parameters.stop_loss_pct

    def max_holding_candles(self) -> int:
        return self.parameters.max_holding_candles

    def name(self) -> str:
        return "short_compression_breakdown"
