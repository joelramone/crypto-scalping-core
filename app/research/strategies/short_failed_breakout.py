"""Pre-registered Family #8 short failed-breakout baseline."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

TAKE_PROFIT_PCT = 0.012
STOP_LOSS_PCT = 0.008
MAX_HOLDING_CANDLES = 24


class ShortFailedBreakoutParameters(BaseModel):
    """Frozen simulator exit parameters for the Family #8 baseline."""

    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class ShortFailedBreakoutStrategy(BaseStrategy):
    """Enter short after a bullish breakout closes back below resistance."""

    def __init__(
        self,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = ShortFailedBreakoutParameters(
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def direction(self) -> Literal["short"]:
        return "short"

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return the exact causal failed-breakout signal."""
        resistance = df["high"].shift(3).rolling(window=3, min_periods=3).max()
        breakout = (df["high"].shift(1) > resistance) & (
            df["close"].shift(1) > resistance
        )
        rejection = df["close"] < resistance
        return (breakout & rejection).fillna(False).astype(bool)

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
        return "short_failed_breakout"
