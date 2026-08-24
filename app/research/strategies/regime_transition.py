"""Pre-registered bullish base-regime transition strategy."""

from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

TAKE_PROFIT_PCT = 0.012
STOP_LOSS_PCT = 0.008
MAX_HOLDING_CANDLES = 24


class RegimeTransitionParameters(BaseModel):
    """Frozen simulator exits for HYP-REGIME-TRANSITION-001."""

    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class RegimeTransitionStrategy(BaseStrategy):
    """Enter long exactly when the official base regime becomes TREND_UP."""

    direction: ClassVar[Literal["long_only"]] = "long_only"
    timeframe: ClassVar[Literal["15m"]] = "15m"

    def __init__(
        self,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = RegimeTransitionParameters(
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Apply only the pre-registered non-TREND_UP to TREND_UP rule."""
        base_regime = df["base_regime"]
        return (
            base_regime.shift(1).notna()
            & base_regime.shift(1).ne("TREND_UP")
            & base_regime.eq("TREND_UP")
        ).fillna(False).astype(bool)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no strategy exits; the official simulator owns all exits."""
        return pd.Series(False, index=df.index, dtype=bool)

    def take_profit_pct(self) -> float:
        return self.parameters.take_profit_pct

    def stop_loss_pct(self) -> float:
        return self.parameters.stop_loss_pct

    def max_holding_candles(self) -> int:
        return self.parameters.max_holding_candles

    def name(self) -> str:
        return "regime_transition"
