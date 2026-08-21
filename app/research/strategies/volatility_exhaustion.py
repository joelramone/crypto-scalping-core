"""Frozen Bollinger re-entry baseline for volatility-exhaustion research."""

from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

TAKE_PROFIT_PCT = 0.003
STOP_LOSS_PCT = 0.002
MAX_HOLDING_CANDLES = 25


class VolatilityExhaustionParameters(BaseModel):
    """Frozen simulator exit parameters for the discovery baseline."""

    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class VolatilityExhaustionStrategy(BaseStrategy):
    """Long-only entry after price re-enters the official lower Bollinger band."""

    direction: ClassVar[Literal["long_only"]] = "long_only"

    def __init__(
        self,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = VolatilityExhaustionParameters(
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return true only on a lower-band re-entry after a prior excursion."""
        previous_excursion = df["close"].shift(1) < df["bb_lower"].shift(1)
        current_reentry = df["close"] >= df["bb_lower"]
        return (previous_excursion & current_reentry).fillna(False).astype(bool)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no strategy exits; the official simulator owns every exit."""
        return pd.Series(False, index=df.index, dtype=bool)

    def take_profit_pct(self) -> float:
        """Return the frozen take-profit percentage."""
        return self.parameters.take_profit_pct

    def stop_loss_pct(self) -> float:
        """Return the frozen stop-loss percentage."""
        return self.parameters.stop_loss_pct

    def max_holding_candles(self) -> int:
        """Return the frozen maximum holding period."""
        return self.parameters.max_holding_candles

    def name(self) -> str:
        """Return the registered strategy name."""
        return "volatility_exhaustion"
