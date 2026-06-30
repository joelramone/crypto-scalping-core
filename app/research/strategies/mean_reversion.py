"""Mean reversion research strategy."""

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

ATR_MEDIAN_WINDOW = 200
TAKE_PROFIT_PCT = 0.0025
STOP_LOSS_PCT = 0.002
MAX_HOLDING_CANDLES = 20
RSI_THRESHOLD = 30.0
DISTANCE_FROM_EMA20 = -0.002
VOLUME_RATIO = 0.8


class MeanReversionParameters(BaseModel):
    """Configurable mean reversion strategy parameters."""

    rsi_threshold: float = Field(default=RSI_THRESHOLD, gt=0.0, lt=100.0)
    distance_from_ema20: float = Field(default=DISTANCE_FROM_EMA20, lt=0.0)
    volume_ratio: float = Field(default=VOLUME_RATIO, ge=0.0)
    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class MeanReversionStrategy(BaseStrategy):
    """Long-only mean reversion strategy for research backtests."""

    def __init__(
        self,
        rsi_threshold: float = RSI_THRESHOLD,
        distance_from_ema20: float = DISTANCE_FROM_EMA20,
        volume_ratio: float = VOLUME_RATIO,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = MeanReversionParameters(
            rsi_threshold=rsi_threshold,
            distance_from_ema20=distance_from_ema20,
            volume_ratio=volume_ratio,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return mean reversion long-only entry signals."""
        strategy_df = df.copy()
        strategy_df["atr14_median"] = strategy_df["atr14"].rolling(
            window=ATR_MEDIAN_WINDOW,
            min_periods=1,
        ).median()

        return (
            (strategy_df["close"] < strategy_df["ema20"])
            & (strategy_df["distance_from_ema20"] < self.parameters.distance_from_ema20)
            & (strategy_df["rsi14"] < self.parameters.rsi_threshold)
            & (strategy_df["volume_ratio"] > self.parameters.volume_ratio)
            & (strategy_df["atr14"] > strategy_df["atr14_median"])
        )

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no indicator-specific exits for mean reversion."""
        return pd.Series(False, index=df.index)

    def take_profit_pct(self) -> float:
        """Return the strategy-specific take-profit percentage."""
        return self.parameters.take_profit_pct

    def stop_loss_pct(self) -> float:
        """Return the strategy-specific stop-loss percentage."""
        return self.parameters.stop_loss_pct

    def max_holding_candles(self) -> int:
        """Return the strategy-specific maximum holding time in candles."""
        return self.parameters.max_holding_candles

    def name(self) -> str:
        """Return the strategy display name."""
        return "Mean Reversion"
