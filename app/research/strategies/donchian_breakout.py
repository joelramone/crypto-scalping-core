"""Donchian Breakout research strategy."""

import pandas as pd
from pydantic import BaseModel, Field

from app.research.strategies.base import BaseStrategy

ATR_MEDIAN_WINDOW = 200
LOOKBACK = 20
VOLUME_RATIO = 1.2
TAKE_PROFIT_PCT = 0.004
STOP_LOSS_PCT = 0.0025
MAX_HOLDING_CANDLES = 30


class DonchianBreakoutParameters(BaseModel):
    """Configurable Donchian breakout strategy parameters."""

    lookback: int = Field(default=LOOKBACK, ge=2)
    volume_ratio: float = Field(default=VOLUME_RATIO, ge=0.0)
    take_profit_pct: float = Field(default=TAKE_PROFIT_PCT, gt=0.0)
    stop_loss_pct: float = Field(default=STOP_LOSS_PCT, gt=0.0)
    max_holding_candles: int = Field(default=MAX_HOLDING_CANDLES, ge=1)


class DonchianBreakoutStrategy(BaseStrategy):
    """Long-only Donchian breakout strategy for research backtests."""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        volume_ratio: float = VOLUME_RATIO,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_holding_candles: int = MAX_HOLDING_CANDLES,
    ) -> None:
        self.parameters = DonchianBreakoutParameters(
            lookback=lookback,
            volume_ratio=volume_ratio,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_candles=max_holding_candles,
        )

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return Donchian Breakout long-only entry signals."""
        strategy_df = df.copy()
        lookback = self.parameters.lookback

        strategy_df["donchian_high"] = (
            strategy_df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
        )
        strategy_df["atr14_median"] = strategy_df["atr14"].rolling(
            window=ATR_MEDIAN_WINDOW,
            min_periods=1,
        ).median()

        return (
            (strategy_df["close"] > strategy_df["donchian_high"])
            & (strategy_df["close"] > strategy_df["ema200"])
            & (strategy_df["ema20_slope"] > 0.0)
            & (strategy_df["volume_ratio"] > self.parameters.volume_ratio)
            & (strategy_df["atr14"] > strategy_df["atr14_median"])
        )

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no indicator-specific exits for Donchian Breakout."""
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
        return "Donchian Breakout"
