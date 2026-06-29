"""Mean reversion research strategy."""

import pandas as pd

from app.research.strategies.base import BaseStrategy

ATR_MEDIAN_WINDOW = 200
TAKE_PROFIT_PCT = 0.0025
STOP_LOSS_PCT = 0.002
MAX_HOLDING_CANDLES = 20


class MeanReversionStrategy(BaseStrategy):
    """Long-only mean reversion strategy for research backtests."""

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return mean reversion long-only entry signals."""
        strategy_df = df.copy()
        strategy_df["atr14_median"] = strategy_df["atr14"].rolling(
            window=ATR_MEDIAN_WINDOW,
            min_periods=1,
        ).median()

        return (
            (strategy_df["close"] < strategy_df["ema20"])
            & (strategy_df["distance_from_ema20"] < -0.002)
            & (strategy_df["rsi14"] < 30.0)
            & (strategy_df["volume_ratio"] > 0.8)
            & (strategy_df["atr14"] > strategy_df["atr14_median"])
        )

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no indicator-specific exits for mean reversion."""
        return pd.Series(False, index=df.index)

    def take_profit_pct(self) -> float:
        """Return the strategy-specific take-profit percentage."""
        return TAKE_PROFIT_PCT

    def stop_loss_pct(self) -> float:
        """Return the strategy-specific stop-loss percentage."""
        return STOP_LOSS_PCT

    def max_holding_candles(self) -> int:
        """Return the strategy-specific maximum holding time in candles."""
        return MAX_HOLDING_CANDLES

    def name(self) -> str:
        """Return the strategy display name."""
        return "Mean Reversion"
