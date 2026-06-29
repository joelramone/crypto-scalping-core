"""Baseline trend-following research strategy."""

import pandas as pd

from app.research.strategies.base import BaseStrategy

ATR_MEDIAN_WINDOW = 200


class BaselineTrendStrategy(BaseStrategy):
    """First explicit long-only trend baseline for research comparisons."""

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return baseline long-only entry signals."""
        strategy_df = df.copy()
        strategy_df["atr14_median"] = strategy_df["atr14"].rolling(
            window=ATR_MEDIAN_WINDOW,
            min_periods=1,
        ).median()

        return (
            (strategy_df["close"] > strategy_df["ema200"])
            & (strategy_df["ema20_slope"] > 0.0)
            & (strategy_df["rsi14"] >= 45.0)
            & (strategy_df["rsi14"] <= 70.0)
            & (strategy_df["volume_ratio"] > 1.2)
            & (strategy_df["atr14"] > strategy_df["atr14_median"])
        )

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no strategy-specific exits for the baseline."""
        return pd.Series(False, index=df.index)

    def name(self) -> str:
        """Return the strategy display name."""
        return "Baseline Trend"
