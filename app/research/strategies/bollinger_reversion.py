"""Bollinger Reversion research strategy template."""

import pandas as pd

from app.research.strategies.base import BaseStrategy


class BollingerReversionStrategy(BaseStrategy):
    """Empty Bollinger Reversion strategy template for future research."""

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return no entry signals until this template is implemented."""
        return pd.Series(False, index=df.index)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return no exit signals until this template is implemented."""
        return pd.Series(False, index=df.index)

    def name(self) -> str:
        """Return the strategy display name."""
        return "Bollinger Reversion"
