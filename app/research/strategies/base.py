"""Base strategy interface for research backtests."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Minimal interface every research strategy must implement."""

    def take_profit_pct(self) -> float:
        """Return the take-profit percentage used by the simulator."""
        return 0.004

    def stop_loss_pct(self) -> float:
        """Return the stop-loss percentage used by the simulator."""
        return 0.0025

    def max_holding_candles(self) -> int:
        """Return the maximum holding time in candles used by the simulator."""
        return 30

    @abstractmethod
    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series where True marks a long entry signal."""

    @abstractmethod
    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series where True marks a strategy exit signal."""

    @abstractmethod
    def name(self) -> str:
        """Return the human-readable strategy name."""
