"""Base strategy interface for research backtests."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Minimal interface every research strategy must implement."""

    @abstractmethod
    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series where True marks a long entry signal."""

    @abstractmethod
    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series where True marks a strategy exit signal."""

    @abstractmethod
    def name(self) -> str:
        """Return the human-readable strategy name."""
