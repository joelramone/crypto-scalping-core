"""Aggregate experiment-level discovery metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class DiscoverySummary:
    """High-level summary across all loaded leaderboard experiments."""

    total_experiments: int
    strategies_tested: list[str]
    best_strategy: str
    best_timeframe: str
    average_profit_factor: float
    average_expectancy: float
    average_drawdown: float


def _best_group_label(df: pd.DataFrame, group_column: str) -> str:
    """Return the group with the highest average profit factor."""
    grouped = (
        df.groupby(group_column, dropna=False)["profit_factor"]
        .mean()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return "n/a"
    return str(grouped.index[0])


def analyze_experiments(df: pd.DataFrame) -> DiscoverySummary:
    """Compute the core discovery summary over merged experiments."""
    if df.empty:
        raise ValueError("Cannot analyze an empty experiment set.")

    return DiscoverySummary(
        total_experiments=len(df),
        strategies_tested=sorted(df["strategy"].dropna().astype(str).unique().tolist()),
        best_strategy=_best_group_label(df, "strategy"),
        best_timeframe=_best_group_label(df, "timeframe"),
        average_profit_factor=float(df["profit_factor"].mean()),
        average_expectancy=float(df["expectancy"].mean()),
        average_drawdown=float(df["max_drawdown"].mean()),
    )


def select_best_experiments(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the strongest experiments ranked by profit factor then expectancy."""
    if df.empty:
        return df.copy()
    return df.sort_values(
        by=["profit_factor", "expectancy"],
        ascending=[False, False],
    ).head(limit)
