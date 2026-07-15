"""Discover simple recurring patterns among top-performing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

PATTERN_COLUMNS = {
    "timeframe": "timeframe",
    "TP": "take_profit_pct",
    "SL": "stop_loss_pct",
    "holding": "max_holding_candles",
    "RSI": "rsi_threshold",
    "volume": "volume_ratio",
    "lookback": "lookback",
}


@dataclass(slots=True)
class PatternDiscovery:
    """Top-decile recurring values and supporting metadata."""

    top_bucket_size: int
    profit_factor_threshold: float
    common_values: dict[str, Any]


def _most_common_value(df: pd.DataFrame, column: str) -> Any:
    """Return the most frequent non-null value in a column."""
    if column not in df.columns:
        return "n/a"
    non_null = df[column].dropna()
    if non_null.empty:
        return "n/a"
    mode = non_null.mode()
    if mode.empty:
        return "n/a"
    value = mode.iloc[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def discover_top_profit_factor_patterns(df: pd.DataFrame) -> PatternDiscovery:
    """Mine common settings among the top 10 percent by profit factor."""
    if df.empty:
        raise ValueError("Cannot discover patterns from an empty experiment set.")

    profit_factor_threshold = float(df["profit_factor"].quantile(0.9))
    top_df = df[df["profit_factor"] >= profit_factor_threshold].copy()
    if top_df.empty:
        top_df = df.nlargest(1, "profit_factor").copy()
        profit_factor_threshold = float(top_df["profit_factor"].min())

    common_values = {
        label: _most_common_value(top_df, column)
        for label, column in PATTERN_COLUMNS.items()
    }
    return PatternDiscovery(
        top_bucket_size=len(top_df),
        profit_factor_threshold=profit_factor_threshold,
        common_values=common_values,
    )
