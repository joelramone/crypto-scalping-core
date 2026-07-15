"""Shared dataset helpers for research tooling."""

from __future__ import annotations

import pandas as pd

SUPPORTED_INTERVALS = {"1m": "1min", "5m": "5min", "15m": "15min"}
OPTIONAL_SUM_COLUMNS = (
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    """Convert a timestamp series into pandas datetimes for resampling."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty:
            return pd.to_datetime(series, errors="coerce")
        unit = "ms" if non_null.abs().max() >= 1_000_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Return OHLCV data at the requested research timeframe."""
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval '{interval}'. Supported intervals: {supported}")

    if interval == "1m":
        return df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include a 'timestamp' column for resampling.")

    working_df = df.copy()
    working_df["timestamp"] = _coerce_timestamp(working_df["timestamp"])
    if working_df["timestamp"].isna().any():
        raise ValueError("Unable to parse one or more timestamps for resampling.")

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for column in OPTIONAL_SUM_COLUMNS:
        if column in working_df.columns:
            aggregation[column] = "sum"

    resampled = (
        working_df.sort_values("timestamp")
        .set_index("timestamp")
        .resample(SUPPORTED_INTERVALS[interval], label="left", closed="left")
        .agg(aggregation)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled
