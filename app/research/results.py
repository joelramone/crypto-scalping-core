"""Result output helpers for research tooling."""

from pathlib import Path

import pandas as pd


def print_dataset_summary(df: pd.DataFrame, data_path: str | Path) -> None:
    """Print a basic OHLCV dataset summary."""
    print(f"Loaded data: {data_path}")
    print(f"Total candles: {len(df)}")
    print(f"First timestamp: {df['timestamp'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp'].iloc[-1]}")
    print(f"First close: {df['close'].iloc[0]}")
    print(f"Last close: {df['close'].iloc[-1]}")
