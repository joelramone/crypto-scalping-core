"""Minimal executable research backtester entry point."""

import argparse
from pathlib import Path

import pandas as pd

from app.research.config import DEFAULT_DATA_PATH
from app.research.results import print_dataset_summary

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research backtester."""
    parser = argparse.ArgumentParser(description="Run the V2 research backtester.")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to an OHLCV CSV file.",
    )
    return parser.parse_args()


def load_ohlcv_csv(data_path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV candles from a CSV file."""
    df = pd.read_csv(data_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required CSV columns: {missing}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        df[column] = df[column].astype(float)

    return df


def main() -> None:
    """Run the research backtester entry point."""
    args = parse_args()
    df = load_ohlcv_csv(args.data)

    print("Research backtester initialized")
    print_dataset_summary(df, args.data)


if __name__ == "__main__":
    main()
