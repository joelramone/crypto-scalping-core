"""Minimal executable research backtester entry point."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.config import DEFAULT_DATA_PATH
from app.research.features import FEATURE_COLUMNS, compute_features
from app.research.results import print_backtest_metrics, print_dataset_summary
from app.research.simulation import simulate_strategy
from app.research.strategies import (
    BaseStrategy,
    BaselineTrendStrategy,
    BollingerReversionStrategy,
    DonchianBreakoutStrategy,
    EmaPullbackStrategy,
    MeanReversionStrategy,
    PullbackStrategy,
)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
STRATEGIES: dict[str, type[BaseStrategy]] = {
    "baseline_trend": BaselineTrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "pullback": PullbackStrategy,
    "bollinger_reversion": BollingerReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "ema_pullback": EmaPullbackStrategy,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research backtester."""
    parser = argparse.ArgumentParser(description="Run the V2 research backtester.")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to an OHLCV CSV file.",
    )
    parser.add_argument(
        "--strategy",
        default="baseline_trend",
        choices=sorted(STRATEGIES),
        help="Research strategy to run.",
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


def drop_indicator_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values introduced by indicator warmup periods."""
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def format_feature_value(value: Any) -> str:
    """Format feature values for compact command-line output."""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a summary of calculated technical features."""
    last_row = df.iloc[-1]

    print(f"Total rows after features: {len(df)}")
    print(f"Feature columns created: {', '.join(FEATURE_COLUMNS)}")
    print("Last row feature values:")
    for column in FEATURE_COLUMNS:
        print(f"  {column}: {format_feature_value(last_row[column])}")


def load_strategy(strategy_key: str) -> BaseStrategy:
    """Instantiate a configured research strategy by CLI key."""
    return STRATEGIES[strategy_key]()


def main() -> None:
    """Run the research backtester entry point."""
    args = parse_args()
    strategy = load_strategy(args.strategy)
    df = load_ohlcv_csv(args.data)
    featured_df = compute_features(df)
    featured_df = drop_indicator_warmup_rows(featured_df)

    print("Research backtester initialized")
    print_dataset_summary(df, args.data)
    print_feature_summary(featured_df)

    backtest_result = simulate_strategy(featured_df, strategy)
    print_backtest_metrics(strategy.name(), backtest_result.metrics)


if __name__ == "__main__":
    main()
