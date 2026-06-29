"""Result output helpers for research tooling."""

from pathlib import Path

import pandas as pd

from app.research.simulation import BacktestMetrics


def print_dataset_summary(df: pd.DataFrame, data_path: str | Path) -> None:
    """Print a basic OHLCV dataset summary."""
    print(f"Loaded data: {data_path}")
    print(f"Total candles: {len(df)}")
    print(f"First timestamp: {df['timestamp'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp'].iloc[-1]}")
    print(f"First close: {df['close'].iloc[0]}")
    print(f"Last close: {df['close'].iloc[-1]}")


def print_backtest_metrics(strategy_name: str, metrics: BacktestMetrics) -> None:
    """Print strategy comparison backtest metrics."""
    print("Backtest metrics:")
    print(f"  Strategy Name: {strategy_name}")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.4f}")
    print(f"  Expectancy: {metrics.expectancy:.4f} USDT")
    print(f"  Max Drawdown: {metrics.max_drawdown:.4f} USDT")
