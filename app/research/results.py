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


def print_backtest_metrics(metrics: BacktestMetrics) -> None:
    """Print simple rule-based backtest metrics."""
    print("Backtest metrics:")
    print(f"  total trades: {metrics.total_trades}")
    print(f"  wins: {metrics.wins}")
    print(f"  losses: {metrics.losses}")
    print(f"  win rate: {metrics.win_rate:.2%}")
    print(f"  gross pnl: {metrics.gross_pnl:.4f} USDT")
    print(f"  estimated fees: {metrics.estimated_fees:.4f} USDT")
    print(f"  net pnl: {metrics.net_pnl:.4f} USDT")
    print(f"  profit factor: {metrics.profit_factor:.4f}")
    print(f"  expectancy: {metrics.expectancy:.4f} USDT")
    print(f"  average win: {metrics.average_win:.4f} USDT")
    print(f"  average loss: {metrics.average_loss:.4f} USDT")
    print(f"  max drawdown: {metrics.max_drawdown:.4f} USDT")
