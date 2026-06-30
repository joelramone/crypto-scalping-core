"""Simple reusable grid-search optimizer for research strategies."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.research.backtester import drop_indicator_warmup_rows, load_ohlcv_csv
from app.research.features import compute_features
from app.research.optimizer.leaderboard import (
    LeaderboardRow,
    build_leaderboard_rows,
    print_top_results,
    write_leaderboard_csv,
)
from app.research.simulation import BacktestMetrics, simulate_strategy
from app.research.strategies import BaseStrategy, MeanReversionStrategy

MIN_TRADES = 100

MEAN_REVERSION_GRID: dict[str, list[Any]] = {
    "rsi_threshold": [20.0, 25.0, 30.0, 35.0, 40.0],
    "distance_from_ema20": [-0.001, -0.0015, -0.002, -0.0025, -0.003],
    "volume_ratio": [0.5, 0.7, 0.8, 1.0, 1.2],
    "take_profit_pct": [0.002, 0.0025, 0.003, 0.004],
    "stop_loss_pct": [0.0015, 0.002, 0.0025],
    "max_holding_candles": [10, 15, 20, 30],
}

OPTIMIZER_STRATEGIES: dict[str, type[BaseStrategy]] = {
    "mean_reversion": MeanReversionStrategy,
}

PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "mean_reversion": MEAN_REVERSION_GRID,
}


class GridSearchResult(BaseModel):
    """Completed optimizer evaluation for one parameter combination."""

    strategy: str
    parameters: dict[str, Any]
    metrics: BacktestMetrics


class GridSearchSummary(BaseModel):
    """Full optimizer output before CSV persistence."""

    strategy: str
    evaluated_configurations: int = Field(ge=0)
    ranked_results: list[GridSearchResult]
    leaderboard_rows: list[LeaderboardRow]


def expand_parameter_grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into concrete parameter combinations."""
    parameter_names = list(parameter_grid)
    combinations: list[dict[str, Any]] = []
    for values in product(*(parameter_grid[name] for name in parameter_names)):
        combinations.append(dict(zip(parameter_names, values, strict=True)))
    return combinations


def run_grid_search(
    df: pd.DataFrame,
    strategy_key: str,
    strategy_class: type[BaseStrategy],
    parameter_grid: dict[str, list[Any]],
    min_trades: int = MIN_TRADES,
) -> GridSearchSummary:
    """Run a strategy over every parameter combination and rank passing results."""
    all_results: list[GridSearchResult] = []

    for parameters in expand_parameter_grid(parameter_grid):
        strategy = strategy_class(**parameters)
        metrics = simulate_strategy(df, strategy).metrics
        if metrics.total_trades < min_trades:
            continue
        all_results.append(
            GridSearchResult(
                strategy=strategy_key,
                parameters=parameters,
                metrics=metrics,
            )
        )

    all_results.sort(
        key=lambda result: (result.metrics.profit_factor, result.metrics.expectancy),
        reverse=True,
    )
    ranked_pairs = [(result.parameters, result.metrics) for result in all_results]
    leaderboard_rows = build_leaderboard_rows(strategy_key, ranked_pairs)

    return GridSearchSummary(
        strategy=strategy_key,
        evaluated_configurations=len(expand_parameter_grid(parameter_grid)),
        ranked_results=all_results,
        leaderboard_rows=leaderboard_rows,
    )


def parse_args() -> argparse.Namespace:
    """Parse optimizer command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a research strategy grid search.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(OPTIMIZER_STRATEGIES),
        help="Research strategy to optimize.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to an OHLCV CSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the leaderboard CSV output.",
    )
    return parser.parse_args()


def load_featured_data(data_path: str | Path) -> pd.DataFrame:
    """Load OHLCV data and calculate research features once for optimization."""
    raw_df = load_ohlcv_csv(data_path)
    featured_df = compute_features(raw_df)
    return drop_indicator_warmup_rows(featured_df)


def main() -> None:
    """Run the grid-search optimizer CLI."""
    args = parse_args()
    featured_df = load_featured_data(args.data)
    strategy_class = OPTIMIZER_STRATEGIES[args.strategy]
    parameter_grid = PARAMETER_GRIDS[args.strategy]

    summary = run_grid_search(
        df=featured_df,
        strategy_key=args.strategy,
        strategy_class=strategy_class,
        parameter_grid=parameter_grid,
    )

    write_leaderboard_csv(summary.leaderboard_rows, args.output)
    print(
        f"Evaluated {summary.evaluated_configurations} configurations for "
        f"{args.strategy}."
    )
    print(f"Wrote leaderboard: {args.output}")
    print_top_results(summary.leaderboard_rows, limit=10)


if __name__ == "__main__":
    main()
