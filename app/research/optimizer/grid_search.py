"""Simple reusable grid-search optimizer for research strategies."""

from __future__ import annotations

import argparse
import importlib.util
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
    "rsi_threshold": [20, 25, 30, 35, 40],
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


class GridSearchConfig(BaseModel):
    """YAML-backed optimizer experiment configuration."""

    strategy: str
    data: Path
    output: Path
    min_trades: int = Field(default=MIN_TRADES, ge=0)
    parameters: dict[str, list[Any]] = Field(min_length=1)


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
    parameter_combinations = expand_parameter_grid(parameter_grid)
    total_combinations = len(parameter_combinations)

    for index, parameters in enumerate(parameter_combinations, start=1):
        print(f"Progress: {index}/{total_combinations}")
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
        evaluated_configurations=total_combinations,
        ranked_results=all_results,
        leaderboard_rows=leaderboard_rows,
    )


def parse_args() -> argparse.Namespace:
    """Parse optimizer command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a research strategy grid search.")
    parser.add_argument(
        "--config",
        help="Path to a YAML experiment config. Overrides strategy, data, output, and params.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(OPTIMIZER_STRATEGIES),
        help="Research strategy to optimize.",
    )
    parser.add_argument(
        "--data",
        help="Path to an OHLCV CSV file.",
    )
    parser.add_argument(
        "--output",
        help="Path for the leaderboard CSV output.",
    )
    return parser.parse_args()


def _parse_scalar(value: str) -> Any:
    """Parse a minimal YAML scalar used by optimizer configs."""
    value = value.strip()
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"\'')


def _load_simple_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load the small mapping/list YAML shape used for optimizer configs."""
    config: dict[str, Any] = {}
    parameters: dict[str, list[Any]] = {}
    current_parameter: str | None = None
    in_parameters = False

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        stripped = line_without_comment.strip()
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))

        if indent == 0:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"Invalid YAML line: {raw_line}")
            in_parameters = key == "parameters"
            current_parameter = None
            if in_parameters:
                config["parameters"] = parameters
            else:
                config[key] = _parse_scalar(value)
            continue

        if in_parameters and indent == 2 and stripped.endswith(":"):
            current_parameter = stripped[:-1]
            parameters[current_parameter] = []
            continue

        if in_parameters and indent == 4 and stripped.startswith("- ") and current_parameter:
            parameters[current_parameter].append(_parse_scalar(stripped[2:]))
            continue

        raise ValueError(f"Unsupported YAML shape near line: {raw_line}")

    return config


def load_grid_search_config(config_path: str | Path) -> GridSearchConfig:
    """Load an optimizer experiment config from YAML."""
    path = Path(config_path)
    yaml_spec = importlib.util.find_spec("yaml")
    if yaml_spec is not None:
        import yaml

        loaded_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        loaded_config = _load_simple_yaml_config(path)
    if not isinstance(loaded_config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return GridSearchConfig.model_validate(loaded_config)


def build_config_from_args(args: argparse.Namespace) -> GridSearchConfig:
    """Resolve CLI arguments into a single optimizer configuration."""
    if args.config:
        return load_grid_search_config(args.config)

    missing_args = [name for name in ("strategy", "data", "output") if getattr(args, name) is None]
    if missing_args:
        missing = ", ".join(f"--{name}" for name in missing_args)
        raise SystemExit(f"Missing required arguments without --config: {missing}")

    return GridSearchConfig(
        strategy=args.strategy,
        data=Path(args.data),
        output=Path(args.output),
        min_trades=MIN_TRADES,
        parameters=PARAMETER_GRIDS[args.strategy],
    )


def load_featured_data(data_path: str | Path) -> pd.DataFrame:
    """Load OHLCV data and calculate research features once for optimization."""
    raw_df = load_ohlcv_csv(data_path)
    featured_df = compute_features(raw_df)
    return drop_indicator_warmup_rows(featured_df)


def main() -> None:
    """Run the grid-search optimizer CLI."""
    args = parse_args()
    config = build_config_from_args(args)
    if config.strategy not in OPTIMIZER_STRATEGIES:
        valid_strategies = ", ".join(sorted(OPTIMIZER_STRATEGIES))
        raise SystemExit(f"Unsupported strategy '{config.strategy}'. Valid choices: {valid_strategies}")

    parameter_combinations = expand_parameter_grid(config.parameters)
    print(f"Strategy: {config.strategy}")
    print(f"Data path: {config.data}")
    print(f"Output path: {config.output}")
    print(f"Parameter combinations: {len(parameter_combinations)}")

    featured_df = load_featured_data(config.data)
    strategy_class = OPTIMIZER_STRATEGIES[config.strategy]

    summary = run_grid_search(
        df=featured_df,
        strategy_key=config.strategy,
        strategy_class=strategy_class,
        parameter_grid=config.parameters,
        min_trades=config.min_trades,
    )

    write_leaderboard_csv(summary.leaderboard_rows, config.output)
    print(
        f"Evaluated {summary.evaluated_configurations} configurations for "
        f"{config.strategy}."
    )
    print(f"Wrote leaderboard: {config.output}")
    print_top_results(summary.leaderboard_rows, limit=10)


if __name__ == "__main__":
    main()
