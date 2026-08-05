"""Import historical leaderboard CSVs into permanent Research Memory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.discovery.leaderboard_loader import load_leaderboard_csv
from app.research.memory.experiment_store import find_experiment_by_leaderboard_file
from app.research.memory.experiment_writer import write_imported_experiment_memory
from app.research.optimizer.grid_search import (
    GridSearchConfig,
    GridSearchResult,
    GridSearchSummary,
    load_grid_search_config,
)
from app.research.optimizer.leaderboard import LeaderboardRow
from app.research.simulation import BacktestMetrics

LEADERBOARD_PARAM_COLUMNS = {
    "strategy",
    "timeframe",
    "rank",
    "total_trades",
    "wins",
    "losses",
    "win_rate",
    "gross_profit",
    "gross_loss",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "gross_pnl",
    "net_pnl",
    "average_holding_candles",
}


def _display_path(path: Path) -> str:
    """Render a path relative to cwd when possible, else absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse historical leaderboard import arguments."""
    parser = argparse.ArgumentParser(description="Import a historical leaderboard into Research Memory.")
    parser.add_argument("--leaderboard", required=True, help="Path to a leaderboard CSV file.")
    parser.add_argument("--config", required=True, help="Path to the matching optimizer YAML config.")
    return parser.parse_args()


def _validate_file(path_str: str, label: str) -> Path:
    """Ensure an import input file exists."""
    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return path


def _count_parameter_combinations(parameter_grid: dict[str, list[Any]]) -> int:
    """Return the cartesian product size of a parameter grid."""
    counts = [len(values) for values in parameter_grid.values()]
    total = 1
    for count in counts:
        total *= count
    return total


def _extract_parameters(row: pd.Series) -> dict[str, Any]:
    """Extract non-metric leaderboard columns into a parameter dict."""
    parameters: dict[str, Any] = {}
    for column, value in row.items():
        if column in LEADERBOARD_PARAM_COLUMNS:
            continue
        if pd.isna(value) or value == "":
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        parameters[column] = value
    return parameters


def _build_summary_from_leaderboard(df: pd.DataFrame, config: GridSearchConfig) -> GridSearchSummary:
    """Convert a leaderboard CSV into a sortable GridSearchSummary."""
    sorted_df = df.sort_values(
        by=["profit_factor", "expectancy", "max_drawdown", "total_trades"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    ranked_results: list[GridSearchResult] = []
    leaderboard_rows: list[LeaderboardRow] = []
    for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        wins = int(row["wins"]) if "wins" in row.index and pd.notna(row["wins"]) else 0
        losses = int(row["losses"]) if "losses" in row.index and pd.notna(row["losses"]) else 0
        gross_profit = (
            float(row["gross_profit"])
            if "gross_profit" in row.index and pd.notna(row["gross_profit"])
            else 0.0
        )
        gross_loss = (
            float(row["gross_loss"])
            if "gross_loss" in row.index and pd.notna(row["gross_loss"])
            else 0.0
        )
        average_holding_candles = (
            float(row["average_holding_candles"])
            if "average_holding_candles" in row.index
            and pd.notna(row["average_holding_candles"])
            else 0.0
        )
        metrics = BacktestMetrics(
            total_trades=int(row["total_trades"]),
            wins=wins,
            losses=losses,
            win_rate=float(row["win_rate"]),
            gross_pnl=float(row["gross_pnl"]),
            estimated_fees=0.0,
            net_pnl=float(row["net_pnl"]),
            profit_factor=float(row["profit_factor"]),
            expectancy=float(row["expectancy"]),
            average_win=0.0,
            average_loss=0.0,
            max_drawdown=float(row["max_drawdown"]),
        )
        parameters = _extract_parameters(row)
        ranked_results.append(
            GridSearchResult(
                strategy=config.strategy,
                parameters=parameters,
                metrics=metrics,
                average_holding_candles=average_holding_candles,
            )
        )
        leaderboard_rows.append(
            LeaderboardRow(
                strategy=config.strategy,
                timeframe=config.timeframe,
                rank=rank,
                total_trades=int(row["total_trades"]),
                wins=wins,
                losses=losses,
                win_rate=float(row["win_rate"]),
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                profit_factor=float(row["profit_factor"]),
                expectancy=float(row["expectancy"]),
                max_drawdown=float(row["max_drawdown"]),
                gross_pnl=float(row["gross_pnl"]),
                net_pnl=float(row["net_pnl"]),
                average_holding_candles=average_holding_candles,
                parameters=parameters,
            )
        )

    return GridSearchSummary(
        strategy=config.strategy,
        timeframe=config.timeframe,
        evaluated_configurations=_count_parameter_combinations(config.parameters),
        ranked_results=ranked_results,
        leaderboard_rows=leaderboard_rows,
    )


def import_historical_leaderboard(
    leaderboard_path: str | Path,
    config_path: str | Path,
) -> tuple[bool, str]:
    """Import a historical leaderboard into permanent Research Memory."""
    leaderboard = _validate_file(str(leaderboard_path), "Leaderboard file")
    config_file = _validate_file(str(config_path), "Config file")

    existing = find_experiment_by_leaderboard_file(str(leaderboard))
    if existing is not None:
        return (
            False,
            f"Leaderboard already imported for experiment {existing['experiment_id']}: {leaderboard}",
        )

    config = load_grid_search_config(config_file)
    config = config.model_copy(update={"output": leaderboard, "config_file": config_file})
    leaderboard_df = load_leaderboard_csv(leaderboard)
    summary = _build_summary_from_leaderboard(leaderboard_df, config)
    artifacts = write_imported_experiment_memory(config, summary)

    lines = [
        f"Imported experiment ID: {artifacts.experiment_id}",
        f"Leaderboard: {leaderboard}",
        f"Journal: {_display_path(artifacts.journal_path)}",
        f"Memory index: {_display_path(artifacts.index_path)}",
    ]
    return True, "\n".join(lines)


def main() -> None:
    """Run the historical leaderboard import CLI."""
    args = parse_args()
    imported, message = import_historical_leaderboard(args.leaderboard, args.config)
    print(message)
    if imported:
        return


if __name__ == "__main__":
    main()
