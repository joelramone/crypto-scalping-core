"""Leaderboard helpers for research optimizer results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.research.simulation import BacktestMetrics

LEADERBOARD_COLUMNS = [
    "strategy",
    "rank",
    "total_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "gross_pnl",
    "net_pnl",
    "rsi_threshold",
    "distance_from_ema20",
    "volume_ratio",
    "take_profit_pct",
    "stop_loss_pct",
    "max_holding_candles",
]


class LeaderboardRow(BaseModel):
    """A ranked optimizer result row ready for CSV export."""

    strategy: str
    rank: int = Field(ge=1)
    total_trades: int = Field(ge=0)
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    gross_pnl: float
    net_pnl: float
    parameters: dict[str, Any]

    def to_csv_row(self) -> dict[str, Any]:
        """Return a flat CSV row using the permanent leaderboard columns."""
        row: dict[str, Any] = {
            "strategy": self.strategy,
            "rank": self.rank,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
        }
        for column in LEADERBOARD_COLUMNS:
            if column not in row:
                row[column] = self.parameters.get(column, "")
        return row


def build_leaderboard_rows(
    strategy_name: str,
    ranked_results: list[tuple[dict[str, Any], BacktestMetrics]],
) -> list[LeaderboardRow]:
    """Convert ranked optimizer metrics into leaderboard rows."""
    rows: list[LeaderboardRow] = []
    for index, (parameters, metrics) in enumerate(ranked_results, start=1):
        rows.append(
            LeaderboardRow(
                strategy=strategy_name,
                rank=index,
                total_trades=metrics.total_trades,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                expectancy=metrics.expectancy,
                max_drawdown=metrics.max_drawdown,
                gross_pnl=metrics.gross_pnl,
                net_pnl=metrics.net_pnl,
                parameters=parameters,
            )
        )
    return rows


def write_leaderboard_csv(rows: list[LeaderboardRow], output_path: str | Path) -> None:
    """Write optimizer leaderboard rows to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def print_top_results(rows: list[LeaderboardRow], limit: int = 10) -> None:
    """Print the highest-ranked optimizer rows to the terminal."""
    print(f"Top {min(limit, len(rows))} optimizer results:")
    if not rows:
        print("  No configurations passed the optimizer filters.")
        return

    for row in rows[:limit]:
        params = ", ".join(
            f"{key}={row.parameters[key]}" for key in LEADERBOARD_COLUMNS if key in row.parameters
        )
        print(
            f"  #{row.rank} trades={row.total_trades} "
            f"win_rate={row.win_rate:.2%} "
            f"profit_factor={row.profit_factor:.4f} "
            f"expectancy={row.expectancy:.4f} "
            f"net_pnl={row.net_pnl:.4f} "
            f"params: {params}"
        )
