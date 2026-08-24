"""Leaderboard helpers for research optimizer results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.research.simulation import BacktestMetrics
from app.research.analysis.trade_diagnostics import TradeDiagnostics

DIAGNOSTIC_COLUMNS = [
    "gross_pnl_before_fees", "total_fees", "net_expectancy", "gross_expectancy",
    "fee_expectancy", "gross_profit_before_fees", "gross_loss_before_fees",
    "gross_profit_factor", "net_profit_factor", "flats", "average_winner",
    "median_winner", "average_loser", "median_loser", "payoff_ratio",
    "break_even_win_rate", "actual_minus_break_even_win_rate", "median_holding",
    "holding_p25", "holding_p75", "holding_p95", "raw_entry_signals",
    "completed_trades", "suppressed_signals", "suppression_rate",
    "raw_signals_per_opened_trade", "positive_months", "negative_months",
    "profitable_month_percentage", "best_month", "worst_month",
    "positive_pnl_concentration_top_2_months",
] + [
    f"{reason}_{suffix}"
    for reason in ("take_profit", "stop_loss", "max_holding", "strategy_exit")
    for suffix in ("exits", "exit_percentage", "net_pnl")
] + ["monthly_diagnostics"]

LEADERBOARD_COLUMNS = [
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
    *DIAGNOSTIC_COLUMNS,
    "lookback",
    "rsi_threshold",
    "distance_from_ema20",
    "volume_ratio",
    "bb_std_multiplier",
    "take_profit_pct",
    "stop_loss_pct",
    "max_holding_candles",
    "min_quality_score",
    "min_body_to_range",
    "min_close_location",
    "min_range_expansion",
    "min_atr_expansion",
    "min_ema20_slope_pct",
    "min_ema_alignment_strength",
    "min_breakout_distance_pct",
    "min_close_location_filter",
]


class LeaderboardRow(BaseModel):
    """A ranked optimizer result row ready for CSV export."""

    strategy: str
    timeframe: str
    rank: int = Field(ge=1)
    total_trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    gross_pnl: float
    net_pnl: float
    average_holding_candles: float
    parameters: dict[str, Any]
    diagnostics: TradeDiagnostics | None = None

    def to_csv_row(self) -> dict[str, Any]:
        """Return a flat CSV row using the permanent leaderboard columns."""
        row: dict[str, Any] = {
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "rank": self.rank,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "average_holding_candles": self.average_holding_candles,
        }
        for column in LEADERBOARD_COLUMNS:
            if column not in row:
                row[column] = self.parameters.get(column, "")
        if self.diagnostics is not None:
            diagnostic = self.diagnostics
            for column in DIAGNOSTIC_COLUMNS:
                if hasattr(diagnostic, column):
                    value = getattr(diagnostic, column)
                    row[column] = "" if value is None else value
            for reason, exit_diagnostic in diagnostic.exits.items():
                row[f"{reason}_exits"] = exit_diagnostic.count
                row[f"{reason}_exit_percentage"] = exit_diagnostic.percentage
                row[f"{reason}_net_pnl"] = exit_diagnostic.net_pnl
            row["monthly_diagnostics"] = "[" + ",".join(
                item.model_dump_json() for item in diagnostic.monthly
            ) + "]"
        return row


def build_leaderboard_rows(
    strategy_name: str,
    timeframe: str,
    ranked_results: list[tuple[dict[str, Any], BacktestMetrics]],
    average_holding_candles: list[float] | None = None,
    diagnostics: list[TradeDiagnostics] | None = None,
) -> list[LeaderboardRow]:
    """Convert ranked optimizer metrics into leaderboard rows."""
    rows: list[LeaderboardRow] = []
    holdings = average_holding_candles or [0.0] * len(ranked_results)
    result_diagnostics = diagnostics or [None] * len(ranked_results)
    for index, ((parameters, metrics), average_holding, diagnostic) in enumerate(
        zip(ranked_results, holdings, result_diagnostics, strict=True),
        start=1,
    ):
        rows.append(
            LeaderboardRow(
                strategy=strategy_name,
                timeframe=timeframe,
                rank=index,
                total_trades=metrics.total_trades,
                wins=metrics.wins,
                losses=metrics.losses,
                win_rate=metrics.win_rate,
                gross_profit=metrics.average_win * metrics.wins,
                gross_loss=abs(metrics.average_loss * metrics.losses),
                profit_factor=metrics.profit_factor,
                expectancy=metrics.expectancy,
                max_drawdown=metrics.max_drawdown,
                gross_pnl=metrics.gross_pnl,
                net_pnl=metrics.net_pnl,
                average_holding_candles=average_holding,
                parameters=parameters,
                diagnostics=diagnostic,
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
            f"  #{row.rank} timeframe={row.timeframe} trades={row.total_trades} "
            f"win_rate={row.win_rate:.2%} "
            f"profit_factor={row.profit_factor:.4f} "
            f"expectancy={row.expectancy:.4f} "
            f"net_pnl={row.net_pnl:.4f} "
            f"params: {params}"
        )
