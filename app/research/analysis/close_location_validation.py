"""Focused Donchian close-location validation report."""

from __future__ import annotations

from math import inf
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.simulation import BacktestTrade, calculate_metrics, simulate_strategy
from app.research.strategies import DonchianBreakoutStrategy

QUARTERS = ("2025 Q1", "2025 Q2", "2025 Q3", "2025 Q4")
BASELINE_EXPECTED = {
    "trades": 559,
    "profit_factor": 0.871868236257362,
    "expectancy": -0.04832520016209745,
    "net_pnl": -27.013786890612476,
}


def _single_parameters(parameter_grid: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        name: values[0]
        for name, values in parameter_grid.items()
        if name != "min_close_location_filter"
    }


def _overall_row(
    threshold: float,
    trades: list[BacktestTrade],
    baseline_trades: int,
) -> dict[str, float | int]:
    metrics = calculate_metrics(trades)
    winning = [trade for trade in trades if trade.net_pnl > 0.0]
    losing = [trade for trade in trades if trade.net_pnl <= 0.0]
    return {
        "threshold": threshold,
        "trades": metrics.total_trades,
        "winners": metrics.wins,
        "losers": metrics.losses,
        "win_rate": metrics.win_rate,
        "gross_profit": sum(trade.net_pnl for trade in winning),
        "gross_loss": abs(sum(trade.net_pnl for trade in losing)),
        "profit_factor": metrics.profit_factor,
        "expectancy": metrics.expectancy,
        "net_pnl": metrics.net_pnl,
        "max_drawdown": metrics.max_drawdown,
        "average_holding": (
            sum(trade.holding_candles for trade in trades) / len(trades)
            if trades
            else 0.0
        ),
        "retention": len(trades) / baseline_trades if baseline_trades else 0.0,
    }


def _quarter_rows(
    threshold: float,
    trades: list[BacktestTrade],
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[BacktestTrade]] = {quarter: [] for quarter in QUARTERS}
    for trade in trades:
        timestamp = pd.Timestamp(trade.entry_timestamp)
        quarter = f"{timestamp.year} Q{timestamp.quarter}"
        if quarter in grouped:
            grouped[quarter].append(trade)

    rows = []
    for quarter in QUARTERS:
        metrics = calculate_metrics(grouped[quarter])
        rows.append(
            {
                "threshold": threshold,
                "quarter": quarter,
                "trades": metrics.total_trades,
                "profit_factor": metrics.profit_factor,
                "expectancy": metrics.expectancy,
                "net_pnl": metrics.net_pnl,
            }
        )
    return rows


def run_close_location_validation(
    df: pd.DataFrame,
    parameter_grid: dict[str, list[Any]],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]]]:
    """Run only the configured close-location thresholds through the official simulator."""
    fixed = _single_parameters(parameter_grid)
    thresholds = parameter_grid["min_close_location_filter"]
    trade_sets = {
        float(threshold): simulate_strategy(
            df,
            DonchianBreakoutStrategy(
                **fixed,
                min_close_location_filter=float(threshold),
            ),
        ).trades
        for threshold in thresholds
    }
    baseline_threshold = float(thresholds[0])
    baseline_count = len(trade_sets[baseline_threshold])
    overall = [
        _overall_row(threshold, trades, baseline_count)
        for threshold, trades in trade_sets.items()
    ]
    quarterly = [
        row
        for threshold, trades in trade_sets.items()
        for row in _quarter_rows(threshold, trades)
    ]
    return overall, quarterly


def _format_number(value: float | int, digits: int = 6) -> str:
    if value == inf:
        return "inf"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _overall_table(rows: list[dict[str, float | int]]) -> list[str]:
    lines = [
        "| Filter | Trades | Retention | Winners | Losers | Win rate | Gross profit | Gross loss | PF | Expectancy | Net PnL | Max DD | Avg holding |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['trades']} | {row['retention']:.1%} "
            f"| {row['winners']} | {row['losers']} | {row['win_rate']:.2%} "
            f"| {_format_number(row['gross_profit'])} | {_format_number(row['gross_loss'])} "
            f"| {_format_number(row['profit_factor'])} | {_format_number(row['expectancy'])} "
            f"| {_format_number(row['net_pnl'])} | {_format_number(row['max_drawdown'])} "
            f"| {row['average_holding']:.2f} |"
        )
    return lines


def _quarter_table(rows: list[dict[str, float | int | str]]) -> list[str]:
    lines = [
        "| Filter | Quarter | Trades | PF | Expectancy | Net PnL | Warning |",
        "|---:|:---|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        warning = "<20 trades" if int(row["trades"]) < 20 else ""
        lines.append(
            f"| {float(row['threshold']):.2f} | {row['quarter']} | {row['trades']} "
            f"| {_format_number(float(row['profit_factor']))} "
            f"| {_format_number(float(row['expectancy']))} "
            f"| {_format_number(float(row['net_pnl']))} | {warning} |"
        )
    return lines


def build_close_location_validation_report(
    overall: list[dict[str, float | int]],
    quarterly: list[dict[str, float | int | str]],
) -> str:
    """Render reconciliation, comparisons, warnings, stability, and verdict."""
    baseline = overall[0]
    baseline_ok = (
        baseline["trades"] == BASELINE_EXPECTED["trades"]
        and abs(float(baseline["profit_factor"]) - BASELINE_EXPECTED["profit_factor"]) < 1e-12
        and abs(float(baseline["expectancy"]) - BASELINE_EXPECTED["expectancy"]) < 1e-12
        and abs(float(baseline["net_pnl"]) - BASELINE_EXPECTED["net_pnl"]) < 1e-12
    )
    baseline_quarters = {
        str(row["quarter"]): row
        for row in quarterly
        if float(row["threshold"]) == float(baseline["threshold"])
    }
    consistency: list[str] = []
    improving_rows: list[dict[str, float | int]] = []
    for row in overall[1:]:
        threshold = float(row["threshold"])
        filtered_quarters = {
            str(item["quarter"]): item
            for item in quarterly
            if float(item["threshold"]) == threshold
        }
        improved = sum(
            float(filtered_quarters[q]["expectancy"])
            > float(baseline_quarters[q]["expectancy"])
            for q in QUARTERS
        )
        consistency.append(
            f"- `{threshold:.2f}` improves quarterly expectancy in {improved}/4 quarters."
        )
        if (
            float(row["profit_factor"]) > float(baseline["profit_factor"])
            and float(row["expectancy"]) > float(baseline["expectancy"])
        ):
            improving_rows.append(row)

    low_total = [row for row in overall if int(row["trades"]) < 100]
    low_quarter = [row for row in quarterly if int(row["trades"]) < 20]
    if not improving_rows:
        verdict = "reject"
    else:
        best = max(improving_rows, key=lambda row: float(row["profit_factor"]))
        best_threshold = float(best["threshold"])
        best_quarters = [
            row for row in quarterly if float(row["threshold"]) == best_threshold
        ]
        improved_quarters = sum(
            float(row["expectancy"])
            > float(baseline_quarters[str(row["quarter"])]["expectancy"])
            for row in best_quarters
        )
        if (
            int(best["trades"]) >= 100
            and all(int(row["trades"]) >= 20 for row in best_quarters)
            and improved_quarters >= 3
        ):
            verdict = "candidate for walk-forward validation"
        else:
            verdict = "promising but unstable"

    warning_lines = [
        *[
            f"- Filter `{float(row['threshold']):.2f}` has fewer than 100 total trades ({row['trades']})."
            for row in low_total
        ],
        *[
            f"- Filter `{float(row['threshold']):.2f}`, {row['quarter']} has fewer than 20 trades ({row['trades']})."
            for row in low_quarter
        ],
    ]
    if not warning_lines:
        warning_lines = ["- No configured sample-size warning was triggered."]

    lines = [
        "# Donchian Breakout 15m Close-Location Validation v1",
        "",
        "## Baseline reconciliation",
        "",
        f"- Status: **{'PASS' if baseline_ok else 'FAIL'}**",
        f"- Trades: {baseline['trades']} (expected {BASELINE_EXPECTED['trades']})",
        f"- Profit Factor: {_format_number(float(baseline['profit_factor']), 12)}",
        f"- Expectancy: {_format_number(float(baseline['expectancy']), 12)} USDT",
        f"- Net PnL: {_format_number(float(baseline['net_pnl']), 12)} USDT",
        "",
        "## Overall comparison",
        "",
        *_overall_table(overall),
        "",
        "## Quarterly comparison",
        "",
        *_quarter_table(quarterly),
        "",
        "## Quarterly consistency",
        "",
        *consistency,
        "",
        "## Sample-size warnings",
        "",
        *warning_lines,
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    return "\n".join(lines)


def write_close_location_validation_report(
    df: pd.DataFrame,
    parameter_grid: dict[str, list[Any]],
    output_path: str | Path,
) -> Path:
    """Run the focused validation and persist its Markdown report."""
    overall, quarterly = run_close_location_validation(df, parameter_grid)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_close_location_validation_report(overall, quarterly),
        encoding="utf-8",
    )
    return path
