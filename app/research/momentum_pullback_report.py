"""Falsification report for the frozen Momentum Pullback baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from app.research.simulation import BacktestResult, BacktestTrade, calculate_metrics


def _monthly_groups(trades: list[BacktestTrade]) -> dict[str, list[BacktestTrade]]:
    groups: dict[str, list[BacktestTrade]] = {}
    for trade in trades:
        month = pd.Timestamp(trade.exit_timestamp).strftime("%Y-%m")
        groups.setdefault(month, []).append(trade)
    return groups


def determine_baseline_verdict(result: BacktestResult) -> str:
    """Apply exactly the pre-registered sample, performance, and concentration rules."""
    metrics = result.metrics
    if metrics.total_trades < 100:
        return "INSUFFICIENT_SAMPLE"
    if metrics.profit_factor <= 1.0 or metrics.expectancy <= 0 or metrics.net_pnl <= 0:
        return "BASELINE_REJECT"

    positive_monthly_pnl = [
        calculate_metrics(trades).net_pnl
        for trades in _monthly_groups(result.trades).values()
        if calculate_metrics(trades).net_pnl > 0
    ]
    total_positive = sum(positive_monthly_pnl)
    if total_positive > 0 and max(positive_monthly_pnl, default=0.0) / total_positive > 0.8:
        return "BASELINE_REJECT"
    return "BASELINE_CANDIDATE"


def _monthly_rows(trades: list[BacktestTrade]) -> list[str]:
    rows = [
        "| Month | Trades | Profit Factor | Expectancy | Net PnL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for month, month_trades in sorted(_monthly_groups(trades).items()):
        metrics = calculate_metrics(month_trades)
        rows.append(
            f"| {month} | {metrics.total_trades} | {metrics.profit_factor:.4f} | "
            f"{metrics.expectancy:.4f} USDT | {metrics.net_pnl:.4f} USDT |"
        )
    if not trades:
        rows.append("| None | 0 | 0.0000 | 0.0000 USDT | 0.0000 USDT |")
    return rows


def build_momentum_pullback_report(
    *,
    result: BacktestResult,
    total_candles: int,
    feature_rows: int,
    raw_entry_signals: int,
    data_path: str | Path,
    timeframe: str,
    parameters: dict[str, object],
) -> str:
    """Render only the metrics required by the baseline protocol."""
    metrics = result.metrics
    gross_profit = sum(t.net_pnl for t in result.trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in result.trades if t.net_pnl <= 0))
    average_holding = (
        sum(t.holding_candles for t in result.trades) / len(result.trades)
        if result.trades
        else 0.0
    )
    exits = Counter(t.exit_reason for t in result.trades)
    exit_rows = ["| Exit reason | Trades |", "| --- | ---: |"]
    exit_rows.extend(f"| {reason} | {count} |" for reason, count in sorted(exits.items()))
    if not exits:
        exit_rows.append("| None | 0 |")

    return "\n".join(
        [
            "# Momentum Pullback Continuation Baseline v1",
            "",
            "## Execution",
            "",
            "- Strategy: `momentum_pullback_continuation`",
            f"- Dataset: `{data_path}` (2025 discovery data only)",
            f"- Timeframe: `{timeframe}`",
            *[f"- `{key}`: `{value}`" for key, value in parameters.items()],
            "- Parameter combinations executed: 1",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| Total candles | {total_candles} |",
            f"| Feature rows | {feature_rows} |",
            f"| Raw entry signals | {raw_entry_signals} |",
            f"| Completed trades | {metrics.total_trades} |",
            f"| Wins | {metrics.wins} |",
            f"| Losses | {metrics.losses} |",
            f"| Win rate | {metrics.win_rate:.2%} |",
            f"| Gross profit | {gross_profit:.4f} USDT |",
            f"| Gross loss | {gross_loss:.4f} USDT |",
            f"| Fees | {metrics.estimated_fees:.4f} USDT |",
            f"| Profit Factor | {metrics.profit_factor:.4f} |",
            f"| Expectancy | {metrics.expectancy:.4f} USDT |",
            f"| Net PnL | {metrics.net_pnl:.4f} USDT |",
            f"| Max drawdown | {metrics.max_drawdown:.4f} USDT |",
            f"| Average holding candles | {average_holding:.2f} |",
            "",
            "## Exit-reason distribution",
            "",
            *exit_rows,
            "",
            "## Monthly metrics",
            "",
            *_monthly_rows(result.trades),
            "",
            "## Deterministic verdict",
            "",
            f"**{determine_baseline_verdict(result)}**",
            "",
        ]
    )


def write_momentum_pullback_report(report: str, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
