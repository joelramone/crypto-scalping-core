"""Markdown reporting for the frozen Volatility Exhaustion baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from app.research.simulation import BacktestResult, BacktestTrade, calculate_metrics


def determine_baseline_verdict(result: BacktestResult) -> str:
    """Apply the pre-registered baseline stop rules to an official result."""
    metrics = result.metrics
    if metrics.total_trades < 100:
        return "INSUFFICIENT_SAMPLE"
    if (
        metrics.profit_factor <= 1.0
        or metrics.expectancy <= 0.0
        or metrics.net_pnl <= 0.0
    ):
        return "BASELINE_REJECT"
    return "BASELINE_CANDIDATE"


def _monthly_rows(trades: list[BacktestTrade]) -> list[str]:
    if not trades:
        return ["No completed trades."]

    grouped: dict[str, list[BacktestTrade]] = {}
    for trade in trades:
        period = pd.Timestamp(trade.exit_timestamp).strftime("%Y-%m")
        grouped.setdefault(period, []).append(trade)

    rows = [
        "| Month | Trades | Win rate | Profit Factor | Expectancy | Net PnL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for period, period_trades in sorted(grouped.items()):
        metrics = calculate_metrics(period_trades)
        rows.append(
            f"| {period} | {metrics.total_trades} | {metrics.win_rate:.2%} | "
            f"{metrics.profit_factor:.4f} | {metrics.expectancy:.4f} USDT | "
            f"{metrics.net_pnl:.4f} USDT |"
        )
    return rows


def build_volatility_exhaustion_report(
    *,
    result: BacktestResult,
    total_candles: int,
    feature_rows: int,
    data_path: str | Path,
    timeframe: str,
    parameters: dict[str, object],
) -> str:
    """Render the baseline report solely from execution inputs and trade records."""
    metrics = result.metrics
    gross_profit = sum(trade.net_pnl for trade in result.trades if trade.net_pnl > 0.0)
    gross_loss = abs(sum(trade.net_pnl for trade in result.trades if trade.net_pnl <= 0.0))
    average_holding = (
        sum(trade.holding_candles for trade in result.trades) / len(result.trades)
        if result.trades
        else 0.0
    )
    exit_counts = Counter(trade.exit_reason for trade in result.trades)
    exit_rows = ["| Exit reason | Trades |", "| --- | ---: |"]
    exit_rows.extend(
        f"| {reason} | {count} |" for reason, count in sorted(exit_counts.items())
    )
    if not exit_counts:
        exit_rows.append("| None | 0 |")

    parameter_rows = [f"- `{name}`: `{value}`" for name, value in parameters.items()]
    verdict = determine_baseline_verdict(result)
    lines = [
        "# Volatility Exhaustion Baseline v1",
        "",
        "## Execution",
        "",
        "- Strategy: `volatility_exhaustion`",
        f"- Dataset: `{data_path}`",
        f"- Timeframe: `{timeframe}`",
        *parameter_rows,
        "- Configurations: exactly one; no optimization",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Total candles | {total_candles} |",
        f"| Feature rows after warm-up | {feature_rows} |",
        f"| Total trades | {metrics.total_trades} |",
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
        "## Pre-registered verdict",
        "",
        f"**{verdict}**",
        "",
        "Verdict rules: fewer than 100 trades is `INSUFFICIENT_SAMPLE`; with at least "
        "100 trades, Profit Factor at most 1, expectancy at most 0, or net PnL at "
        "most 0 is `BASELINE_REJECT`; otherwise the result is `BASELINE_CANDIDATE`.",
        "",
    ]
    return "\n".join(lines)


def write_volatility_exhaustion_report(report: str, output_path: str | Path) -> None:
    """Persist a rendered baseline report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
