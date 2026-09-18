"""Markdown reporting for research throughput automation."""

from __future__ import annotations

from typing import Any


def render_report(diagnostics: dict[str, Any]) -> str:
    """Render a compact human-review report without making trading decisions."""
    return "\n".join(
        [
            "# Research Throughput Report",
            "",
            f"- Strategy: {diagnostics['strategy']}",
            f"- Timeframe: {diagnostics['timeframe']}",
            f"- Trades: {diagnostics['trades']}",
            f"- Gross PnL before fees: {diagnostics['gross_pnl_before_fees']:.6f}",
            f"- Fees: {diagnostics['total_fees']:.6f}",
            f"- Net PnL: {diagnostics['net_pnl']:.6f}",
            f"- Gross PF: {diagnostics['gross_profit_factor']:.6f}",
            f"- Net PF: {diagnostics['net_profit_factor']:.6f}",
            f"- Gross expectancy: {diagnostics['gross_expectancy']:.6f}",
            f"- Net expectancy: {diagnostics['net_expectancy']:.6f}",
            f"- Suppression rate: {diagnostics['suppression_rate']:.2%}",
            f"- Profitable months: {diagnostics['profitable_month_percentage']:.2%}",
            f"- Diagnostic class: {diagnostics['failure_mode']}",
            "",
        ]
    )
