"""Analyze completed baseline leaderboards without rerunning market experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.research.automation.diagnostics import diagnose_leaderboard


def analyze_existing(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Diagnose existing leaderboards only; never execute the optimizer."""
    diagnostics = [diagnose_leaderboard(path) for path in paths]
    if not diagnostics:
        raise ValueError("At least one existing leaderboard is required.")
    return diagnostics


def render_failure_pattern_report(diagnostics: list[dict[str, Any]]) -> str:
    """Render a deterministic cross-family failure-pattern report."""
    counts = Counter(item["failure_mode"] for item in diagnostics)
    lines = [
        "# Existing Baseline Failure Pattern Report",
        "",
        "> Analysis-only report. No optimizer execution, market-data loading, experiment allocation, or Research Memory mutation.",
        "",
        "## Summary",
        "",
        f"- Baselines analyzed: {len(diagnostics)}",
        f"- NO_GROSS_EDGE: {counts['NO_GROSS_EDGE']}",
        f"- FEE_DOMINATED: {counts['FEE_DOMINATED']}",
        f"- POSITIVE_NET_EDGE: {counts['POSITIVE_NET_EDGE']}",
        "",
        "## Cross-Family Comparison",
        "",
        "| Strategy | TF | Trades | Gross PnL | Fees | Net PnL | Gross PF | Net PF | Gross Exp. | Net Exp. | Profitable Months | Failure Mode |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in diagnostics:
        lines.append(
            "| {strategy} | {timeframe} | {trades} | {gross_pnl_before_fees:.4f} | "
            "{total_fees:.4f} | {net_pnl:.4f} | {gross_profit_factor:.4f} | "
            "{net_profit_factor:.4f} | {gross_expectancy:.4f} | {net_expectancy:.4f} | "
            "{profitable_month_percentage:.1%} | {failure_mode} |".format(**item)
        )

    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- `NO_GROSS_EDGE`: gross PnL before fees is zero or negative; transaction costs are not the primary cause of failure.",
        "- `FEE_DOMINATED`: gross PnL is positive but net PnL is zero or negative; costs consume the observed gross edge.",
        "- `POSITIVE_NET_EDGE`: net PnL remains positive after fees; this label is descriptive and does not authorize promotion or tuning.",
        "",
        "## Governance Boundary",
        "",
        "This report is descriptive research triage only. It does not rerun experiments, allocate EXP IDs, modify Research Memory, tune parameters, rescue rejected hypotheses, or authorize production trading.",
        "",
    ])
    return "\n".join(lines)


def write_failure_pattern_report(paths: Iterable[str | Path], output: str | Path) -> Path:
    """Analyze existing leaderboards and persist the report."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_failure_pattern_report(analyze_existing(paths)), encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing baseline leaderboards without rerunning experiments.")
    parser.add_argument("leaderboards", nargs="+", help="Existing leaderboard CSV paths.")
    parser.add_argument("--output", default="research/reports/failure_patterns.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_failure_pattern_report(args.leaderboards, args.output)
    print(f"Wrote failure-pattern report: {output}")


if __name__ == "__main__":
    main()
