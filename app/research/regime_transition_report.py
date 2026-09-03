"""Deterministic report for HYP-REGIME-TRANSITION-001."""

from pathlib import Path
from typing import Any

from app.research.analysis.trade_diagnostics import TradeDiagnostics
from app.research.governance.verdicts import determine_baseline_verdict
from app.research.simulation import BacktestResult


determine_verdict = determine_baseline_verdict


def build_report(
    result: BacktestResult,
    diagnostics: TradeDiagnostics,
    total_candles: int,
    featured_rows: int,
    parameters: dict[str, Any],
) -> str:
    """Render every permanent diagnostic without recalculating it."""
    lines = [
        "# HYP-REGIME-TRANSITION-001 Baseline",
        "",
        "## Data boundaries",
        "",
        "- 2025 = DISCOVERY_USED",
        "- 2026-01-01 through 2026-08-05 = NOT USED",
        "- post-2026-08-05 = RESERVED / NOT ACCESSED",
        "",
        "## Execution",
        "",
        f"- Candles: {total_candles}",
        f"- Featured rows: {featured_rows}",
        "- Parameter combinations: 1",
        f"- Parameters: `{parameters}`",
        f"- Raw signals: {diagnostics.raw_entry_signals}",
        f"- Completed trades: {diagnostics.completed_trades}",
        f"- Max drawdown: {result.metrics.max_drawdown:.10f}",
        "",
        "## Permanent TradeDiagnostics",
        "",
        "```json",
        diagnostics.model_dump_json(indent=2),
        "```",
        "",
        "## Deterministic verdict",
        "",
        f"**{determine_verdict(diagnostics)}**",
        "",
        "No optimization was performed. No regime threshold was changed.",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: str, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
