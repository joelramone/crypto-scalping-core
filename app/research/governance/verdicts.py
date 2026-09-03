"""Reusable deterministic verdict policies for completed experiments."""

from app.research.analysis.trade_diagnostics import TradeDiagnostics


def determine_baseline_verdict(diagnostics: TradeDiagnostics) -> str:
    """Apply the authoritative preregistered baseline gates in order."""
    if diagnostics.completed_trades < 100:
        return "INSUFFICIENT_SAMPLE"
    if (
        diagnostics.gross_expectancy <= 0.0
        or diagnostics.net_profit_factor <= 1.0
        or diagnostics.net_expectancy <= 0.0
        or diagnostics.net_pnl <= 0.0
        or diagnostics.positive_pnl_concentration_top_2_months > 0.80
    ):
        return "BASELINE_REJECT"
    return "BASELINE_CANDIDATE"
