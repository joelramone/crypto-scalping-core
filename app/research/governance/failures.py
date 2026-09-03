"""Failure classifications derived from completed trade diagnostics."""

from app.research.analysis.trade_diagnostics import TradeDiagnostics


def classify_baseline_failure(diagnostics: TradeDiagnostics) -> str | None:
    """Classify the supported baseline failure from permanent economics."""
    if (
        diagnostics.gross_expectancy > 0.0
        and diagnostics.fee_expectancy >= diagnostics.gross_expectancy
        and diagnostics.net_expectancy <= 0.0
    ):
        return "FEE_DOMINATED"
    return None
