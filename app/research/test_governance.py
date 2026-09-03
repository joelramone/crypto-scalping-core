"""Focused tests for deterministic experiment governance."""

import pytest

from app.research.analysis.trade_diagnostics import TradeDiagnostics
from app.research.governance.failures import classify_baseline_failure
from app.research.governance.verdicts import determine_baseline_verdict


def _diagnostics(**updates: float | int) -> TradeDiagnostics:
    values: dict[str, float | int] = {
        "completed_trades": 100,
        "gross_expectancy": 0.20,
        "fee_expectancy": 0.05,
        "net_expectancy": 0.15,
        "net_profit_factor": 1.20,
        "net_pnl": 15.0,
        "positive_pnl_concentration_top_2_months": 0.50,
    }
    values.update(updates)
    return TradeDiagnostics.model_construct(**values)


def test_baseline_verdict_is_insufficient_below_one_hundred_trades():
    assert determine_baseline_verdict(_diagnostics(completed_trades=99)) == (
        "INSUFFICIENT_SAMPLE"
    )


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"gross_expectancy": 0.0}, "BASELINE_REJECT"),
        ({"net_profit_factor": 1.0}, "BASELINE_REJECT"),
        (
            {"positive_pnl_concentration_top_2_months": 0.800001},
            "BASELINE_REJECT",
        ),
        ({}, "BASELINE_CANDIDATE"),
    ],
)
def test_baseline_verdict_gates(
    updates: dict[str, float], expected: str
):
    assert determine_baseline_verdict(_diagnostics(**updates)) == expected


def test_fee_dominated_uses_per_trade_economics():
    diagnostics = _diagnostics(
        gross_expectancy=0.069345,
        fee_expectancy=0.079972,
        net_expectancy=-0.010627,
    )

    assert classify_baseline_failure(diagnostics) == "FEE_DOMINATED"


def test_positive_gross_and_net_is_not_fee_dominated():
    assert classify_baseline_failure(_diagnostics()) is None
