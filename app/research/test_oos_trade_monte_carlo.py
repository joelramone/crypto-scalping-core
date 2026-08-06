from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from app.research.monte_carlo.oos_trade_monte_carlo import (
    DistributionSummary,
    distribution,
    longest_streak,
    method_summary,
    sequence_metrics,
    simulate,
    verdict,
)


PNL = [1.0, -2.0, 3.0, -1.0, -1.0]


def test_permutation_preserves_total_and_multiset() -> None:
    rows = simulate(PNL, simulations=5, seed=42)
    permutations = [row for row in rows if row.method == "permutation"]
    assert all(row.final_pnl == pytest.approx(sum(PNL)) for row in permutations)

    rng = np.random.default_rng(42)
    for _ in permutations:
        assert Counter(rng.permutation(PNL)) == Counter(PNL)


def test_bootstrap_has_configured_trade_count_and_seed_is_deterministic() -> None:
    first = simulate(PNL, simulations=3, seed=42)
    second = simulate(PNL, simulations=3, seed=42)
    different = simulate(PNL, simulations=3, seed=43)
    assert first == second
    assert first != different
    assert all(row.losing_trade_pct * len(PNL) / 100 == pytest.approx(round(row.losing_trade_pct * len(PNL) / 100)) for row in first)


def test_drawdown_streak_and_ruin_calculations() -> None:
    values = np.asarray([2.0, -1.0, -4.0, -1.0, 3.0])
    metrics = sequence_metrics(values, starting_capital=10.0, ruin_drawdown_pct=0.40)
    assert metrics["max_drawdown"] == 6.0
    assert metrics["minimum_equity"] == 6.0
    assert metrics["ruined"] is True
    assert longest_streak(values, winning=False) == 3
    assert longest_streak(values, winning=True) == 1


def test_percentiles_come_from_simulation_values() -> None:
    result = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.percentile([1.0, 2.0, 3.0, 4.0, 5.0], [1, 5, 25, 75, 95, 99])
    assert result.percentile_1 == expected[0]
    assert result.percentile_95 == expected[4]


def _summary(positive: float, median: float, ruin: float, dd95: float) -> dict[str, object]:
    template = DistributionSummary(
        mean=median, median=median, standard_deviation=0, minimum=median,
        maximum=median, percentile_1=median, percentile_5=median,
        percentile_25=median, percentile_75=median, percentile_95=median,
        percentile_99=median,
    )
    drawdown = template.model_copy(update={"percentile_95": dd95})
    return {"probability_positive": positive, "probability_ruin": ruin, "final_pnl": template, "max_drawdown": drawdown}


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (_summary(0.49, 1, 0, 10), "REJECT"),
        (_summary(0.60, 1, 0, 10), "FRAGILE_EDGE"),
        (_summary(0.70, 1, 0, 10), "ROBUSTNESS_CANDIDATE"),
    ],
)
def test_verdict_is_deterministic(summary: dict[str, object], expected: str) -> None:
    assert verdict(summary) == expected
    assert verdict(summary) == expected


def test_method_summary_uses_simulation_rows() -> None:
    rows = [row for row in simulate(PNL, simulations=4, seed=7) if row.method == "bootstrap"]
    summary = method_summary(rows)
    final = summary["final_pnl"]
    assert isinstance(final, DistributionSummary)
    assert final.median == np.median([row.final_pnl for row in rows])
