"""Monte Carlo sequence-risk validation for fixed filtered OOS trades."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.research.backtester import drop_indicator_warmup_rows, load_ohlcv_csv
from app.research.data_utils import resample_ohlcv
from app.research.features import compute_features
from app.research.simulation import BacktestTrade, calculate_metrics
from app.research.walk_forward.close_location_walk_forward import (
    TIMEFRAME,
    filtered_oos_trades,
)

EXPECTED_COUNT = 112
EXPECTED_TOTAL_PNL = 3.074296
EXPECTED_MAX_DRAWDOWN = 6.548168
EXPECTED_WINS = 51
EXPECTED_LOSSES = 61
DEFAULT_REPORT = Path("research/reports/donchian_breakout_15m_close_location_mc_v1.md")
DEFAULT_CSV = Path("research/monte_carlo/donchian_breakout_15m_close_location_mc_v1.csv")
Method = Literal["permutation", "bootstrap"]


class SourceMetrics(BaseModel):
    trade_count: int
    total_net_pnl: float
    expectancy: float
    observed_max_drawdown: float
    wins: int
    losses: int


class SimulationRow(BaseModel):
    method: Method
    simulation_id: int = Field(ge=1)
    seed: int
    final_pnl: float
    max_drawdown: float
    longest_losing_streak: int
    longest_winning_streak: int
    minimum_equity: float
    maximum_equity: float
    losing_trade_pct: float
    profitable: bool
    ruined: bool


class DistributionSummary(BaseModel):
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    percentile_1: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float


def reconcile_trades(trades: list[BacktestTrade]) -> SourceMetrics:
    """Reconcile raw official records, failing rather than substituting summaries."""
    official = calculate_metrics(trades)
    source = SourceMetrics(
        trade_count=official.total_trades,
        total_net_pnl=official.net_pnl,
        expectancy=official.expectancy,
        observed_max_drawdown=official.max_drawdown,
        wins=official.wins,
        losses=official.losses,
    )
    failures = []
    if source.trade_count != EXPECTED_COUNT:
        failures.append(f"trades {source.trade_count} != {EXPECTED_COUNT}")
    if not np.isclose(source.total_net_pnl, EXPECTED_TOTAL_PNL, atol=5e-7, rtol=0):
        failures.append(f"net PnL {source.total_net_pnl:.12f} != {EXPECTED_TOTAL_PNL:.6f}")
    if not np.isclose(source.observed_max_drawdown, EXPECTED_MAX_DRAWDOWN, atol=5e-7, rtol=0):
        failures.append(f"max drawdown {source.observed_max_drawdown:.12f} != {EXPECTED_MAX_DRAWDOWN:.6f}")
    if (source.wins, source.losses) != (EXPECTED_WINS, EXPECTED_LOSSES):
        failures.append(
            f"wins/losses {source.wins}/{source.losses} != {EXPECTED_WINS}/{EXPECTED_LOSSES}"
        )
    if failures:
        raise ValueError("Filtered OOS trade reconciliation failed: " + "; ".join(failures))
    return source


def longest_streak(values: np.ndarray, winning: bool) -> int:
    condition = values > 0 if winning else values <= 0
    longest = current = 0
    for matched in condition:
        current = current + 1 if matched else 0
        longest = max(longest, current)
    return longest


def sequence_metrics(
    values: np.ndarray,
    starting_capital: float,
    ruin_drawdown_pct: float,
) -> dict[str, float | int | bool]:
    """Calculate path metrics using initial capital as the first equity peak."""
    equity = starting_capital + np.cumsum(values)
    path = np.concatenate(([starting_capital], equity))
    peaks = np.maximum.accumulate(path)
    max_drawdown = float(np.max(peaks - path))
    ruin_level = starting_capital * (1.0 - ruin_drawdown_pct)
    final_pnl = float(np.sum(values))
    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "longest_losing_streak": longest_streak(values, winning=False),
        "longest_winning_streak": longest_streak(values, winning=True),
        "minimum_equity": float(np.min(path)),
        "maximum_equity": float(np.max(path)),
        "losing_trade_pct": float(np.mean(values <= 0) * 100.0),
        "profitable": final_pnl > 0.0,
        "ruined": bool(np.any(path <= ruin_level)),
    }


def simulate(
    pnl: list[float],
    simulations: int = 10_000,
    seed: int = 42,
    starting_capital: float = 100.0,
    ruin_drawdown_pct: float = 0.20,
) -> list[SimulationRow]:
    if not pnl:
        raise ValueError("Monte Carlo requires at least one OOS trade.")
    if simulations < 1:
        raise ValueError("simulations must be positive.")
    if starting_capital <= 0 or not 0 < ruin_drawdown_pct <= 1:
        raise ValueError("capital must be positive and ruin drawdown must be in (0, 1].")
    observed = np.asarray(pnl, dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[SimulationRow] = []
    for method in ("permutation", "bootstrap"):
        for simulation_id in range(1, simulations + 1):
            sample = (
                rng.permutation(observed)
                if method == "permutation"
                else rng.choice(observed, size=len(observed), replace=True)
            )
            metrics = sequence_metrics(sample, starting_capital, ruin_drawdown_pct)
            if method == "permutation":
                # Use the canonical observed sum so every permutation is bit-identical.
                metrics["final_pnl"] = float(np.sum(observed))
                metrics["profitable"] = metrics["final_pnl"] > 0.0
            rows.append(
                SimulationRow(
                    method=method,
                    simulation_id=simulation_id,
                    seed=seed,
                    **metrics,
                )
            )
    return rows


def distribution(values: list[float]) -> DistributionSummary:
    array = np.asarray(values, dtype=float)
    percentiles = np.percentile(array, [1, 5, 25, 75, 95, 99])
    return DistributionSummary(
        mean=float(np.mean(array)), median=float(np.median(array)),
        standard_deviation=float(np.std(array)), minimum=float(np.min(array)),
        maximum=float(np.max(array)), percentile_1=float(percentiles[0]),
        percentile_5=float(percentiles[1]), percentile_25=float(percentiles[2]),
        percentile_75=float(percentiles[3]), percentile_95=float(percentiles[4]),
        percentile_99=float(percentiles[5]),
    )


def method_summary(rows: list[SimulationRow]) -> dict[str, object]:
    final = np.asarray([row.final_pnl for row in rows])
    drawdown = np.asarray([row.max_drawdown for row in rows])
    losing = [float(row.longest_losing_streak) for row in rows]
    worst_final_cutoff = np.percentile(final, 5)
    worst_drawdown_cutoff = np.percentile(drawdown, 95)
    return {
        "final_pnl": distribution(final.tolist()),
        "max_drawdown": distribution(drawdown.tolist()),
        "longest_losing_streak": distribution(losing),
        "probability_positive": float(np.mean(final > 0)),
        "probability_non_positive": float(np.mean(final <= 0)),
        "probability_dd_gt_5": float(np.mean(drawdown > 5)),
        "probability_dd_gt_10": float(np.mean(drawdown > 10)),
        "probability_dd_gt_20": float(np.mean(drawdown > 20)),
        "probability_ruin": float(np.mean([row.ruined for row in rows])),
        "expected_shortfall_final_worst_5pct": float(np.mean(final[final <= worst_final_cutoff])),
        "expected_shortfall_drawdown_worst_5pct": float(np.mean(drawdown[drawdown >= worst_drawdown_cutoff])),
    }


def verdict(bootstrap: dict[str, object]) -> str:
    positive = float(bootstrap["probability_positive"])
    ruin = float(bootstrap["probability_ruin"])
    final = bootstrap["final_pnl"]
    drawdown = bootstrap["max_drawdown"]
    assert isinstance(final, DistributionSummary) and isinstance(drawdown, DistributionSummary)
    if positive < 0.50 or final.median <= 0 or ruin >= 0.05 or drawdown.percentile_95 > 20:
        return "REJECT"
    if positive >= 0.65 and ruin < 0.01:
        return "ROBUSTNESS_CANDIDATE"
    return "FRAGILE_EDGE"


def _stats_table(summary: dict[str, object]) -> list[str]:
    lines = ["| Metric | Mean | Median | Std | Min | Max | P1 | P5 | P25 | P75 | P95 | P99 |", "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("final_pnl", "max_drawdown", "longest_losing_streak"):
        value = summary[key]
        assert isinstance(value, DistributionSummary)
        lines.append("| " + key.replace("_", " ") + " | " + " | ".join(f"{number:.6f}" for number in value.model_dump().values()) + " |")
    return lines


def build_report(source: SourceMetrics, rows: list[SimulationRow], starting_capital: float, ruin_pct: float) -> str:
    grouped = {method: [row for row in rows if row.method == method] for method in ("permutation", "bootstrap")}
    summaries = {method: method_summary(values) for method, values in grouped.items()}
    decision = verdict(summaries["bootstrap"])
    sections = [
        "# Donchian Breakout 15m Close-Location Monte Carlo v1", "", "## 1. Objective", "",
        "Measure sequence risk and sampling robustness of the fixed candidate; this is validation only.", "",
        "## 2. Exact source trades", "", "Official `BacktestTrade` records from the two filtered (`0.94`) walk-forward OOS test windows only.", "",
        "## 3. Reconciliation", "", f"- Source trade count: {source.trade_count}", f"- Source total net PnL: {source.total_net_pnl:.6f} USDT", f"- Source expectancy: {source.expectancy:.6f} USDT", f"- Source observed max drawdown: {source.observed_max_drawdown:.6f} USDT", f"- Source wins: {source.wins}", f"- Source losses: {source.losses}", "",
        "## 4. Simulation methods", "", "Permutation shuffles the complete observed sequence. Bootstrap samples the same trade count with replacement.", "",
        "## 5. Assumptions", "", f"Starting capital is {starting_capital:.2f} USDT. Ruin is equity <= {starting_capital * (1-ruin_pct):.2f} USDT. Historical net PnL is not resized.", "",
    ]
    for number, method in ((6, "permutation"), (7, "bootstrap")):
        summary = summaries[method]
        sections += [f"## {number}. {method.title()} results", "", *_stats_table(summary), "", f"- P(final PnL > 0): {float(summary['probability_positive']):.2%}", f"- P(final PnL <= 0): {float(summary['probability_non_positive']):.2%}", ""]
    bootstrap = summaries["bootstrap"]
    sections += [
        "## 8. Drawdown distribution", "", f"Bootstrap P(DD > 5/10/20 USDT): {float(bootstrap['probability_dd_gt_5']):.2%} / {float(bootstrap['probability_dd_gt_10']):.2%} / {float(bootstrap['probability_dd_gt_20']):.2%}.", "",
        "## 9. Losing-streak distribution", "", "The tables report the full longest-losing-streak distribution for each method.", "",
        "## 10. Tail-risk analysis", "", f"Bootstrap ruin probability: {float(bootstrap['probability_ruin']):.2%}. Worst-5% final-PnL expected shortfall: {float(bootstrap['expected_shortfall_final_worst_5pct']):.6f} USDT. Worst-5% drawdown expected shortfall: {float(bootstrap['expected_shortfall_drawdown_worst_5pct']):.6f} USDT.", "",
        "## 11. Sample-size warning", "", "Only 112 observed OOS trades underpin resampling; estimates, especially tail estimates, remain uncertain.", "",
        "## 12. Deterministic verdict", "", f"**{decision}**", "", "Even ROBUSTNESS_CANDIDATE is not approval for real-money trading.", "",
        "## 13. Recommended next validation", "", "Collect additional untouched OOS trades and repeat the same fixed validation without parameter changes.", "",
    ]
    return "\n".join(sections)


def write_csv(rows: list[SimulationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SimulationRow.model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/BTCUSDT_1m.csv"))
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--starting-capital", type=float, default=100.0)
    parser.add_argument("--ruin-drawdown-pct", type=float, default=0.20)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    featured = drop_indicator_warmup_rows(compute_features(resample_ohlcv(load_ohlcv_csv(args.data), TIMEFRAME)))
    trades = filtered_oos_trades(featured)
    source = reconcile_trades(trades)
    print(source.model_dump_json(indent=2))
    rows = simulate([trade.net_pnl for trade in trades], args.simulations, args.seed, args.starting_capital, args.ruin_drawdown_pct)
    write_csv(rows, args.output_csv)
    report = build_report(source, rows, args.starting_capital, args.ruin_drawdown_pct)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
