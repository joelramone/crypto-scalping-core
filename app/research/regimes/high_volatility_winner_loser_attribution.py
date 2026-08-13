"""Descriptive winner/loser attribution for the frozen Donchian candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.regimes.analysis import prepare_dataset
from app.research.regimes.high_volatility_attribution import (
    DEFAULT_DATA_2025,
    DEFAULT_DATA_2026,
    FROZEN_CANDIDATE_FILTER,
    build_strategy,
    partition_entry_high_volatility,
)
from app.research.simulation import BacktestTrade, simulate_strategy

REPORT_PATH = Path("research/reports/donchian_high_volatility_winner_loser_attribution_v1.md")
CSV_PATH = Path("research/regimes/donchian_high_volatility_winner_loser_attribution_v1.csv")
PERIODS = ("2025", "2026")
SMALL_GROUP_SIZE = 10

# All columns are existing, causal entry-candle outputs of the official feature pipeline.
FEATURES = (
    "realized_volatility_20",
    "high_volatility_threshold",
    "volatility_threshold_ratio",
    "regime_atr14",
    "atr_pct",
    "atr_expansion_ratio",
    "range_expansion_ratio",
    "close_location_value",
    "body_to_range",
    "breakout_distance_pct",
    "volume_ratio",
    "rsi14",
    "ema20_slope_pct",
    "regime_ema20_slope_pct",
    "regime_ema50_slope_pct",
    "ema_alignment_strength",
    "ema20_ema50_separation",
    "ema50_ema200_separation",
    "adx14",
)

Relationship = Literal["same_direction", "contradictory", "not_comparable"]


class FeatureAttribution(BaseModel):
    period: str
    feature: str
    winner_count: int = Field(ge=0)
    loser_count: int = Field(ge=0)
    winner_mean: float | None
    winner_median: float | None
    loser_mean: float | None
    loser_median: float | None
    difference_winner_minus_loser: float | None
    correlation_with_net_pnl: float | None
    small_sample: bool
    cross_period_relationship: Relationship


def frozen_high_volatility_trades(
    classified_df: pd.DataFrame,
) -> tuple[list[BacktestTrade], pd.Series]:
    """Simulate exactly frozen 0.94 once and retain official HIGH_VOL records."""
    strategy = build_strategy("frozen_0.94_candidate")
    if strategy.min_close_location_filter != FROZEN_CANDIDATE_FILTER:
        raise AssertionError("Only the frozen 0.94 candidate may be attributed")
    trades = simulate_strategy(classified_df, strategy).trades
    high, _ = partition_entry_high_volatility(trades, classified_df)
    if not all(isinstance(item, BacktestTrade) for item in high):
        raise TypeError("Attribution requires official BacktestTrade records")
    return high, strategy.last_breakout_distance_pct


def entry_records(
    classified_df: pd.DataFrame,
    trades: list[BacktestTrade],
    breakout_distance_pct: pd.Series,
    period: str,
) -> pd.DataFrame:
    """Copy existing entry features next to outcomes without altering trades."""
    records: list[dict[str, object]] = []
    for trade in trades:
        entry = classified_df.iloc[trade.entry_index]
        if not bool(entry["is_high_volatility"]):
            raise AssertionError("A non-HIGH_VOLATILITY entry reached attribution")
        record = {feature: entry.get(feature) for feature in FEATURES}
        record["breakout_distance_pct"] = breakout_distance_pct.iloc[trade.entry_index]
        threshold = float(entry["high_volatility_threshold"])
        record["volatility_threshold_ratio"] = (
            float(entry["realized_volatility_20"]) / threshold if threshold else float("nan")
        )
        record.update(period=period, net_pnl=trade.net_pnl, is_winner=trade.net_pnl > 0.0)
        records.append(record)
    return pd.DataFrame(records, columns=("period", "net_pnl", "is_winner", *FEATURES))


def _finite(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def summarize(records: pd.DataFrame, period: str) -> list[FeatureAttribution]:
    winners = records.loc[records["is_winner"]]
    losers = records.loc[~records["is_winner"]]
    rows: list[FeatureAttribution] = []
    for feature in FEATURES:
        win = pd.to_numeric(winners[feature], errors="coerce").dropna()
        loss = pd.to_numeric(losers[feature], errors="coerce").dropna()
        paired = records[[feature, "net_pnl"]].apply(pd.to_numeric, errors="coerce").dropna()
        win_mean, loss_mean = _finite(win.mean()), _finite(loss.mean())
        difference = None if win_mean is None or loss_mean is None else win_mean - loss_mean
        correlation = (
            paired[feature].corr(paired["net_pnl"])
            if len(paired) >= 3 and paired[feature].nunique() > 1 and paired["net_pnl"].nunique() > 1
            else float("nan")
        )
        rows.append(FeatureAttribution(
            period=period, feature=feature, winner_count=len(win), loser_count=len(loss),
            winner_mean=win_mean, winner_median=_finite(win.median()),
            loser_mean=loss_mean, loser_median=_finite(loss.median()),
            difference_winner_minus_loser=difference,
            correlation_with_net_pnl=_finite(correlation),
            small_sample=min(len(win), len(loss)) < SMALL_GROUP_SIZE,
            cross_period_relationship="not_comparable",
        ))
    return rows


def analyze(datasets: dict[str, pd.DataFrame]) -> list[FeatureAttribution]:
    """Analyze periods independently, then pool records without testing cutoffs."""
    if tuple(datasets) != PERIODS:
        raise ValueError("2025 and 2026 must be supplied and independently attributable")
    entries: dict[str, pd.DataFrame] = {}
    for period in PERIODS:
        trades, breakout = frozen_high_volatility_trades(datasets[period])
        entries[period] = entry_records(datasets[period], trades, breakout, period)
    rows = [row for period in PERIODS for row in summarize(entries[period], period)]
    by_key = {(row.period, row.feature): row for row in rows}
    for feature in FEATURES:
        first, second = by_key[("2025", feature)], by_key[("2026", feature)]
        differences = (first.difference_winner_minus_loser, second.difference_winner_minus_loser)
        relationship: Relationship = "not_comparable"
        if all(value is not None and value != 0.0 for value in differences):
            relationship = "same_direction" if differences[0] * differences[1] > 0 else "contradictory"  # type: ignore[operator]
        first.cross_period_relationship = relationship
        second.cross_period_relationship = relationship
    combined = pd.concat([entries[period] for period in PERIODS], ignore_index=True)
    combined_rows = summarize(combined, "combined")
    for row in combined_rows:
        row.cross_period_relationship = by_key[("2025", row.feature)].cross_period_relationship
    return rows + combined_rows


def _value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.8f}"


def _table(rows: list[FeatureAttribution], period: str) -> str:
    lines = [
        "| Feature | W n | L n | W mean | W median | L mean | L median | W-L | Corr(net PnL) | Small sample | Cross-period |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in rows:
        if row.period == period:
            lines.append(f"| {row.feature} | {row.winner_count} | {row.loser_count} | {_value(row.winner_mean)} | {_value(row.winner_median)} | {_value(row.loser_mean)} | {_value(row.loser_median)} | {_value(row.difference_winner_minus_loser)} | {_value(row.correlation_with_net_pnl)} | {'yes' if row.small_sample else 'no'} | {row.cross_period_relationship} |")
    return "\n".join(lines)


def build_report(rows: list[FeatureAttribution]) -> str:
    relationships = {feature: next(row.cross_period_relationship for row in rows if row.period == "combined" and row.feature == feature) for feature in FEATURES}
    same = ", ".join(feature for feature, value in relationships.items() if value == "same_direction") or "None"
    contradictory = ", ".join(feature for feature, value in relationships.items() if value == "contradictory") or "None"
    return f"""# Donchian HIGH_VOLATILITY Winner-vs-Loser Attribution v1

## Scope and frozen methodology
This is descriptive research attribution, not a HIGH_VOLATILITY-only strategy or a production-readiness claim. Each dataset is independently processed by the official feature and regime pipeline. The official simulator is run once per period with the frozen Donchian 15m candidate (`min_close_location_filter=0.94`; all other parameters unchanged). Its original `BacktestTrade` objects are selected only when their entry candle has the existing `is_high_volatility` label. Features are observed at entry. Winners have net PnL > 0; all other completed trades are losers. Difference means winner mean minus loser mean. Pearson correlation is descriptive and omitted when undefined. No feature threshold is derived or tested.

## 2025 results
{_table(rows, '2025')}

## 2026 external OOS results
{_table(rows, '2026')}

## Combined results
{_table(rows, 'combined')}

## Cross-period assessment
Same-direction features: **{same}**.

Contradictory features: **{contradictory}**.

Rows marked small sample have fewer than {SMALL_GROUP_SIZE} valid observations in either outcome group. Those relationships are explicitly treated as sample-dominated, irrespective of their apparent magnitude.

## Limitations
Only 55 HIGH_VOLATILITY trades were reported across the two periods, and splitting winners from losers makes each comparison smaller. The hypothesis and periods are not discovery-independent. Correlation is univariate, sensitive to outliers and feature collinearity, and is not causal. The 2026 dataset is partial through 2026-08-05. Pooling periods can conceal distribution shifts. Multiple descriptive comparisons increase the chance of incidental patterns. No uncertainty interval or multiplicity-adjusted inference is claimed.

## Recommended next research step
Pre-register a replication of these directional hypotheses on untouched later data, retaining the candidate, regime definition, features, simulator, exits, fees, and sizing unchanged. Do not translate this report into thresholds or implement that replication in this change.
"""


def write_outputs(rows: list[FeatureAttribution], report: Path, csv_path: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(rows), encoding="utf-8")
    pd.DataFrame([row.model_dump() for row in rows]).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-2025", type=Path, default=DEFAULT_DATA_2025)
    parser.add_argument("--data-2026", type=Path, default=DEFAULT_DATA_2026)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()
    datasets = {"2025": prepare_dataset(args.data_2025, "15m"), "2026": prepare_dataset(args.data_2026, "15m")}
    write_outputs(analyze(datasets), args.report, args.output_csv)


if __name__ == "__main__":
    main()
