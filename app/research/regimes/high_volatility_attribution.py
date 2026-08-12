"""Fixed-rule Donchian HIGH_VOLATILITY trade attribution experiment."""

from __future__ import annotations

import argparse
from decimal import Decimal
from math import inf
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.regimes.analysis import prepare_dataset
from app.research.regimes.classifier import RegimeConfig
from app.research.simulation import BacktestTrade, calculate_metrics, simulate_strategy
from app.research.strategies import DonchianBreakoutStrategy

REPORT_PATH = Path("research/reports/donchian_high_volatility_attribution_v1.md")
CSV_PATH = Path("research/regimes/donchian_high_volatility_attribution_v1.csv")
DEFAULT_DATA_2025 = Path("data/BTCUSDT_1m.csv")
DEFAULT_DATA_2026 = Path("data/BTCUSDT_1m_2026-01-01_through_2026-08-05_binance_usdm_raw.csv")
FROZEN_CANDIDATE_FILTER = 0.94
MEANINGFUL_REMAINING_TRADES = 30

Group = Literal["ALL", "HIGH_VOLATILITY", "NON_HIGH_VOLATILITY"]
Variant = Literal["phase1_default", "frozen_0.94_candidate"]
Verdict = Literal[
    "CONSISTENT_AVOIDANCE_HYPOTHESIS", "MIXED_EVIDENCE", "NO_AVOIDANCE_EVIDENCE"
]


class AttributionMetrics(BaseModel):
    trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_rate: float
    gross_profit: float
    gross_loss: float
    fees: float
    profit_factor: float
    expectancy: float
    net_pnl: float
    max_drawdown: float
    average_holding_candles: float


class AttributionRow(BaseModel):
    period: str
    variant: Variant
    min_close_location_filter: float
    group: Group
    metrics: AttributionMetrics
    high_volatility_trade_percentage: float
    high_volatility_loss_percentage: float
    high_volatility_negative_pnl_percentage: float | None
    delta_profit_factor_all_vs_non_high: float
    delta_expectancy_all_vs_non_high: float
    delta_max_drawdown_all_vs_non_high: float


def build_strategy(variant: Variant) -> DonchianBreakoutStrategy:
    """Return only a Phase 1 default or the previously frozen candidate."""
    if variant == "phase1_default":
        return DonchianBreakoutStrategy()
    return DonchianBreakoutStrategy(
        lookback=3, volume_ratio=0.4, take_profit_pct=0.012,
        stop_loss_pct=0.008, max_holding_candles=24, min_quality_score=0,
        min_close_location_filter=FROZEN_CANDIDATE_FILTER,
    )


def partition_entry_high_volatility(
    trades: list[BacktestTrade], classified_df: pd.DataFrame
) -> tuple[list[BacktestTrade], list[BacktestTrade]]:
    """Partition the original official records solely by their entry-candle overlay."""
    high = [trade for trade in trades if bool(classified_df.iloc[trade.entry_index]["is_high_volatility"])]
    non_high = [trade for trade in trades if not bool(classified_df.iloc[trade.entry_index]["is_high_volatility"])]
    if len(high) + len(non_high) != len(trades):
        raise AssertionError("HIGH_VOLATILITY partitions do not reconcile")
    if {id(trade) for trade in high}.intersection(id(trade) for trade in non_high):
        raise AssertionError("A trade exists in both volatility partitions")
    if high + non_high and not all(isinstance(trade, BacktestTrade) for trade in high + non_high):
        raise TypeError("Attribution requires official BacktestTrade records")
    return high, non_high


def _decimal_net(trades: list[BacktestTrade]) -> Decimal:
    return sum((Decimal(str(trade.net_pnl)) for trade in trades), Decimal(0))


def _metrics(trades: list[BacktestTrade]) -> AttributionMetrics:
    official = calculate_metrics(trades)
    winners = [trade for trade in trades if trade.net_pnl > 0.0]
    losers = [trade for trade in trades if trade.net_pnl <= 0.0]
    return AttributionMetrics(
        trades=official.total_trades, wins=official.wins, losses=official.losses,
        win_rate=official.win_rate,
        gross_profit=sum(trade.net_pnl for trade in winners),
        gross_loss=abs(sum(trade.net_pnl for trade in losers)),
        fees=sum(trade.fees for trade in trades), profit_factor=official.profit_factor,
        expectancy=official.expectancy, net_pnl=official.net_pnl,
        max_drawdown=official.max_drawdown,
        average_holding_candles=(
            sum(trade.holding_candles for trade in trades) / len(trades) if trades else 0.0
        ),
    )


def attribute(
    classified_df: pd.DataFrame, period: str, variant: Variant
) -> list[AttributionRow]:
    """Run once, then describe entry-time partitions of official trade records."""
    threshold = 0.0 if variant == "phase1_default" else FROZEN_CANDIDATE_FILTER
    all_trades = simulate_strategy(classified_df, build_strategy(variant)).trades
    high, non_high = partition_entry_high_volatility(all_trades, classified_df)
    if _decimal_net(all_trades) != _decimal_net(high) + _decimal_net(non_high):
        raise AssertionError("Partition net PnL does not reconcile exactly")
    groups: dict[Group, list[BacktestTrade]] = {
        "ALL": all_trades, "HIGH_VOLATILITY": high, "NON_HIGH_VOLATILITY": non_high,
    }
    metrics = {group: _metrics(records) for group, records in groups.items()}
    all_metrics, high_metrics, non_metrics = (
        metrics["ALL"], metrics["HIGH_VOLATILITY"], metrics["NON_HIGH_VOLATILITY"]
    )
    trade_share = len(high) / len(all_trades) if all_trades else 0.0
    loss_share = high_metrics.losses / all_metrics.losses if all_metrics.losses else 0.0
    negative_share = (
        high_metrics.gross_loss / all_metrics.gross_loss if all_metrics.gross_loss else None
    )
    return [
        AttributionRow(
            period=period, variant=variant, min_close_location_filter=threshold,
            group=group, metrics=group_metrics,
            high_volatility_trade_percentage=trade_share,
            high_volatility_loss_percentage=loss_share,
            high_volatility_negative_pnl_percentage=negative_share,
            delta_profit_factor_all_vs_non_high=non_metrics.profit_factor - all_metrics.profit_factor,
            delta_expectancy_all_vs_non_high=non_metrics.expectancy - all_metrics.expectancy,
            delta_max_drawdown_all_vs_non_high=non_metrics.max_drawdown - all_metrics.max_drawdown,
        )
        for group, group_metrics in metrics.items()
    ]


def determine_verdict(rows: list[AttributionRow], variant: Variant) -> Verdict:
    all_rows = {(row.period, row.group): row for row in rows if row.variant == variant}
    comparisons = []
    for period in ("2025", "2026"):
        all_metric = all_rows[(period, "ALL")].metrics
        non_metric = all_rows[(period, "NON_HIGH_VOLATILITY")].metrics
        comparisons.append((
            non_metric.profit_factor > all_metric.profit_factor,
            non_metric.expectancy > all_metric.expectancy,
            non_metric.trades >= MEANINGFUL_REMAINING_TRADES,
        ))
    if all(all(item) for item in comparisons):
        return "CONSISTENT_AVOIDANCE_HYPOTHESIS"
    if not any(pf or expectancy for pf, expectancy, _ in comparisons):
        return "NO_AVOIDANCE_EVIDENCE"
    return "MIXED_EVIDENCE"


def _number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return "inf" if value == inf else f"{value:.6f}"


def _table(rows: list[AttributionRow], period: str, variant: Variant) -> str:
    lines = [
        "| Group | Trades | Wins | Losses | Win rate | Gross profit | Gross loss | Fees | PF | Expectancy | Net PnL | Max DD | Avg hold |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.period != period or row.variant != variant:
            continue
        m = row.metrics
        lines.append(f"| {row.group} | {m.trades} | {m.wins} | {m.losses} | {m.win_rate:.2%} | {_number(m.gross_profit)} | {_number(m.gross_loss)} | {_number(m.fees)} | {_number(m.profit_factor)} | {_number(m.expectancy)} | {_number(m.net_pnl)} | {_number(m.max_drawdown)} | {m.average_holding_candles:.2f} |")
    return "\n".join(lines)


def build_report(rows: list[AttributionRow], config: RegimeConfig) -> str:
    verdicts = {variant: determine_verdict(rows, variant) for variant in ("phase1_default", "frozen_0.94_candidate")}
    summaries = {(r.period, r.variant): r for r in rows if r.group == "ALL"}
    def comparisons(variant: Variant) -> str:
        result = []
        for period in ("2025", "2026"):
            r = summaries[(period, variant)]
            result.append(f"- {period}: HIGH_VOLATILITY trades {r.high_volatility_trade_percentage:.2%}; losses {r.high_volatility_loss_percentage:.2%}; negative PnL {(_number(r.high_volatility_negative_pnl_percentage))}; delta PF {r.delta_profit_factor_all_vs_non_high:+.6f}; delta expectancy {r.delta_expectancy_all_vs_non_high:+.6f}; delta max DD {r.delta_max_drawdown_all_vs_non_high:+.6f}.")
        return "\n".join(result)
    return f"""# Donchian HIGH_VOLATILITY Attribution v1

## 1. Objective
Determine whether Donchian trades entered during the existing HIGH_VOLATILITY state consistently degrade economics relative to trades entered outside it.

## 2. Frozen methodology
Each period is independently prepared by the Phase 1 pipeline. The overlay remains realized-volatility-20 above its shifted trailing {config.high_volatility_percentile:.0%} percentile over {config.volatility_lookback} candles after {config.volatility_min_history} observations. One official simulation produces `BacktestTrade` records; partitions select those same objects using only `is_high_volatility` at `entry_index`. No entry timing, outcome, fee, exit, sizing, feature, strategy parameter, or threshold is changed. Percentages are fractions (loss concentration is losing-trade count; negative-PnL concentration is the HIGH_VOLATILITY share of total absolute losing net PnL). Deltas are NON_HIGH_VOLATILITY minus ALL.

## 3. 2025 results
{_table(rows, '2025', 'phase1_default')}

## 4. 2026 results
{_table(rows, '2026', 'phase1_default')}

## 5. Frozen 0.94 candidate attribution
### 2025
{_table(rows, '2025', 'frozen_0.94_candidate')}

### 2026
{_table(rows, '2026', 'frozen_0.94_candidate')}

## 6. Cross-period comparison
Phase 1 default:
{comparisons('phase1_default')}

Frozen 0.94 candidate:
{comparisons('frozen_0.94_candidate')}

## 7. Loss concentration analysis
The comparison above reports both the share of losing trades and the share of absolute losing net PnL attributable to HIGH_VOLATILITY. The latter is N/A only when ALL has no negative PnL.

## 8. Evidence for/against HIGH_VOLATILITY avoidance
Default: **{verdicts['phase1_default']}**. Frozen candidate: **{verdicts['frozen_0.94_candidate']}**. This is descriptive attribution and is not approval for a production filter.

## 9. Limitations
The same historical periods motivated this fixed attribution; HIGH_VOLATILITY is an observational entry label, not a causal intervention. Subsequence drawdown is path-dependent and does not represent a re-simulated filtered strategy. The 2026 period is partial. A remaining sample of 30 trades is the pre-registered descriptive minimum, not proof of statistical power.

## 10. Deterministic verdict
**{verdicts['phase1_default']}** for the Phase 1 default and **{verdicts['frozen_0.94_candidate']}** for the frozen candidate, under the stated rule.

## 11. Recommended next experiment
If and only if the hypothesis is consistent, pre-register the unchanged avoidance rule and evaluate it prospectively on untouched later data or another instrument through the official simulator. Do not tune the volatility definition on these periods.
"""


def write_outputs(rows: list[AttributionRow], report: Path, csv_path: Path) -> None:
    records = []
    for row in rows:
        data = row.model_dump(exclude={"metrics"})
        data.update(row.metrics.model_dump())
        records.append(data)
    report.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(rows, RegimeConfig()), encoding="utf-8")
    pd.DataFrame(records).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-2025", type=Path, default=DEFAULT_DATA_2025)
    parser.add_argument("--data-2026", type=Path, default=DEFAULT_DATA_2026)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()
    datasets = {"2025": prepare_dataset(args.data_2025, "15m"), "2026": prepare_dataset(args.data_2026, "15m")}
    rows = [row for period, frame in datasets.items() for variant in ("phase1_default", "frozen_0.94_candidate") for row in attribute(frame, period, variant)]
    write_outputs(rows, args.report, args.output_csv)


if __name__ == "__main__":
    main()
