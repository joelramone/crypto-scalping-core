"""Frozen 2026 external OOS validation for the validated Donchian candidate."""

from __future__ import annotations

import argparse
from datetime import datetime
from math import inf
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.research.backtester import drop_indicator_warmup_rows, load_ohlcv_csv
from app.research.data_utils import resample_ohlcv
from app.research.features import compute_features
from app.research.simulation import BacktestTrade, calculate_metrics, simulate_strategy
from app.research.strategies import DonchianBreakoutStrategy

SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
DATA_PATH = Path(
    "data/BTCUSDT_1m_2026-01-01_through_2026-08-05_binance_usdm_raw.csv"
)
REPORT_PATH = Path(
    "research/reports/donchian_breakout_15m_external_oos_2026_v1.md"
)
OUTPUT_CSV_PATH = Path(
    "research/validation/donchian_breakout_15m_external_oos_2026_v1.csv"
)
BASELINE_FILTER = 0.0
CANDIDATE_FILTER = 0.94
EXPECTED_FIRST_TIMESTAMP = pd.Timestamp("2026-01-01T00:00:00Z")
EXPECTED_LAST_TIMESTAMP = pd.Timestamp("2026-08-05T23:59:00Z")
EXPECTED_SOURCE_ROWS = 312_480
MATERIAL_DRAWDOWN_MULTIPLIER = 1.25
MATERIAL_UNDERPERFORMANCE_RATIO = 0.80
ADEQUATE_RETENTION = 0.25
CONCENTRATION_LIMIT = 0.80

Variant = Literal["baseline", "candidate"]
Verdict = Literal["REJECT", "FRAGILE_EDGE", "EXTERNAL_OOS_CONFIRMED"]


class ValidationMetrics(BaseModel):
    """Metrics recomputed from official BacktestTrade records."""

    trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_rate: float
    gross_profit: float
    gross_loss: float
    total_fees: float
    profit_factor: float
    expectancy: float
    net_pnl: float
    max_drawdown: float
    average_holding_candles: float
    trade_retention: float


class MetricRow(BaseModel):
    """Aggregate or entry-month result for one frozen variant."""

    period: str
    variant: Variant
    min_close_location_filter: float
    metrics: ValidationMetrics


class ExternalOOSResult(BaseModel):
    """Persistible result of the frozen validation."""

    source_path: str
    source_first_timestamp: datetime
    source_last_timestamp: datetime
    source_rows: int
    aggregate_results: list[MetricRow]
    monthly_results: list[MetricRow]
    verdict: Verdict
    sample_size_classification: str


def build_strategy(threshold: float) -> DonchianBreakoutStrategy:
    """Build one of the two permitted variants with fixed parameters."""
    return DonchianBreakoutStrategy(
        lookback=3,
        volume_ratio=0.4,
        take_profit_pct=0.012,
        stop_loss_pct=0.008,
        max_holding_candles=24,
        min_quality_score=0,
        min_close_location_filter=threshold,
    )


def _metrics(trades: list[BacktestTrade], baseline_count: int) -> ValidationMetrics:
    official = calculate_metrics(trades)
    winning = [trade for trade in trades if trade.net_pnl > 0.0]
    losing = [trade for trade in trades if trade.net_pnl <= 0.0]
    return ValidationMetrics(
        trades=official.total_trades,
        wins=official.wins,
        losses=official.losses,
        win_rate=official.win_rate,
        gross_profit=sum(trade.net_pnl for trade in winning),
        gross_loss=abs(sum(trade.net_pnl for trade in losing)),
        total_fees=sum(trade.fees for trade in trades),
        profit_factor=official.profit_factor,
        expectancy=official.expectancy,
        net_pnl=official.net_pnl,
        max_drawdown=official.max_drawdown,
        average_holding_candles=(
            sum(trade.holding_candles for trade in trades) / len(trades)
            if trades
            else 0.0
        ),
        trade_retention=(len(trades) / baseline_count if baseline_count else 0.0),
    )


def _month_key(timestamp: object) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m")


def _months() -> list[str]:
    return [f"2026-{month:02d}" for month in range(1, 9)]


def _sample_size(trades: int) -> str:
    if trades < 30:
        return "insufficient"
    if trades < 100:
        return "preliminary"
    return "usable but still limited"


def _highly_concentrated(monthly: list[MetricRow]) -> bool:
    candidate_positive = sorted(
        (
            row.metrics.net_pnl
            for row in monthly
            if row.variant == "candidate" and row.metrics.net_pnl > 0.0
        ),
        reverse=True,
    )
    total = sum(candidate_positive)
    return total > 0.0 and sum(candidate_positive[:2]) / total >= CONCENTRATION_LIMIT


def determine_verdict(
    baseline: ValidationMetrics,
    candidate: ValidationMetrics,
    monthly: list[MetricRow],
) -> Verdict:
    """Apply deterministic external OOS verdict rules.

    Material underperformance means both candidate PF and expectancy are below 80%
    of baseline. Drawdown is materially worse above 125% of baseline. Adequate
    retention is at least 25%; concentration means two months supply at least 80%
    of the sum of positive candidate monthly PnL.
    """
    materially_underperforms = (
        candidate.profit_factor < baseline.profit_factor * MATERIAL_UNDERPERFORMANCE_RATIO
        and candidate.expectancy < baseline.expectancy * MATERIAL_UNDERPERFORMANCE_RATIO
    )
    if (
        candidate.profit_factor < 1.0
        or candidate.expectancy <= 0.0
        or candidate.net_pnl <= 0.0
        or materially_underperforms
    ):
        return "REJECT"

    drawdown_worse = candidate.max_drawdown > (
        baseline.max_drawdown * MATERIAL_DRAWDOWN_MULTIPLIER
    )
    improves = (
        candidate.profit_factor > baseline.profit_factor
        or candidate.expectancy > baseline.expectancy
    )
    positive_months = sum(
        row.metrics.net_pnl > 0.0
        for row in monthly
        if row.variant == "candidate"
    )
    confirmed = (
        candidate.trades >= 100
        and improves
        and not drawdown_worse
        and candidate.trade_retention >= ADEQUATE_RETENTION
        and positive_months >= 3
        and not _highly_concentrated(monthly)
    )
    return "EXTERNAL_OOS_CONFIRMED" if confirmed else "FRAGILE_EDGE"


def validate_source(raw: pd.DataFrame) -> pd.DataFrame:
    """Enforce the frozen 2026 source boundary before feature calculation."""
    data = raw.copy()
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    if (
        len(data) != EXPECTED_SOURCE_ROWS
        or timestamps.iloc[0] != EXPECTED_FIRST_TIMESTAMP
        or timestamps.iloc[-1] != EXPECTED_LAST_TIMESTAMP
        or not timestamps.is_monotonic_increasing
        or timestamps.duplicated().any()
        or not timestamps.dt.year.eq(2026).all()
    ):
        raise ValueError("Source must be the exact validated 2026 BTCUSDT dataset.")
    data["timestamp"] = timestamps
    return data


def run_external_oos(featured_df: pd.DataFrame, source_path: str = str(DATA_PATH)) -> ExternalOOSResult:
    """Run exactly baseline and candidate through the official simulator."""
    if featured_df.empty:
        raise ValueError("External OOS validation requires featured 2026 candles.")
    data = featured_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    if not data["timestamp"].dt.year.eq(2026).all():
        raise ValueError("External OOS trades and warm-up must use only 2026 data.")
    data = data.sort_values("timestamp").reset_index(drop=True)

    variants: tuple[tuple[Variant, float], ...] = (
        ("baseline", BASELINE_FILTER),
        ("candidate", CANDIDATE_FILTER),
    )
    entries = {
        variant: build_strategy(threshold).generate_entries(data)
        for variant, threshold in variants
    }
    if (entries["candidate"] & ~entries["baseline"]).any():
        raise AssertionError("Candidate signals must be a subset of baseline signals.")

    trade_sets = {
        variant: simulate_strategy(data, build_strategy(threshold)).trades
        for variant, threshold in variants
    }
    baseline_count = len(trade_sets["baseline"])
    aggregate = [
        MetricRow(
            period="2026-01-01 through 2026-08-05",
            variant=variant,
            min_close_location_filter=threshold,
            metrics=_metrics(trade_sets[variant], baseline_count),
        )
        for variant, threshold in variants
    ]
    monthly: list[MetricRow] = []
    for month in _months():
        month_sets = {
            variant: [trade for trade in trade_sets[variant] if _month_key(trade.entry_timestamp) == month]
            for variant, _ in variants
        }
        month_baseline_count = len(month_sets["baseline"])
        for variant, threshold in variants:
            monthly.append(
                MetricRow(
                    period=month,
                    variant=variant,
                    min_close_location_filter=threshold,
                    metrics=_metrics(month_sets[variant], month_baseline_count),
                )
            )
    baseline_metrics = aggregate[0].metrics
    candidate_metrics = aggregate[1].metrics
    return ExternalOOSResult(
        source_path=source_path,
        source_first_timestamp=EXPECTED_FIRST_TIMESTAMP.to_pydatetime(),
        source_last_timestamp=EXPECTED_LAST_TIMESTAMP.to_pydatetime(),
        source_rows=EXPECTED_SOURCE_ROWS,
        aggregate_results=aggregate,
        monthly_results=monthly,
        verdict=determine_verdict(baseline_metrics, candidate_metrics, monthly),
        sample_size_classification=_sample_size(candidate_metrics.trades),
    )


def _number(value: float) -> str:
    return "inf" if value == inf else f"{value:.6f}"


def _cells(metrics: ValidationMetrics, monthly: bool = False) -> str:
    core = (
        f"{metrics.trades} | {metrics.win_rate:.2%} | {_number(metrics.profit_factor)} | "
        f"{_number(metrics.expectancy)} | {_number(metrics.net_pnl)} | "
        f"{_number(metrics.max_drawdown)}"
    )
    if monthly:
        return core
    return (
        f"{metrics.trades} | {metrics.wins} | {metrics.losses} | {metrics.win_rate:.2%} | "
        f"{_number(metrics.gross_profit)} | {_number(metrics.gross_loss)} | "
        f"{_number(metrics.total_fees)} | {_number(metrics.profit_factor)} | "
        f"{_number(metrics.expectancy)} | {_number(metrics.net_pnl)} | "
        f"{_number(metrics.max_drawdown)} | {metrics.average_holding_candles:.2f} | "
        f"{metrics.trade_retention:.2%}"
    )


def build_report(result: ExternalOOSResult) -> str:
    aggregate_rows = [
        f"| {row.variant} | {row.min_close_location_filter:.2f} | {_cells(row.metrics)} |"
        for row in result.aggregate_results
    ]
    monthly_rows = [
        f"| {row.period} | {row.variant} | {row.min_close_location_filter:.2f} | "
        f"{_cells(row.metrics, monthly=True)} |"
        for row in result.monthly_results
    ]
    baseline = result.aggregate_results[0].metrics
    candidate = result.aggregate_results[1].metrics
    comparison = [
        f"- PF > 1: **{'yes' if candidate.profit_factor > 1 else 'no'}**",
        f"- Expectancy > 0: **{'yes' if candidate.expectancy > 0 else 'no'}**",
        f"- Net PnL > 0: **{'yes' if candidate.net_pnl > 0 else 'no'}**",
        "- Improves PF or expectancy versus 2026 baseline: "
        f"**{'yes' if candidate.profit_factor > baseline.profit_factor or candidate.expectancy > baseline.expectancy else 'no'}**",
        "- Drawdown not materially worse (<=125% of baseline): "
        f"**{'yes' if candidate.max_drawdown <= baseline.max_drawdown * MATERIAL_DRAWDOWN_MULTIPLIER else 'no'}**",
        f"- Trade retention >=25%: **{'yes' if candidate.trade_retention >= ADEQUATE_RETENTION else 'no'}**",
    ]
    return "\n".join(
        [
            "# Donchian Breakout 15m External OOS 2026 v1",
            "",
            "## Frozen setup and data boundary",
            "",
            f"- Symbol: `{SYMBOL}`; timeframe: `{TIMEFRAME}`.",
            f"- Source: `{result.source_path}` only ({result.source_rows} one-minute rows).",
            f"- Source interval: `{result.source_first_timestamp.isoformat()}` through `{result.source_last_timestamp.isoformat()}`.",
            "- Fixed strategy: lookback=3, volume_ratio=0.4, take_profit_pct=0.012, stop_loss_pct=0.008, max_holding_candles=24, min_quality_score=0.",
            "- Variants: baseline filter=0.00 and candidate filter=0.94. No optimization or alternate threshold was run.",
            "- Warm-up: the official feature pipeline runs on the full 2026 source after official 15m resampling; rows with incomplete indicators are then dropped. Thus only earlier 2026 candles warm features, and no 2025 candle or trade is used.",
            "- Trades are produced by the official `simulate_strategy` simulator as `BacktestTrade` records; all metrics below are recomputed from those records.",
            "- Monthly partitions use the UTC month of each trade's opening timestamp. August ends with the source on 2026-08-05.",
            "",
            "## Aggregate 2026 results",
            "",
            "| Variant | Filter | Trades | Wins | Losses | Win rate | Gross profit | Gross loss | Fees | PF | Expectancy | Net PnL | Max DD | Avg holding candles | Retention |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *aggregate_rows,
            "",
            "## Monthly results (UTC entry month)",
            "",
            "| Month | Variant | Filter | Trades | Win rate | PF | Expectancy | Net PnL | Max DD |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
            *monthly_rows,
            "",
            "## Prior 2025 walk-forward OOS reference (not combined)",
            "",
            "| Trades | PF | Expectancy | Net PnL | Max DD |",
            "|---:|---:|---:|---:|---:|",
            "| 112 | 1.080648 | 0.027449 | 3.074296 | 6.548168 |",
            "",
            "The 2025 values are a reference only; every 2026 value uses exclusively 2026 trade records.",
            "",
            "## Deterministic assessment",
            "",
            *comparison,
            f"- Candidate sample size: **{result.sample_size_classification}** ({candidate.trades} trades).",
            "- Material thresholds: underperformance requires both PF and expectancy below 80% of baseline; drawdown worsens materially above 125% of baseline; high concentration means the top two positive months contribute at least 80% of positive monthly PnL.",
            "",
            "## Verdict",
            "",
            f"**{result.verdict}**",
            "",
            "This external validation does not authorize real-money trading.",
            "",
        ]
    )


def write_outputs(result: ExternalOOSResult, report_path: Path, csv_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result), encoding="utf-8")
    records = []
    for scope, rows in (("aggregate", result.aggregate_results), ("monthly", result.monthly_results)):
        for row in rows:
            records.append(
                {
                    "scope": scope,
                    "period": row.period,
                    "variant": row.variant,
                    "min_close_location_filter": row.min_close_location_filter,
                    **row.metrics.model_dump(),
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV_PATH)
    args = parser.parse_args()
    raw = validate_source(load_ohlcv_csv(args.data))
    featured = drop_indicator_warmup_rows(
        compute_features(resample_ohlcv(raw, TIMEFRAME))
    )
    result = run_external_oos(featured, str(args.data))
    write_outputs(result, args.report, args.output_csv)
    print(build_report(result))


if __name__ == "__main__":
    main()
