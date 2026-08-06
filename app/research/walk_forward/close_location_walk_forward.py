"""Fixed-parameter Donchian close-location walk-forward validation."""

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
DATA_PATH = Path("data/BTCUSDT_1m.csv")
TIMEFRAME = "15m"
REPORT_PATH = Path(
    "research/reports/donchian_breakout_15m_close_location_walk_forward_v1.md"
)
WINDOW_CSV_PATH = Path(
    "research/walk_forward/donchian_breakout_15m_close_location_walk_forward_v1.csv"
)
TRAIN_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3
BASELINE_FILTER = 0.0
FILTERED_FILTER = 0.94

Variant = Literal["baseline", "filtered"]
Verdict = Literal[
    "reject",
    "promising but unconfirmed",
    "candidate for Monte Carlo validation",
]


class WalkForwardWindow(BaseModel):
    """Calendar boundaries for one fixed-parameter OOS evaluation."""

    window: int = Field(ge=1)
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class OOSMetrics(BaseModel):
    """Required metrics for one variant and evaluation period."""

    trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    expectancy: float
    net_pnl: float
    max_drawdown: float
    average_holding: float
    filtered_trade_retention: float


class WindowResult(BaseModel):
    """Results for one variant in one independent OOS window."""

    boundaries: WalkForwardWindow
    variant: Variant
    min_close_location_filter: float
    metrics: OOSMetrics


class AggregateResult(BaseModel):
    """Metrics recomputed from all concatenated OOS trades for a variant."""

    variant: Variant
    min_close_location_filter: float
    metrics: OOSMetrics


class WalkForwardResult(BaseModel):
    """Complete deterministic fixed-parameter validation result."""

    windows: list[WalkForwardWindow]
    window_results: list[WindowResult]
    aggregates: list[AggregateResult]
    verdict: Verdict


def build_windows(first_timestamp: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[WalkForwardWindow]:
    """Build complete rolling 6m/3m calendar windows with non-overlapping OOS tests."""
    first = pd.Timestamp(first_timestamp)
    train_start = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    windows: list[WalkForwardWindow] = []
    while True:
        train_end = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end_exclusive:
            break
        windows.append(
            WalkForwardWindow(
                window=len(windows) + 1,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
            )
        )
        train_start += pd.DateOffset(months=STEP_MONTHS)
    return windows


def _strategy(threshold: float) -> DonchianBreakoutStrategy:
    return DonchianBreakoutStrategy(
        lookback=3,
        volume_ratio=0.4,
        take_profit_pct=0.012,
        stop_loss_pct=0.008,
        max_holding_candles=24,
        min_quality_score=0,
        min_close_location_filter=threshold,
    )


def _metrics(trades: list[BacktestTrade], baseline_count: int) -> OOSMetrics:
    official = calculate_metrics(trades)
    winning = [trade for trade in trades if trade.net_pnl > 0.0]
    losing = [trade for trade in trades if trade.net_pnl <= 0.0]
    return OOSMetrics(
        trades=official.total_trades,
        wins=official.wins,
        losses=official.losses,
        win_rate=official.win_rate,
        gross_profit=sum(trade.net_pnl for trade in winning),
        gross_loss=abs(sum(trade.net_pnl for trade in losing)),
        profit_factor=official.profit_factor,
        expectancy=official.expectancy,
        net_pnl=official.net_pnl,
        max_drawdown=official.max_drawdown,
        average_holding=(
            sum(trade.holding_candles for trade in trades) / len(trades)
            if trades
            else 0.0
        ),
        filtered_trade_retention=(len(trades) / baseline_count if baseline_count else 0.0),
    )


def filtered_oos_trades(featured_df: pd.DataFrame) -> list[BacktestTrade]:
    """Return the official filtered trades from every complete OOS test window."""
    if featured_df.empty:
        raise ValueError("Walk-forward validation requires featured candles.")
    data = featured_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    interval = data["timestamp"].diff().dropna().median()
    windows = build_windows(data["timestamp"].iloc[0], data["timestamp"].iloc[-1] + interval)
    if not windows:
        raise ValueError("Dataset does not contain one complete 6m train / 3m test window.")

    trades: list[BacktestTrade] = []
    for window in windows:
        timestamps = data["timestamp"]
        test_df = data.loc[
            (timestamps >= pd.Timestamp(window.test_start))
            & (timestamps < pd.Timestamp(window.test_end))
        ].reset_index(drop=True)
        trades.extend(simulate_strategy(test_df, _strategy(FILTERED_FILTER)).trades)
    return trades


def _verdict(
    baseline: OOSMetrics,
    filtered: OOSMetrics,
    window_results: list[WindowResult],
) -> Verdict:
    """Apply the documented deterministic progression criteria."""
    aggregate_improves = (
        filtered.profit_factor > baseline.profit_factor
        and filtered.expectancy > baseline.expectancy
        and filtered.max_drawdown < baseline.max_drawdown
    )
    if not aggregate_improves:
        return "reject"

    filtered_windows = [row for row in window_results if row.variant == "filtered"]
    baseline_by_window = {
        row.boundaries.window: row.metrics
        for row in window_results
        if row.variant == "baseline"
    }
    improving_windows = sum(
        row.metrics.expectancy > baseline_by_window[row.boundaries.window].expectancy
        for row in filtered_windows
    )
    candidate = (
        filtered.profit_factor > 1.0
        and filtered.expectancy > 0.0
        and filtered.net_pnl > 0.0
        and filtered.filtered_trade_retention >= 0.25
        and improving_windows >= (len(filtered_windows) + 1) // 2
    )
    return (
        "candidate for Monte Carlo validation"
        if candidate
        else "promising but unconfirmed"
    )


def run_walk_forward(featured_df: pd.DataFrame) -> WalkForwardResult:
    """Evaluate only the fixed baseline and 0.94 filter in complete OOS windows."""
    if featured_df.empty:
        raise ValueError("Walk-forward validation requires featured candles.")
    data = featured_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    interval = data["timestamp"].diff().dropna().median()
    end_exclusive = data["timestamp"].iloc[-1] + interval
    windows = build_windows(data["timestamp"].iloc[0], end_exclusive)
    if not windows:
        raise ValueError("Dataset does not contain one complete 6m train / 3m test window.")

    window_results: list[WindowResult] = []
    all_trades: dict[Variant, list[BacktestTrade]] = {"baseline": [], "filtered": []}
    thresholds: tuple[tuple[Variant, float], ...] = (
        ("baseline", BASELINE_FILTER),
        ("filtered", FILTERED_FILTER),
    )
    for window in windows:
        timestamps = data["timestamp"]
        test_df = data.loc[
            (timestamps >= pd.Timestamp(window.test_start))
            & (timestamps < pd.Timestamp(window.test_end))
        ].reset_index(drop=True)
        trade_sets = {
            variant: simulate_strategy(test_df, _strategy(threshold)).trades
            for variant, threshold in thresholds
        }
        baseline_count = len(trade_sets["baseline"])
        for variant, threshold in thresholds:
            trades = trade_sets[variant]
            all_trades[variant].extend(trades)
            window_results.append(
                WindowResult(
                    boundaries=window,
                    variant=variant,
                    min_close_location_filter=threshold,
                    metrics=_metrics(trades, baseline_count),
                )
            )

    aggregate_baseline_count = len(all_trades["baseline"])
    aggregates = [
        AggregateResult(
            variant=variant,
            min_close_location_filter=threshold,
            metrics=_metrics(all_trades[variant], aggregate_baseline_count),
        )
        for variant, threshold in thresholds
    ]
    return WalkForwardResult(
        windows=windows,
        window_results=window_results,
        aggregates=aggregates,
        verdict=_verdict(
            aggregates[0].metrics,
            aggregates[1].metrics,
            window_results,
        ),
    )


def _number(value: float) -> str:
    return "inf" if value == inf else f"{value:.6f}"


def _metrics_cells(metrics: OOSMetrics) -> str:
    return (
        f"{metrics.trades} | {metrics.wins} | {metrics.losses} | {metrics.win_rate:.2%} | "
        f"{_number(metrics.gross_profit)} | {_number(metrics.gross_loss)} | "
        f"{_number(metrics.profit_factor)} | {_number(metrics.expectancy)} | "
        f"{_number(metrics.net_pnl)} | {_number(metrics.max_drawdown)} | "
        f"{metrics.average_holding:.2f} | {metrics.filtered_trade_retention:.2%}"
    )


def build_report(result: WalkForwardResult) -> str:
    """Render the fixed setup, boundaries, OOS metrics, aggregation, and verdict."""
    header = (
        "Trades | Wins | Losses | Win rate | Gross profit | Gross loss | PF | "
        "Expectancy | Net PnL | Max DD | Avg holding | Retention"
    )
    rows = []
    for row in result.window_results:
        boundary = row.boundaries
        rows.append(
            f"| {boundary.window} | {boundary.test_start.isoformat()} | "
            f"{boundary.test_end.isoformat()} | {row.variant} | "
            f"{row.min_close_location_filter:.2f} | {_metrics_cells(row.metrics)} |"
        )
    aggregate_rows = [
        f"| {row.variant} | {row.min_close_location_filter:.2f} | "
        f"{_metrics_cells(row.metrics)} |"
        for row in result.aggregates
    ]
    boundary_rows = [
        f"| {window.window} | {window.train_start.isoformat()} | "
        f"{window.train_end.isoformat()} | {window.test_start.isoformat()} | "
        f"{window.test_end.isoformat()} |"
        for window in result.windows
    ]
    return "\n".join(
        [
            "# Donchian Breakout 15m Close-Location Walk-Forward v1",
            "",
            "## Fixed validation setup",
            "",
            f"- Symbol: `{SYMBOL}`",
            f"- Source: `{DATA_PATH}`",
            f"- Timeframe: `{TIMEFRAME}`",
            "- Strategy: lookback=3, volume_ratio=0.4, take_profit_pct=0.012, "
            "stop_loss_pct=0.008, max_holding_candles=24, min_quality_score=0",
            "- Variants: baseline=0.00; filtered=0.94",
            "- Schedule: rolling 6-month train, 3-month test, 3-month step",
            "- No fitting or optimization is performed; train dates define the fixed walk-forward schedule only.",
            "- All intervals are half-open: start inclusive, end exclusive.",
            "",
            "## Window boundaries",
            "",
            "| Window | Train start | Train end | OOS start | OOS end |",
            "|---:|:---|:---|:---|:---|",
            *boundary_rows,
            "",
            "## Independent OOS window results",
            "",
            f"| Window | OOS start | OOS end | Variant | Filter | {header} |",
            "|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Aggregate OOS comparison",
            "",
            "Aggregate metrics are recomputed from concatenated, chronological OOS trades. "
            "Window Profit Factors and drawdowns are not averaged.",
            "",
            f"| Variant | Filter | {header} |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *aggregate_rows,
            "",
            "## Deterministic verdict rules",
            "",
            "- Reject unless filtered aggregate PF and expectancy improve and max drawdown decreases.",
            "- Candidate requires PF > 1, positive expectancy and net PnL, retention >= 25%, "
            "and expectancy improvement in at least half of OOS windows.",
            "- An aggregate improvement that misses candidate requirements is promising but unconfirmed.",
            "",
            "## Verdict",
            "",
            f"**{result.verdict}**",
            "",
        ]
    )


def write_outputs(
    result: WalkForwardResult,
    report_path: Path = REPORT_PATH,
    csv_path: Path = WINDOW_CSV_PATH,
) -> None:
    """Persist the Markdown report and machine-readable window-level metrics."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result), encoding="utf-8")
    records = []
    for row in result.window_results:
        records.append(
            {
                **row.boundaries.model_dump(),
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
    parser.add_argument("--window-csv", type=Path, default=WINDOW_CSV_PATH)
    args = parser.parse_args()
    raw = load_ohlcv_csv(args.data)
    featured = drop_indicator_warmup_rows(compute_features(resample_ohlcv(raw, TIMEFRAME)))
    result = run_walk_forward(featured)
    write_outputs(result, args.report, args.window_csv)
    print(build_report(result))


if __name__ == "__main__":
    main()
