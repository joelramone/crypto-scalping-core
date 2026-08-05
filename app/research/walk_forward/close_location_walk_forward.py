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
STRATEGY = "donchian_breakout"
DEFAULT_DATA_PATH = Path("data/BTCUSDT_1m.csv")
DEFAULT_REPORT_PATH = Path("research/reports/donchian_breakout_15m_close_location_walk_forward_v1.md")
DEFAULT_CSV_PATH = Path("research/walk_forward/donchian_breakout_15m_close_location_walk_forward_v1.csv")
BASELINE_FILTER = 0.0
FILTERED_FILTER = 0.94
FIXED_PARAMETERS: dict[str, int | float] = {
    "lookback": 3,
    "volume_ratio": 0.4,
    "take_profit_pct": 0.012,
    "stop_loss_pct": 0.008,
    "max_holding_candles": 24,
    "min_quality_score": 0,
}

Variant = Literal["baseline", "filtered"]
Verdict = Literal["REJECT", "PROMISING_BUT_UNCONFIRMED", "CANDIDATE_FOR_MONTE_CARLO"]


class WalkForwardWindow(BaseModel):
    window: int = Field(ge=1)
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class OOSMetrics(BaseModel):
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
    filtered_trade_retention: float


class WindowResult(BaseModel):
    boundaries: WalkForwardWindow
    variant: Variant
    min_close_location_filter: float
    metrics: OOSMetrics


class AggregateResult(BaseModel):
    variant: Variant
    min_close_location_filter: float
    metrics: OOSMetrics


class WalkForwardResult(BaseModel):
    windows: list[WalkForwardWindow]
    window_results: list[WindowResult]
    aggregates: list[AggregateResult]
    verdict: Verdict


def build_windows(
    first_timestamp: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
) -> list[WalkForwardWindow]:
    """Generate complete, month-aligned, half-open calendar windows."""
    if min(train_months, test_months, step_months) <= 0:
        raise ValueError("Window month counts must be positive.")
    train_start = pd.Timestamp(first_timestamp).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    limit = pd.Timestamp(end_exclusive)
    windows: list[WalkForwardWindow] = []
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > limit:
            break
        windows.append(WalkForwardWindow(
            window=len(windows) + 1,
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            test_start=train_end.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
        ))
        train_start += pd.DateOffset(months=step_months)
    return windows


class _TestIntervalStrategy(DonchianBreakoutStrategy):
    """Use the official strategy logic while restricting entry eligibility to OOS."""

    def __init__(self, threshold: float, test_start: datetime, test_end: datetime) -> None:
        super().__init__(**FIXED_PARAMETERS, min_close_location_filter=threshold)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        if self.test_start.tzinfo is None:
            self.test_start = self.test_start.tz_localize("UTC")
        if self.test_end.tzinfo is None:
            self.test_end = self.test_end.tz_localize("UTC")

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        entries = super().generate_entries(df)
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        return entries & timestamps.ge(self.test_start) & timestamps.lt(self.test_end)


def _metrics(trades: list[BacktestTrade], baseline_count: int, *, baseline: bool = False) -> OOSMetrics:
    official = calculate_metrics(trades)
    winners = [trade for trade in trades if trade.net_pnl > 0]
    losers = [trade for trade in trades if trade.net_pnl <= 0]
    return OOSMetrics(
        trades=official.total_trades,
        wins=official.wins,
        losses=official.losses,
        win_rate=official.win_rate,
        gross_profit=sum(trade.net_pnl for trade in winners),
        gross_loss=abs(sum(trade.net_pnl for trade in losers)),
        fees=official.estimated_fees,
        profit_factor=official.profit_factor,
        expectancy=official.expectancy,
        net_pnl=official.net_pnl,
        max_drawdown=official.max_drawdown,
        average_holding_candles=(sum(t.holding_candles for t in trades) / len(trades) if trades else 0.0),
        filtered_trade_retention=(1.0 if baseline and baseline_count else len(trades) / baseline_count if baseline_count else 0.0),
    )


def determine_verdict(baseline: OOSMetrics, filtered: OOSMetrics, rows: list[WindowResult]) -> Verdict:
    baseline_by_window = {r.boundaries.window: r.metrics for r in rows if r.variant == "baseline"}
    filtered_rows = [r for r in rows if r.variant == "filtered"]
    expectancy_improvements = sum(r.metrics.expectancy > baseline_by_window[r.boundaries.window].expectancy for r in filtered_rows)
    improves_most = expectancy_improvements > len(filtered_rows) / 2
    materially_worse_drawdown = filtered.max_drawdown > baseline.max_drawdown * 1.10
    if not improves_most or materially_worse_drawdown:
        return "REJECT"
    robustness_improves = filtered.profit_factor > baseline.profit_factor and filtered.expectancy > baseline.expectancy
    severe_sample_warning = filtered.trades < 30
    if (
        robustness_improves
        and filtered.profit_factor > 1
        and filtered.expectancy > 0
        and filtered.net_pnl > 0
        and len(filtered_rows) > 2
        and not severe_sample_warning
    ):
        return "CANDIDATE_FOR_MONTE_CARLO"
    return "PROMISING_BUT_UNCONFIRMED"


def run_walk_forward(
    featured_df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 3,
    step_months: int = 3,
) -> WalkForwardResult:
    """Run both fixed variants through the official simulator in each OOS window."""
    if featured_df.empty:
        raise ValueError("Walk-forward validation requires featured candles.")
    data = featured_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    interval = data["timestamp"].diff().dropna().median()
    windows = build_windows(data["timestamp"].iloc[0], data["timestamp"].iloc[-1] + interval, train_months, test_months, step_months)
    if not windows:
        raise ValueError("Dataset does not contain one complete train/test window.")

    rows: list[WindowResult] = []
    aggregate_trades: dict[Variant, list[BacktestTrade]] = {"baseline": [], "filtered": []}
    variants: tuple[tuple[Variant, float], ...] = (("baseline", BASELINE_FILTER), ("filtered", FILTERED_FILTER))
    for window in windows:
        # Include all available history so feature and strategy rolling calculations
        # warm up naturally. Entry masking makes the simulation strictly OOS.
        simulation_df = data.loc[data["timestamp"] < pd.Timestamp(window.test_end)].reset_index(drop=True)
        trade_sets: dict[Variant, list[BacktestTrade]] = {}
        for variant, threshold in variants:
            strategy = _TestIntervalStrategy(threshold, window.test_start, window.test_end)
            trades = simulate_strategy(simulation_df, strategy).trades
            if any(not (pd.Timestamp(window.test_start) <= pd.Timestamp(t.entry_timestamp) < pd.Timestamp(window.test_end)) for t in trades):
                raise RuntimeError("Official simulator returned a trade outside its OOS interval.")
            trade_sets[variant] = trades
        baseline_count = len(trade_sets["baseline"])
        for variant, threshold in variants:
            trades = trade_sets[variant]
            aggregate_trades[variant].extend(trades)
            rows.append(WindowResult(
                boundaries=window,
                variant=variant,
                min_close_location_filter=threshold,
                metrics=_metrics(trades, baseline_count, baseline=variant == "baseline"),
            ))

    baseline_total = len(aggregate_trades["baseline"])
    aggregates = [AggregateResult(
        variant=variant,
        min_close_location_filter=threshold,
        metrics=_metrics(aggregate_trades[variant], baseline_total, baseline=variant == "baseline"),
    ) for variant, threshold in variants]
    return WalkForwardResult(
        windows=windows,
        window_results=rows,
        aggregates=aggregates,
        verdict=determine_verdict(aggregates[0].metrics, aggregates[1].metrics, rows),
    )


def _n(value: float) -> str:
    return "inf" if value == inf else f"{value:.6f}"


def _row_metrics(m: OOSMetrics) -> str:
    return f"{m.trades} | {m.wins} | {m.losses} | {m.win_rate:.2%} | {_n(m.gross_profit)} | {_n(m.gross_loss)} | {_n(m.fees)} | {_n(m.profit_factor)} | {_n(m.expectancy)} | {_n(m.net_pnl)} | {_n(m.max_drawdown)} | {m.average_holding_candles:.2f} | {m.filtered_trade_retention:.2%}"


def build_report(result: WalkForwardResult) -> str:
    baseline = next(r.metrics for r in result.aggregates if r.variant == "baseline")
    filtered = next(r.metrics for r in result.aggregates if r.variant == "filtered")
    by_variant = {v: [r for r in result.window_results if r.variant == v] for v in ("baseline", "filtered")}
    base_by_window = {r.boundaries.window: r.metrics for r in by_variant["baseline"]}
    comparisons = {
        "PF": sum(r.metrics.profit_factor > base_by_window[r.boundaries.window].profit_factor for r in by_variant["filtered"]),
        "expectancy": sum(r.metrics.expectancy > base_by_window[r.boundaries.window].expectancy for r in by_variant["filtered"]),
        "net PnL": sum(r.metrics.net_pnl > base_by_window[r.boundaries.window].net_pnl for r in by_variant["filtered"]),
        "lower drawdown": sum(r.metrics.max_drawdown < base_by_window[r.boundaries.window].max_drawdown for r in by_variant["filtered"]),
    }
    boundaries = [f"| {w.window} | {w.train_start.isoformat()} | {w.train_end.isoformat()} | {w.test_start.isoformat()} | {w.test_end.isoformat()} |" for w in result.windows]
    header = "Trades | Wins | Losses | Win rate | Gross profit | Gross loss | Fees | PF | Expectancy | Net PnL | Max DD | Avg holding | Retention"
    def table_rows(variant: Variant) -> list[str]:
        return [f"| {r.boundaries.window} | {r.boundaries.test_start.isoformat()} | {r.boundaries.test_end.isoformat()} | {_row_metrics(r.metrics)} |" for r in by_variant[variant]]
    profitable = {v: sum(r.metrics.net_pnl > 0 for r in by_variant[v]) for v in by_variant}
    worst = {v: min(by_variant[v], key=lambda r: r.metrics.net_pnl) for v in by_variant}
    return "\n".join([
        "# Donchian Breakout 15m Close-Location Walk-Forward v1", "",
        "## 1. Methodology", "",
        f"Fixed `{STRATEGY}` parameters: " + ", ".join(f"`{k}={v}`" for k, v in FIXED_PARAMETERS.items()) + ".",
        "The informational six-month training window performs no fitting. Exactly two variants are evaluated: 0.00 baseline and 0.94 filtered. Test windows are three calendar months and advance three calendar months.", "",
        "## 2. Exact window boundaries", "", "Intervals are half-open: start included, end excluded.", "",
        "| Window | Train start | Train end | Test start | Test end |", "|---:|:---|:---|:---|:---|", *boundaries, "",
        "## 3. Leakage controls", "",
        "Features are computed by the official feature pipeline using backward-looking history only. Each simulation receives historical warm-up candles, while an entry mask permits openings only in `[test_start, test_end)`. Pre-test trades therefore cannot open or consume position state. No result changes a parameter. A position still open at `test_end` is liquidated on the final candle before the boundary by the official simulator at that candle's close and recorded as `max_holding`; no future candle is supplied.", "",
        "## 4. Window-level baseline results", "", f"| Window | Test start | Test end | {header} |", "|---:|:---|:---|" + "|---:" * 13 + "|", *table_rows("baseline"), "",
        "## 5. Window-level filtered results", "", f"| Window | Test start | Test end | {header} |", "|---:|:---|:---|" + "|---:" * 13 + "|", *table_rows("filtered"), "",
        "## 6. Aggregate OOS comparison", "",
        "Trades are concatenated in independent OOS-window order. Gross profit/loss, fees, net PnL, PF, expectancy, and the equity-sequence drawdown are recomputed from trade records; window PFs and drawdowns are not averaged.", "",
        f"| Variant | Filter | {header} |", "|:---|---:|" + "|---:" * 13 + "|",
        f"| baseline | 0.00 | {_row_metrics(baseline)} |", f"| filtered | 0.94 | {_row_metrics(filtered)} |", "",
        f"Filtered exceeds baseline in: PF {comparisons['PF']}/{len(result.windows)}, expectancy {comparisons['expectancy']}/{len(result.windows)}, net PnL {comparisons['net PnL']}/{len(result.windows)}, and lower drawdown {comparisons['lower drawdown']}/{len(result.windows)}. Profitable windows: baseline {profitable['baseline']}/{len(result.windows)}, filtered {profitable['filtered']}/{len(result.windows)}. Worst windows by net PnL: baseline window {worst['baseline'].boundaries.window} ({_n(worst['baseline'].metrics.net_pnl)}), filtered window {worst['filtered'].boundaries.window} ({_n(worst['filtered'].metrics.net_pnl)}).", "",
        "## 7. Trade-retention analysis", "", f"Aggregate filtered retention is {filtered.filtered_trade_retention:.2%} ({filtered.trades}/{baseline.trades} executed trades). Because position occupancy changes after removals, retention compares official executed trade counts; the 0.94 signal predicate itself is strictly a subset of baseline signals.", "",
        "## 8. Sample-size limitations", "", f"Only {len(result.windows)} OOS windows and {filtered.trades} aggregate filtered trades are available. One year of data permits only two independent three-month OOS periods after the initial six-month training span, so market-regime coverage and statistical power are limited.", "",
        "## 9. Deterministic verdict", "", f"**{result.verdict}**", "",
        "REJECT applies when filtered expectancy does not improve in most windows or aggregate drawdown worsens by more than 10%. PROMISING_BUT_UNCONFIRMED applies when robustness improves but PF is not above 1, or only two windows are available. CANDIDATE_FOR_MONTE_CARLO additionally requires PF > 1, positive expectancy and net PnL, improvement in most windows, more than two windows, and at least 30 filtered trades.", "",
        "## 10. Recommended next step", "", "Retain these fixed parameters and repeat the same untouched validation on additional unseen years. Run Monte Carlo only if the expanded OOS sample meets the candidate rule.", "",
    ])


def write_outputs(result: WalkForwardResult, report_path: Path, csv_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result), encoding="utf-8")
    records = [{**row.boundaries.model_dump(), "variant": row.variant, "min_close_location_filter": row.min_close_location_filter, **row.metrics.model_dump()} for row in result.window_results]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(csv_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--timeframe", default="15m", choices=["15m"])
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_ohlcv_csv(args.data)
    featured = drop_indicator_warmup_rows(compute_features(resample_ohlcv(raw, args.timeframe)))
    result = run_walk_forward(featured, args.train_months, args.test_months, args.step_months)
    write_outputs(result, args.output_report, args.output_csv)
    print(build_report(result))


if __name__ == "__main__":
    main()
