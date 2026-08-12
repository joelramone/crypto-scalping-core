"""Phase 1 observational market-regime analysis command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.research.backtester import drop_indicator_warmup_rows, load_ohlcv_csv
from app.research.data_utils import SUPPORTED_INTERVALS, resample_ohlcv
from app.research.features import compute_features
from app.research.regimes.classifier import BASE_REGIMES, RegimeConfig, classify_regimes
from app.research.regimes.features import REGIME_FEATURE_COLUMNS, compute_regime_features
from app.research.simulation import BacktestTrade, calculate_metrics, simulate_strategy
from app.research.strategies import (
    BaselineTrendStrategy,
    BollingerReversionStrategy,
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
)

STRATEGIES = {
    "donchian_breakout": DonchianBreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "bollinger_reversion": BollingerReversionStrategy,
    "baseline_trend": BaselineTrendStrategy,
}
REPORT_REGIMES = (*BASE_REGIMES, "HIGH_VOLATILITY")


def prepare_dataset(path: str | Path, timeframe: str) -> pd.DataFrame:
    """Load, resample, feature, and label one dataset in complete isolation."""
    candles = resample_ohlcv(load_ohlcv_csv(path), timeframe)
    featured = compute_features(candles)
    featured = compute_regime_features(featured)
    featured = drop_indicator_warmup_rows(featured)
    featured = featured.dropna(subset=REGIME_FEATURE_COLUMNS).reset_index(drop=True)
    return classify_regimes(featured)


def _mask(df: pd.DataFrame, regime: str) -> pd.Series:
    if regime == "HIGH_VOLATILITY":
        return df["is_high_volatility"]
    return df["regime"].eq(regime)


def distribution_rows(df: pd.DataFrame, period: str) -> list[dict[str, object]]:
    """Build descriptive candle statistics for each base regime and overlay."""
    rows: list[dict[str, object]] = []
    for regime in REPORT_REGIMES:
        subset = df.loc[_mask(df, regime)]
        run_ids = _mask(df, regime).ne(_mask(df, regime).shift()).cumsum()
        durations = _mask(df, regime).groupby(run_ids).sum()
        durations = durations[durations > 0]
        rows.append(
            {
                "record_type": "distribution",
                "period": period,
                "strategy": "",
                "regime": regime,
                "candles": len(subset),
                "percentage": len(subset) / len(df) if len(df) else 0.0,
                "average_atr": subset["regime_atr14"].mean(),
                "average_realized_volatility": subset["realized_volatility_20"].mean(),
                "average_return_4": subset["return_4"].mean(),
                "average_return_12": subset["return_12"].mean(),
                "average_return_24": subset["return_24"].mean(),
                "average_duration_candles": durations.mean() if len(durations) else 0.0,
            }
        )
    return rows


def transition_rows(df: pd.DataFrame, period: str) -> list[dict[str, object]]:
    """Count transitions between consecutive exclusive base regimes."""
    previous = df["regime"].shift(1)
    changed = previous.notna() & previous.ne(df["regime"])
    counts = pd.DataFrame({"from": previous[changed], "to": df.loc[changed, "regime"]})
    rows: list[dict[str, object]] = []
    for (source, target), count in counts.value_counts().sort_index().items():
        rows.append(
            {"record_type": "transition", "period": period, "strategy": source,
             "regime": target, "transition_count": int(count)}
        )
    return rows


def assign_trades_to_regimes(
    trades: Iterable[BacktestTrade], df: pd.DataFrame
) -> dict[str, list[BacktestTrade]]:
    """Assign official simulator trades using only their entry candle label."""
    assigned = {regime: [] for regime in REPORT_REGIMES}
    for trade in trades:
        entry = df.iloc[trade.entry_index]
        assigned[str(entry["regime"])].append(trade)
        if bool(entry["is_high_volatility"]):
            assigned["HIGH_VOLATILITY"].append(trade)
    return assigned


def strategy_rows(df: pd.DataFrame, period: str) -> list[dict[str, object]]:
    """Simulate unchanged strategy defaults and aggregate by entry regime."""
    rows: list[dict[str, object]] = []
    for strategy_name, strategy_type in STRATEGIES.items():
        result = simulate_strategy(df, strategy_type())
        for regime, trades in assign_trades_to_regimes(result.trades, df).items():
            metrics = calculate_metrics(trades)
            rows.append(
                {
                    "record_type": "strategy_regime",
                    "period": period,
                    "strategy": strategy_name,
                    "regime": regime,
                    "trades": metrics.total_trades,
                    "wins": metrics.wins,
                    "losses": metrics.losses,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "expectancy": metrics.expectancy,
                    "net_pnl": metrics.net_pnl,
                    "max_drawdown": metrics.max_drawdown,
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return pd.DataFrame(rows).reindex(columns=columns).to_markdown(index=False, floatfmt=".6f")


def _research_answers(
    distributions: list[dict[str, object]], strategies: list[dict[str, object]]
) -> str:
    """Render data-driven answers without turning observations into filters."""
    strategy_index = {
        (str(row["period"]), str(row["strategy"]), str(row["regime"])): row
        for row in strategies
    }
    distribution_index = {
        (str(row["period"]), str(row["regime"])): row for row in distributions
    }

    def pair(period: str, strategy: str, regime: str, metric: str) -> float:
        return float(strategy_index[(period, strategy, regime)][metric])

    lines: list[str] = []
    for period in ("2025", "2026"):
        donchian_trend_trades = sum(
            int(pair(period, "donchian_breakout", regime, "trades"))
            for regime in ("TREND_UP", "TREND_DOWN")
        )
        donchian_trend_pnl = sum(
            pair(period, "donchian_breakout", regime, "net_pnl")
            for regime in ("TREND_UP", "TREND_DOWN")
        )
        range_expectancy = pair(period, "mean_reversion", "RANGE", "expectancy")
        high_vol_pnl = sum(
            pair(period, strategy, "HIGH_VOLATILITY", "net_pnl")
            for strategy in STRATEGIES
        )
        lines.append(
            f"- **{period}:** Donchian trend regimes contain {donchian_trend_trades} trades "
            f"and {donchian_trend_pnl:.6f} USDT net PnL. Mean reversion RANGE expectancy "
            f"is {range_expectancy:.6f} USDT. Across the four strategies, HIGH_VOLATILITY "
            f"entry trades total {high_vol_pnl:.6f} USDT net PnL."
        )
    shifts = []
    for regime in BASE_REGIMES:
        change = 100.0 * (
            float(distribution_index[("2026", regime)]["percentage"])
            - float(distribution_index[("2025", regime)]["percentage"])
        )
        shifts.append((abs(change), regime, change))
    _, largest_regime, largest_change = max(shifts)
    lines.extend(
        [
            f"- **Distribution difference:** the largest exclusive-regime share change is "
            f"{largest_regime} at {largest_change:+.2f} percentage points in 2026 versus 2025.",
            "- **Rejected candidate attribution:** distribution and conditional results can "
            "show compatibility with a regime-shift hypothesis, but cannot establish that the "
            "2026 regime mix caused the rejected candidate's loss.",
            "- **Avoidance hypothesis:** a regime is a credible avoidance hypothesis only when "
            "negative expectancy is directionally consistent across both independently evaluated "
            "periods and supported by non-trivial trade counts; it is not a filter recommendation.",
        ]
    )
    return "\n".join(lines)


def build_report(all_rows: list[dict[str, object]], config: RegimeConfig) -> str:
    distributions = [row for row in all_rows if row["record_type"] == "distribution"]
    transitions = [row for row in all_rows if row["record_type"] == "transition"]
    strategies = [row for row in all_rows if row["record_type"] == "strategy_regime"]
    dist_columns = ["period", "regime", "candles", "percentage", "average_atr",
                    "average_realized_volatility", "average_return_4", "average_return_12",
                    "average_return_24", "average_duration_candles"]
    strategy_columns = ["period", "strategy", "regime", "trades", "wins", "losses",
                        "win_rate", "profit_factor", "expectancy", "net_pnl", "max_drawdown"]
    answers = _research_answers(distributions, strategies)
    return f"""# Market Regime Analysis v1

## 1. Executive summary

This is observational Phase 1 research. It labels each year independently, runs unchanged default research strategies through the official simulator, and generates hypotheses rather than filters. `HIGH_VOLATILITY` is an overlay, so its counts intentionally overlap the exclusive base regimes.

## 2. Regime methodology

`TREND_UP` requires EMA20 > EMA50 > EMA200, positive one-candle percentage slopes, and ADX14 >= {config.trend_adx_threshold}. `TREND_DOWN` is its exact mirror. `RANGE` requires low ADX, small absolute EMA separations, and small absolute slopes. Remaining candles are explicitly `NEUTRAL`. High volatility is a separate overlay to preserve direction while testing volatility concentration.

All calculations are causal. OHLCV is resampled into closed 15-minute bars; indicators use trailing rolling/EWM calculations. The high-volatility threshold is shifted one candle. Regime assignment uses the simulator trade's entry index, never its outcome.

## 3. Feature definitions

- ATR14: Wilder-style exponentially smoothed true range; ATR percentage divides it by close.
- Realized volatility: standard deviation of one-candle percentage returns over 20 candles.
- EMA20, EMA50, EMA200: causal exponential moving averages.
- EMA slopes: one-candle percentage change in EMA20 and EMA50.
- EMA separations: EMA20/EMA50 - 1 and EMA50/EMA200 - 1.
- ADX14: Wilder-style directional movement strength.
- Returns: trailing close percentage changes over 4, 12, and 24 candles. These are descriptive and not forward returns.

## 4. Thresholds used

- Trend ADX: {config.trend_adx_threshold}
- Range ADX: {config.range_adx_threshold}
- Maximum absolute range EMA separation: {config.range_max_ema_separation_pct:.4%}
- Maximum absolute range slope: {config.range_max_slope_pct:.4%}
- High-volatility threshold: trailing {config.high_volatility_percentile:.0%} percentile over {config.volatility_lookback} candles, with {config.volatility_min_history} prior observations and a one-candle shift.

These conventional, explainable defaults were selected without reference to strategy PnL and were not optimized.

## 5. 2025 regime distribution

{_markdown_table([r for r in distributions if r['period'] == '2025'], dist_columns)}

## 6. 2026 regime distribution

{_markdown_table([r for r in distributions if r['period'] == '2026'], dist_columns)}

## 7. Regime transition analysis

Only changes between exclusive base regimes are counted; persistence is represented by average duration.

{_markdown_table(transitions, ['period', 'strategy', 'regime', 'transition_count']).replace('strategy', 'from').replace('regime', 'to')}

## 8. Strategy × regime matrix

`HIGH_VOLATILITY` overlaps base-regime rows. Profit Factor uses net winning PnL divided by absolute net losing PnL. Max drawdown is recalculated on each pair's chronological trade subsequence.

{_markdown_table(strategies, strategy_columns)}

## 9. Cross-year comparison

The distribution table provides the direct percentage-point comparison. Differences should be interpreted descriptively: 2026 is a partial year and each period was independently warmed up and labeled. A regime-share change alone cannot establish that it caused a strategy result.

Combined descriptive candle summary (concatenated only after independent labeling):

{_markdown_table([r for r in distributions if r['period'] == 'combined_descriptive'], dist_columns)}

## 10. Strongest observations

{answers}

## 11. Weak or contradictory evidence

- Sparse strategy/regime cells cannot support robust conclusions.
- `NEUTRAL` is intentionally broad, and HIGH_VOLATILITY overlaps it and the named base regimes.
- Descriptive trailing returns characterize states; they are not predictive forward returns.
- Different regime distributions do not prove that the rejected Donchian candidate failed *because* of 2026. That claim would require a prospective test or controlled attribution. The candidate remains rejected and is not optimized here.

## 12. Candidate hypotheses for next experiments

1. Prospectively test, on untouched data, whether Donchian economics improve when entries occur in directionally aligned trend regimes.
2. Prospectively test whether mean-reversion entries in RANGE have better expectancy than their non-RANGE entries.
3. Test a pre-registered high-volatility avoidance hypothesis only if the overlay shows consistently concentrated losses in both periods.
4. Test regime robustness using fixed thresholds on another instrument or later period before considering any production filter.

These are hypotheses only. Phase 1 does not create, tune, or recommend trading filters.

## 13. Warning against same-data optimization

**Do not optimize regime thresholds, strategy parameters, or entry filters on these 2025/2026 results.** Doing so would convert descriptive evidence into in-sample selection and invalidate a future confirmation test. Freeze any next-experiment rule before evaluating untouched data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 market-regime research.")
    parser.add_argument("--data-2025", required=True)
    parser.add_argument("--data-2026", required=True)
    parser.add_argument("--timeframe", default="15m", choices=sorted(SUPPORTED_INTERVALS))
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegimeConfig()
    datasets = {
        "2025": prepare_dataset(args.data_2025, args.timeframe),
        "2026": prepare_dataset(args.data_2026, args.timeframe),
    }
    rows: list[dict[str, object]] = []
    for period, df in datasets.items():
        rows.extend(distribution_rows(df, period))
        rows.extend(transition_rows(df, period))
        rows.extend(strategy_rows(df, period))

    combined = pd.concat(datasets.values(), ignore_index=True)
    rows.extend(distribution_rows(combined, "combined_descriptive"))
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(rows, config), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
