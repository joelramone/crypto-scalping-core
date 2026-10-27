"""Minimal executable research backtester entry point."""

import argparse
import hashlib
import io
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.config import DEFAULT_DATA_PATH, DEFAULT_FEE_RATE
from app.research.data_utils import OPTIONAL_SUM_COLUMNS, SUPPORTED_INTERVALS, resample_ohlcv
from app.research.features import FEATURE_COLUMNS, compute_features
from app.research.results import print_backtest_metrics, print_dataset_summary
from app.research.regimes import classify_regimes, compute_regime_features
from app.research.simulation import FIXED_NOTIONAL_USDT, simulate_strategy
from app.research.strategies import (
    BaseStrategy,
    BaselineTrendStrategy,
    BollingerReversionStrategy,
    DonchianBreakoutStrategy,
    EmaPullbackStrategy,
    MeanReversionStrategy,
    MomentumPullbackContinuationStrategy,
    PullbackStrategy,
    RegimeTransitionStrategy,
    ShortCompressionBreakdownStrategy,
    ShortFailedBreakoutStrategy,
    ShortRegimeTransitionStrategy,
    ShortMomentumPullbackContinuationStrategy,
    VolatilityExhaustionStrategy,
)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
STRATEGIES: dict[str, type[BaseStrategy]] = {
    "baseline_trend": BaselineTrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum_pullback_continuation": MomentumPullbackContinuationStrategy,
    "pullback": PullbackStrategy,
    "bollinger_reversion": BollingerReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "ema_pullback": EmaPullbackStrategy,
    "volatility_exhaustion": VolatilityExhaustionStrategy,
    "regime_transition": RegimeTransitionStrategy,
    "short_compression_breakdown": ShortCompressionBreakdownStrategy,
    "short_failed_breakout": ShortFailedBreakoutStrategy,
    "short_regime_transition": ShortRegimeTransitionStrategy,
    "short_momentum_pullback_continuation": ShortMomentumPullbackContinuationStrategy,
}


def add_official_base_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Expose the official causal classifier output under its hypothesis name."""
    classified = classify_regimes(compute_regime_features(df))
    return classified.rename(columns={"regime": "base_regime"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research backtester."""
    parser = argparse.ArgumentParser(description="Run the V2 research backtester.")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to an OHLCV CSV file.",
    )
    parser.add_argument(
        "--strategy",
        default="baseline_trend",
        choices=sorted(STRATEGIES),
        help="Research strategy to run.",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        choices=sorted(SUPPORTED_INTERVALS),
        help="Research timeframe to test after loading source candles.",
    )
    parser.add_argument("--execution-mode", choices=["signal_close", "next_bar_open"], default="signal_close")
    parser.add_argument("--json", action="store_true", help="Fixed baseline_trend/1m metrics; requires --expected-sha256")
    parser.add_argument("--loss-attribution", action="store_true", help="Attribute completed trades in the same JSON next-open run; does not load a previous summary")
    parser.add_argument("--expected-sha256", help="Reject dataset bytes that do not match this SHA-256")
    return parser.parse_args()


def load_ohlcv_csv(data_path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV candles from a CSV file."""
    df = pd.read_csv(data_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required CSV columns: {missing}")

    optional_columns = [column for column in OPTIONAL_SUM_COLUMNS if column in df.columns]
    df = df.loc[:, [*REQUIRED_COLUMNS, *optional_columns]].copy()
    for column in NUMERIC_COLUMNS:
        df[column] = df[column].astype(float)
    for column in optional_columns:
        df[column] = df[column].astype(float)

    return df


def drop_indicator_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values introduced by indicator warmup periods."""
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def format_feature_value(value: Any) -> str:
    """Format feature values for compact command-line output."""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a summary of calculated technical features."""
    last_row = df.iloc[-1]

    print(f"Total rows after features: {len(df)}")
    print(f"Feature columns created: {', '.join(FEATURE_COLUMNS)}")
    print("Last row feature values:")
    for column in FEATURE_COLUMNS:
        print(f"  {column}: {format_feature_value(last_row[column])}")


def load_strategy(strategy_key: str) -> BaseStrategy:
    """Instantiate a configured research strategy by CLI key."""
    return STRATEGIES[strategy_key]()


def code_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.strip()
    return {"revision": revision, "working_tree_dirty": bool(dirty)}


def diagnostic_report(result, strategy, digest, identity, duration, execution_mode, raw_df, featured_df) -> dict:
    metrics = result.metrics
    pf = metrics.profit_factor
    return {
        "status": "TECHNICAL_DIAGNOSTIC", "strategy": "baseline_trend", "timeframe": "1m", "direction": "long",
        "parameters": {"take_profit_pct": strategy.take_profit_pct(), "stop_loss_pct": strategy.stop_loss_pct(),
                       "max_holding_candles": strategy.max_holding_candles(), "notional_usdt": FIXED_NOTIONAL_USDT,
                       "fee_rate_per_side": DEFAULT_FEE_RATE, "indicator_exits": False,
                       "entry_rules": "close > EMA200; EMA20 slope > 0; 45 <= RSI14 <= 70; volume/SMA20 > 1.2; ATR14 > trailing ATR14 median(200,min_periods=1)"},
        "dataset": {"sha256": digest, "raw_rows": len(raw_df), "featured_rows": len(featured_df),
                    "start_utc": str(raw_df.iloc[0]["timestamp"]), "last_open_utc": str(raw_df.iloc[-1]["timestamp"])},
        "code": identity, "execution_duration_seconds": duration,
        "execution_mode": execution_mode,
        "execution_convention": ("signal at completed candle close; entry at following open; entry candle eligible for exits; holding limit includes entry candle"
                                 if execution_mode == "next_bar_open" else "legacy signal-candle close; exits start on following candle"),
        "exit_priority": "stop_loss before take_profit before indicator exit before max-holding close",
        "overlap": "one position; resume signal scan at exit index + 1",
        "metrics": {"trades": metrics.total_trades, "gross_pnl": metrics.gross_pnl, "net_pnl": metrics.net_pnl,
                    "total_fees": metrics.estimated_fees, "profit_factor": pf if math.isfinite(pf) else None,
                    "profit_factor_status": "finite" if math.isfinite(pf) else "positive_profit_without_losses",
                    "maximum_drawdown_usdt": metrics.max_drawdown},
        "limitations": "Fixed complete fills; no spread/slippage/latency/impact/funding model; barrier fills do not model opening gaps. Technical diagnostic only.",
    }


def run_backtest(args) -> None:
    started = time.perf_counter()
    attribution = getattr(args, "loss_attribution", False)
    if attribution and (not args.json or args.execution_mode != "next_bar_open"):
        raise ValueError("--loss-attribution requires --json and --execution-mode next_bar_open")
    if args.json and (args.strategy != "baseline_trend" or args.timeframe != "1m" or not args.expected_sha256):
        raise ValueError("JSON diagnostic requires baseline_trend, 1m and --expected-sha256")
    identity = code_identity() if args.json else None
    payload = Path(args.data).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        raise ValueError("Dataset SHA-256 mismatch; no strategy executed")
    raw_df = load_ohlcv_csv(io.BytesIO(payload))
    if args.json:
        from app.research.data_validation.validate_ohlcv import validate_ohlcv
        validation = validate_ohlcv(raw_df)
        if raw_df.empty or not validation.structurally_valid:
            raise ValueError(f"Invalid diagnostic OHLCV: {validation.errors}")
        if not raw_df[list(NUMERIC_COLUMNS)].map(math.isfinite).all().all():
            raise ValueError("Nonfinite diagnostic OHLCV")
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="raise")
    strategy = load_strategy(args.strategy)
    df = resample_ohlcv(raw_df, args.timeframe)
    featured_df = compute_features(df)
    if args.strategy in {"regime_transition", "short_regime_transition"}:
        featured_df = add_official_base_regime(featured_df)
    featured_df = drop_indicator_warmup_rows(featured_df)
    if not args.json:
        print("Research backtester initialized")
        print_dataset_summary(df, args.data, args.timeframe)
        print_feature_summary(featured_df)
    backtest_result = simulate_strategy(featured_df, strategy, execution_mode=args.execution_mode)
    if args.json:
        report = diagnostic_report(backtest_result, strategy, digest, identity,
                                   time.perf_counter() - started, args.execution_mode, raw_df, featured_df)
        report["dataset"]["path"] = str(args.data)
        if attribution:
            from app.research.analysis.trade_diagnostics import calculate_loss_attribution
            report["loss_attribution"] = calculate_loss_attribution(backtest_result.trades)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    else:
        print_backtest_metrics(strategy.name(), backtest_result.metrics, args.timeframe)


def main() -> None:
    args = parse_args()
    try:
        run_backtest(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if not args.json:
            raise
        print(json.dumps({"status": "DIAGNOSTIC_ERROR", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
