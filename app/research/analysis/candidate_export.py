"""Export baseline Donchian entry candidates and their realized outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from app.research.data_utils import resample_ohlcv
except Exception:  # pragma: no cover - fallback for older environments
    def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if interval == "1m":
            return df.copy()
        rule_map = {"5m": "5min", "15m": "15min"}
        if interval not in rule_map:
            raise ValueError(f"Unsupported interval: {interval}")
        aggregations = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        optional_sum_columns = [
            "quote_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        for column in optional_sum_columns:
            if column in df.columns:
                aggregations[column] = "sum"
        resampled = df.resample(rule_map[interval]).agg(aggregations).dropna(subset=["open", "high", "low", "close"])
        resampled.index.name = "timestamp"
        return resampled.reset_index()


FEATURE_COLUMNS = [
    "quality_score",
    "body_to_range",
    "close_location_value",
    "range_expansion_ratio",
    "atr_expansion_ratio",
    "ema20_slope_pct",
    "ema_alignment_strength",
    "breakout_distance_pct",
    "volume_ratio",
    "rsi14",
    "atr14",
]


@dataclass(slots=True)
class ExportConfig:
    strategy: str
    data: Path
    timeframe: str
    output: Path
    lookback: int = 3
    volume_ratio: float = 0.4
    take_profit_pct: float = 0.012
    stop_loss_pct: float = 0.008
    max_holding_candles: int = 24
    min_quality_score: int = 0
    min_body_to_range: float = 0.6
    min_close_location: float = 0.7
    min_range_expansion: float = 1.25
    min_atr_expansion: float = 0.9
    min_ema20_slope_pct: float = 0.0
    min_ema_alignment_strength: float = 0.0
    min_breakout_distance_pct: float = 0.0005


def _parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export baseline Donchian candidates with realized outcomes.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--volume-ratio", type=float, default=0.4)
    parser.add_argument("--take-profit-pct", type=float, default=0.012)
    parser.add_argument("--stop-loss-pct", type=float, default=0.008)
    parser.add_argument("--max-holding-candles", type=int, default=24)
    parser.add_argument("--min-quality-score", type=int, default=0)
    parser.add_argument("--min-body-to-range", type=float, default=0.6)
    parser.add_argument("--min-close-location", type=float, default=0.7)
    parser.add_argument("--min-range-expansion", type=float, default=1.25)
    parser.add_argument("--min-atr-expansion", type=float, default=0.9)
    parser.add_argument("--min-ema20-slope-pct", type=float, default=0.0)
    parser.add_argument("--min-ema-alignment-strength", type=float, default=0.0)
    parser.add_argument("--min-breakout-distance-pct", type=float, default=0.0005)
    args = parser.parse_args()
    return ExportConfig(
        strategy=args.strategy,
        data=Path(args.data),
        timeframe=args.timeframe,
        output=Path(args.output),
        lookback=args.lookback,
        volume_ratio=args.volume_ratio,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_holding_candles=args.max_holding_candles,
        min_quality_score=args.min_quality_score,
        min_body_to_range=args.min_body_to_range,
        min_close_location=args.min_close_location,
        min_range_expansion=args.min_range_expansion,
        min_atr_expansion=args.min_atr_expansion,
        min_ema20_slope_pct=args.min_ema20_slope_pct,
        min_ema_alignment_strength=args.min_ema_alignment_strength,
        min_breakout_distance_pct=args.min_breakout_distance_pct,
    )


def load_market_data(data_path: Path, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected timestamp column. Available columns: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    resampled = resample_ohlcv(df, timeframe)
    if "timestamp" in resampled.columns:
        resampled["timestamp"] = pd.to_datetime(resampled["timestamp"], utc=True)
        resampled = resampled.sort_values("timestamp").set_index("timestamp")
    else:
        resampled = resampled.sort_index()
    return resampled


def _atr14(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=14, min_periods=14).mean()


def _rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def enrich_features(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    enriched["ema50"] = enriched["close"].ewm(span=50, adjust=False).mean()
    enriched["ema200"] = enriched["close"].ewm(span=200, adjust=False).mean()
    enriched["ema20_slope"] = enriched["ema20"].diff()
    enriched["ema20_slope_pct"] = enriched["ema20"].pct_change().fillna(0.0)
    enriched["atr14"] = _atr14(enriched)
    enriched["rsi14"] = _rsi14(enriched["close"])
    candle_range = (enriched["high"] - enriched["low"]).replace(0.0, pd.NA)
    enriched["body_to_range"] = (enriched["close"] - enriched["open"]).abs() / candle_range
    enriched["close_location_value"] = (enriched["close"] - enriched["low"]) / candle_range
    enriched["candle_range"] = (enriched["high"] - enriched["low"]).fillna(0.0)
    enriched["range_expansion_ratio"] = enriched["candle_range"] / enriched["candle_range"].rolling(window=20, min_periods=20).median()
    enriched["atr_expansion_ratio"] = enriched["candle_range"] / enriched["atr14"].replace(0.0, pd.NA)
    enriched["ema_alignment_strength"] = (
        (enriched["ema20"] - enriched["ema50"]) / enriched["close"].replace(0.0, pd.NA)
        + (enriched["ema50"] - enriched["ema200"]) / enriched["close"].replace(0.0, pd.NA)
    )
    enriched["volume_ratio"] = enriched["volume"] / enriched["volume"].rolling(window=20, min_periods=20).mean()
    donchian_high = enriched["high"].rolling(window=config.lookback, min_periods=config.lookback).max().shift(1)
    enriched["donchian_high"] = donchian_high
    enriched["breakout_distance_pct"] = (enriched["close"] - donchian_high) / donchian_high.replace(0.0, pd.NA)
    enriched["atr_median"] = enriched["atr14"].rolling(window=20, min_periods=20).median()
    numeric_columns = [column for column in FEATURE_COLUMNS if column in enriched.columns]
    enriched[numeric_columns] = enriched[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return enriched


def compute_quality_score(df: pd.DataFrame, config: ExportConfig) -> pd.Series:
    checks = pd.DataFrame(
        {
            "body_to_range": df["body_to_range"] >= config.min_body_to_range,
            "close_location_value": df["close_location_value"] >= config.min_close_location,
            "range_expansion_ratio": df["range_expansion_ratio"] >= config.min_range_expansion,
            "atr_expansion_ratio": df["atr_expansion_ratio"] >= config.min_atr_expansion,
            "ema20_slope_pct": df["ema20_slope_pct"] >= config.min_ema20_slope_pct,
            "ema_alignment_strength": df["ema_alignment_strength"] >= config.min_ema_alignment_strength,
            "breakout_distance_pct": df["breakout_distance_pct"] >= config.min_breakout_distance_pct,
        }
    ).fillna(False)
    return checks.sum(axis=1)


def compute_base_candidates(df: pd.DataFrame, config: ExportConfig) -> pd.Series:
    return (
        (df["close"] > df["donchian_high"])
        & (df["close"] > df["ema200"])
        & (df["ema20_slope"] > 0)
        & (df["volume_ratio"] > config.volume_ratio)
        & (df["atr14"] > df["atr_median"])
    ).fillna(False)


def simulate_candidate_outcomes(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    working = enrich_features(df, config)
    working["quality_score"] = compute_quality_score(working, config)
    base_candidates = compute_base_candidates(working, config)

    rows: list[dict[str, Any]] = []
    position_exit_index = -1
    for index_position, (timestamp, row) in enumerate(working.iterrows()):
        if not bool(base_candidates.iloc[index_position]):
            continue

        export_row: dict[str, Any] = {
            "timestamp": timestamp,
            "entry_price": row["close"],
            "trade_opened": False,
            "trade_result": "not_opened",
            "gross_pnl": pd.NA,
            "net_pnl": pd.NA,
            "exit_reason": "",
            "holding_candles": pd.NA,
        }
        for column in FEATURE_COLUMNS:
            export_row[column] = row[column]

        if index_position <= position_exit_index:
            rows.append(export_row)
            continue

        entry_price = float(row["close"])
        stop_price = entry_price * (1.0 - config.stop_loss_pct)
        take_profit_price = entry_price * (1.0 + config.take_profit_pct)
        exit_idx = min(index_position + config.max_holding_candles, len(working) - 1)
        exit_price = float(working.iloc[exit_idx]["close"])
        exit_reason = "max_holding"

        for future_idx in range(index_position + 1, exit_idx + 1):
            future_row = working.iloc[future_idx]
            if float(future_row["low"]) <= stop_price:
                exit_idx = future_idx
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if float(future_row["high"]) >= take_profit_price:
                exit_idx = future_idx
                exit_price = take_profit_price
                exit_reason = "take_profit"
                break
        else:
            exit_price = float(working.iloc[exit_idx]["close"])

        gross_pnl = (exit_price / entry_price) - 1.0
        export_row.update(
            {
                "trade_opened": True,
                "trade_result": "winner" if gross_pnl > 0 else "loser" if gross_pnl < 0 else "flat",
                "gross_pnl": gross_pnl,
                "net_pnl": gross_pnl,
                "exit_reason": exit_reason,
                "holding_candles": exit_idx - index_position,
            }
        )
        position_exit_index = exit_idx
        rows.append(export_row)

    return pd.DataFrame(rows)


def export_candidates(config: ExportConfig) -> Path:
    if config.strategy != "donchian_breakout":
        raise ValueError("Only donchian_breakout is supported by this exporter.")

    df = load_market_data(config.data, config.timeframe)
    exported = simulate_candidate_outcomes(df, config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(config.output, index=False)
    print(f"Exported {len(exported)} candidates")
    return config.output


def main() -> None:
    config = _parse_args()
    output_path = export_candidates(config)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
