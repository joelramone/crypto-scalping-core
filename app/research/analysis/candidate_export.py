"""Export baseline Donchian entry candidates and their realized outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.backtester import drop_indicator_warmup_rows, load_ohlcv_csv
from app.research.data_utils import resample_ohlcv
from app.research.features import compute_features
from app.research.simulation import simulate_strategy
from app.research.strategies import DonchianBreakoutStrategy


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

EXPORT_COLUMNS = [
    "timestamp",
    "entry_price",
    "trade_opened",
    "trade_result",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "holding_candles",
    "notional",
    "gross_return",
    "fees",
    "gross_pnl",
    "net_pnl",
    *FEATURE_COLUMNS,
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
    raw_df = load_ohlcv_csv(data_path)
    return resample_ohlcv(raw_df, timeframe)


def enrich_features(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    del config
    return drop_indicator_warmup_rows(compute_features(df))


def _build_strategy(config: ExportConfig) -> DonchianBreakoutStrategy:
    return DonchianBreakoutStrategy(
        lookback=config.lookback,
        volume_ratio=config.volume_ratio,
        take_profit_pct=config.take_profit_pct,
        stop_loss_pct=config.stop_loss_pct,
        max_holding_candles=config.max_holding_candles,
        min_quality_score=config.min_quality_score,
        min_body_to_range=config.min_body_to_range,
        min_close_location=config.min_close_location,
        min_range_expansion=config.min_range_expansion,
        min_atr_expansion=config.min_atr_expansion,
        min_ema20_slope_pct=config.min_ema20_slope_pct,
        min_ema_alignment_strength=config.min_ema_alignment_strength,
        min_breakout_distance_pct=config.min_breakout_distance_pct,
    )


def compute_quality_score(df: pd.DataFrame, config: ExportConfig) -> pd.Series:
    strategy = _build_strategy(config)
    strategy.generate_entries(df)
    return strategy.last_quality_scores


def compute_base_candidates(df: pd.DataFrame, config: ExportConfig) -> pd.Series:
    strategy = _build_strategy(config)
    strategy.min_quality_score = 0
    return strategy.generate_entries(df)


def simulate_candidate_outcomes(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    working = enrich_features(df, config)
    strategy = _build_strategy(config)
    base_candidates = strategy.generate_entries(working)
    working["quality_score"] = strategy.last_quality_scores
    working["breakout_distance_pct"] = strategy.last_breakout_distance_pct
    result = simulate_strategy(working, strategy)
    trades_by_entry_index = {trade.entry_index: trade for trade in result.trades}

    rows: list[dict[str, Any]] = []
    for index_position, (_, row) in enumerate(working.iterrows()):
        if not bool(base_candidates.iloc[index_position]):
            continue

        trade = trades_by_entry_index.get(index_position)
        export_row: dict[str, Any] = {
            "timestamp": row["timestamp"],
            "entry_price": row["close"],
            "trade_opened": trade is not None,
            "trade_result": (
                "winner"
                if trade is not None and trade.net_pnl > 0.0
                else "loser"
                if trade is not None and trade.net_pnl < 0.0
                else "flat"
                if trade is not None
                else "not_opened"
            ),
            "exit_timestamp": trade.exit_timestamp if trade is not None else pd.NA,
            "exit_price": trade.exit_price if trade is not None else pd.NA,
            "notional": trade.notional if trade is not None else pd.NA,
            "gross_return": (
                trade.gross_pnl / trade.notional if trade is not None else pd.NA
            ),
            "gross_pnl": trade.gross_pnl if trade is not None else pd.NA,
            "fees": trade.fees if trade is not None else pd.NA,
            "net_pnl": trade.net_pnl if trade is not None else pd.NA,
            "exit_reason": trade.exit_reason if trade is not None else "",
            "holding_candles": trade.holding_candles if trade is not None else pd.NA,
        }
        for column in FEATURE_COLUMNS:
            export_row[column] = row[column]
        rows.append(export_row)

    exported = pd.DataFrame.from_records(rows, columns=EXPORT_COLUMNS)
    if exported.empty:
        exported["trade_opened"] = pd.Series(dtype="bool")
    return exported


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
