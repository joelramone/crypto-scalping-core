"""One-shot runner for HYP-VOLATILITY-EXPANSION-BREAKOUT-001 on Dataset B (TASK-033B).

Usage: python -m app.research.volatility_expansion_report --output <json path>
Refuses to run if the Dataset B SHA-256 does not match the registered hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.aggressor_flow_report import _json_safe, hypothesis_verdict, leg_verdict
from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.config import DEFAULT_FEE_RATE
from app.research.simulation import simulate_strategy
from app.research.strategies import volatility_expansion_breakout as vx

HYPOTHESIS_ID = "HYP-VOLATILITY-EXPANSION-BREAKOUT-001"
DATASET_B = Path("data/binance_usdm_BTCUSDT_1m_20230415_20240101.csv")
DATASET_B_SHA256 = "6f29fa337d2dc24a5047124245fed96cce8a57f48aaa6ccbbc38bf492b010d80"
DATASET_B_START = pd.Timestamp("2023-04-15T00:00:00Z")
DATASET_B_END = pd.Timestamp("2024-01-01T00:00:00Z")


def move_diagnostics(candles: pd.DataFrame, sig: pd.Series, side: str) -> dict[str, Any]:
    sign = 1.0 if side == "long" else -1.0
    signal_candle, post = [], []
    for t in sig[sig].index:
        if t + vx.MAX_HOLDING_CANDLES >= len(candles):
            continue
        signal_candle.append(sign * (candles["close"].iat[t] / candles["open"].iat[t] - 1.0))
        post.append(sign * (candles["close"].iat[t + vx.MAX_HOLDING_CANDLES] / candles["open"].iat[t + 1] - 1.0))

    def stats(values: list[float]) -> dict[str, Any]:
        s = pd.Series(values, dtype=float)
        return {"n": int(s.size), "mean": float(s.mean()) if s.size else None,
                "median": float(s.median()) if s.size else None,
                "fraction_positive": float((s > 0).mean()) if s.size else None}

    return {"signal_candle_open_t_to_close_t": stats(signal_candle),
            "post_signal_open_t_plus_1_to_close_t_plus_32": stats(post),
            "round_trip_fee_fraction": 2 * DEFAULT_FEE_RATE}


def run(dataset: Path = DATASET_B) -> dict[str, Any]:
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != DATASET_B_SHA256:
        raise SystemExit(f"Dataset B SHA-256 mismatch: {digest}")
    one_minute = pd.read_csv(dataset)
    one_minute["timestamp"] = pd.to_datetime(one_minute["timestamp"], utc=True)
    if one_minute["timestamp"].min() < DATASET_B_START or one_minute["timestamp"].max() >= DATASET_B_END:
        raise SystemExit("Dataset B contains rows outside the registered period")
    candles = vx.resample_to_15m(one_minute)
    eligible = vx.atr_prev(candles).notna()

    legs: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        strategy = vx.VolatilityExpansionBreakoutStrategy(side)
        sig = vx.signals(candles, side)
        result = simulate_strategy(candles, strategy, fee_rate=DEFAULT_FEE_RATE, execution_mode="next_bar_open")
        raw = [int(i) for i in sig[sig].index]
        diagnostics = calculate_trade_diagnostics(result.trades, raw_entry_signal_indices=raw).model_dump()
        legs[side] = {"strategy": strategy.name(), "diagnostics": diagnostics, "verdict": leg_verdict(diagnostics),
                      "moves": move_diagnostics(candles, sig, side)}

    return _json_safe({
        "hypothesis_id": HYPOTHESIS_ID,
        "dataset": {"path": str(dataset), "sha256": digest, "rows_1m": len(one_minute), "candles_15m": len(candles),
                    "first": str(candles["timestamp"].iat[0]), "last": str(candles["timestamp"].iat[-1])},
        "frozen": {"atr_lookback": vx.ATR_LOOKBACK, "expansion_mult": vx.EXPANSION_MULT, "loc_high": vx.LOC_HIGH,
                   "loc_low": vx.LOC_LOW, "execution_mode": "next_bar_open", "take_profit_pct": vx.TAKE_PROFIT_PCT,
                   "stop_loss_pct": vx.STOP_LOSS_PCT, "max_holding_candles": vx.MAX_HOLDING_CANDLES,
                   "fee_rate_per_side": DEFAULT_FEE_RATE},
        "expansion_frequency": float(vx.expansion(candles)[eligible].mean()),
        "legs": legs,
        "hypothesis_verdict": hypothesis_verdict(legs),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {args.output}")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"hypothesis_verdict": report["hypothesis_verdict"],
                      "legs": {s: l["verdict"] for s, l in report["legs"].items()}}))


if __name__ == "__main__":
    main()
