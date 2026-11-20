"""One-shot runner for HYP-AGGRESSOR-FLOW-PERSISTENCE-001 on Dataset A (TASK-032B).

Usage: python -m app.research.aggressor_flow_report --output <json path>
Refuses to run if the Dataset A SHA-256 does not match the registered hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.config import DEFAULT_FEE_RATE
from app.research.simulation import simulate_strategy
from app.research.strategies.aggressor_flow_persistence import (
    K, MAX_HOLDING_CANDLES, THETA, AggressorFlowPersistenceStrategy, imbalance, onset_signals, resample_to_5m,
)

HYPOTHESIS_ID = "HYP-AGGRESSOR-FLOW-PERSISTENCE-001"
DATASET_A = Path("data/binance_usdm_BTCUSDT_1m_20230101_20230415.csv")
DATASET_A_SHA256 = "2a49857054bf14663fd2a91a6980254883be6d3d84e5147ff5750155a40607af"
DATASET_A_START = pd.Timestamp("2023-01-01T00:00:00Z")
DATASET_A_END = pd.Timestamp("2023-04-15T00:00:00Z")
MIN_TRADES = 100
MAX_CONCENTRATION = 0.80


def leg_verdict(d: dict[str, Any]) -> str:
    if d["completed_trades"] < MIN_TRADES:
        return "INSUFFICIENT_SAMPLE"
    if (d["net_profit_factor"] <= 1 or d["net_expectancy"] <= 0 or d["net_pnl"] <= 0 or
            d["gross_expectancy"] <= 0 or d["positive_pnl_concentration_top_2_months"] > MAX_CONCENTRATION):
        return "BASELINE_REJECT"
    return "BASELINE_CANDIDATE"


def hypothesis_verdict(legs: dict[str, dict[str, Any]]) -> str:
    verdicts = {side: leg["verdict"] for side, leg in legs.items()}
    opposite = {"long": "short", "short": "long"}
    for side, verdict in verdicts.items():
        if verdict == "BASELINE_CANDIDATE" and legs[opposite[side]]["diagnostics"]["gross_expectancy"] > 0:
            return "BASELINE_CANDIDATE"
    if all(leg["diagnostics"]["completed_trades"] < MIN_TRADES for leg in legs.values()):
        return "INSUFFICIENT_SAMPLE"
    return "BASELINE_REJECT"


def move_diagnostics(candles: pd.DataFrame, signals: pd.Series, side: str) -> dict[str, Any]:
    sign = 1.0 if side == "long" else -1.0
    contemporaneous, post = [], []
    for t in signals[signals].index:
        if t - (K - 1) < 0 or t + MAX_HOLDING_CANDLES >= len(candles):
            continue
        contemporaneous.append(sign * (candles["close"].iat[t] / candles["open"].iat[t - (K - 1)] - 1.0))
        post.append(sign * (candles["close"].iat[t + MAX_HOLDING_CANDLES] / candles["open"].iat[t + 1] - 1.0))

    def stats(values: list[float]) -> dict[str, Any]:
        s = pd.Series(values, dtype=float)
        return {"n": int(s.size), "mean": float(s.mean()) if s.size else None,
                "median": float(s.median()) if s.size else None,
                "fraction_positive": float((s > 0).mean()) if s.size else None}

    return {"contemporaneous_open_t_minus_2_to_close_t": stats(contemporaneous),
            "post_observation_open_t_plus_1_to_close_t_plus_12": stats(post),
            "round_trip_fee_fraction": 2 * DEFAULT_FEE_RATE}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def run(dataset: Path = DATASET_A) -> dict[str, Any]:
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != DATASET_A_SHA256:
        raise SystemExit(f"Dataset A SHA-256 mismatch: {digest}")
    one_minute = pd.read_csv(dataset)
    one_minute["timestamp"] = pd.to_datetime(one_minute["timestamp"], utc=True)
    if one_minute["timestamp"].min() < DATASET_A_START or one_minute["timestamp"].max() >= DATASET_A_END:
        raise SystemExit("Dataset A contains rows outside the registered period")
    candles = resample_to_5m(one_minute)
    imb = imbalance(candles)

    legs: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        strategy = AggressorFlowPersistenceStrategy(side)
        signals = onset_signals(candles, side)
        result = simulate_strategy(candles, strategy, fee_rate=DEFAULT_FEE_RATE, execution_mode="next_bar_open")
        raw = [int(i) for i in signals[signals].index]
        diagnostics = calculate_trade_diagnostics(result.trades, raw_entry_signal_indices=raw).model_dump()
        legs[side] = {"strategy": strategy.name(), "diagnostics": diagnostics,
                      "verdict": leg_verdict(diagnostics),
                      "moves": move_diagnostics(candles, signals, side)}

    return _json_safe({
        "hypothesis_id": HYPOTHESIS_ID,
        "dataset": {"path": str(dataset), "sha256": digest, "rows_1m": len(one_minute), "candles_5m": len(candles),
                    "first": str(candles["timestamp"].iat[0]), "last": str(candles["timestamp"].iat[-1])},
        "frozen": {"theta": THETA, "k": K, "execution_mode": "next_bar_open",
                   "take_profit_pct": strategy.take_profit_pct(), "stop_loss_pct": strategy.stop_loss_pct(),
                   "max_holding_candles": MAX_HOLDING_CANDLES, "fee_rate_per_side": DEFAULT_FEE_RATE},
        "imbalance_share_abs_ge_theta": float((imb.abs() >= THETA).mean()),
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
