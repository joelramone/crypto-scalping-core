"""HYP-DAILY-TREND-XASSET-001: the frozen daily trend rule on SPY, EFA, TLT and GLD.

Reuses app.research.daily_trend unchanged (same rule, fees and execution) on
dividend-adjusted prices, annualized with 252 trading days.

    python -m app.research.daily_trend_cross_asset --output <json path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.daily_trend import DRAWDOWN_RATIO_LIMIT, FEE_RATE, SMA_LENGTHS, evaluate

HYPOTHESIS_ID = "HYP-DAILY-TREND-XASSET-001"
TRADING_DAYS = 252
MIN_PASSING_ASSETS = 3
DATASETS = {
    "SPY": "856a0ddb4d690e6b39a164b01a7bdd5c7db067ce0b09aef3d1d5ebabb4123f27",
    "EFA": "88f2efbb87fd8059d680b173cec94bf2d09a4d3115b9f1030d75ef877b806976",
    "TLT": "4353c055643124b15b12e86b9fc9311a63ab2d99d7bf22a81b718213a53aba4e",
    "GLD": "c5be9eff7f25d4b46bd259bc04924eb744a64a05cb9aa414557523b1b98e12b5",
}


def path_for(ticker: str) -> Path:
    return Path(f"data/yahoo_{ticker}_1d_20050101_20261101.csv")


def load(ticker: str, sha256: str, path: Path | None = None) -> pd.DataFrame:
    path = path or path_for(ticker)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} SHA-256 mismatch: {digest}")
    raw = pd.read_csv(path)
    return pd.DataFrame({"timestamp": pd.to_datetime(raw["date"]), "open": raw["adj_open"].astype(float),
                         "close": raw["adjclose"].astype(float)})


def asset_passes(result: dict[str, Any]) -> dict[str, bool]:
    s, b = result["full"]["strategy"], result["full"]["buy_and_hold"]
    checks = {"sharpe_beats_buy_hold": s["sharpe"] > b["sharpe"],
              "drawdown_within_60pct_of_buy_hold": abs(s["max_drawdown"]) <= DRAWDOWN_RATIO_LIMIT * abs(b["max_drawdown"])}
    checks["passes"] = all(checks.values())
    return checks


def verdict(passes: dict[str, dict[str, bool]]) -> str:
    count = sum(p["passes"] for p in passes.values())
    return "XASSET_SUPPORTS_TREND" if count >= MIN_PASSING_ASSETS else "XASSET_DOES_NOT_SUPPORT"


def run() -> dict[str, Any]:
    results = {t: evaluate(load(t, sha), TRADING_DAYS) for t, sha in DATASETS.items()}
    passes = {t: asset_passes(r) for t, r in results.items()}
    return {"hypothesis_id": HYPOTHESIS_ID,
            "frozen": {"sma_lengths": list(SMA_LENGTHS), "fee_rate": FEE_RATE, "execution": "next_open",
                       "cash_yield": 0.0, "annualization_days": TRADING_DAYS,
                       "drawdown_ratio_limit": DRAWDOWN_RATIO_LIMIT, "min_passing_assets": MIN_PASSING_ASSETS},
            "datasets": {t: {"path": str(path_for(t)), "sha256": sha} for t, sha in DATASETS.items()},
            "results": results, "asset_checks": passes, "passing_assets": sum(p["passes"] for p in passes.values()),
            "verdict": verdict(passes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {args.output}")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "asset_checks": report["asset_checks"]}, indent=2))


if __name__ == "__main__":
    main()
