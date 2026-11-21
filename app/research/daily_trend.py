"""Frozen HYP-DAILY-TREND-BTC-001: long/flat daily trend following vs buy-and-hold.

See research/hypotheses/daily_trend_btc_v1.md. Every constant here is
pre-registered; changing any of them after seeing results is prohibited.

    python -m app.research.daily_trend --output <json path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HYPOTHESIS_ID = "HYP-DAILY-TREND-BTC-001"
SMA_LENGTHS = (50, 100, 200)
WARMUP = max(SMA_LENGTHS) - 1
FEE_RATE = 0.001
DAYS_PER_YEAR = 365
DRAWDOWN_RATIO_LIMIT = 0.6
DATASETS = {
    "BTCUSDT": (Path("data/binance_spot_BTCUSDT_1d_20170817_20261101.csv"),
                "df09a7787ce4d3bc025213b7a79d3b0e56bfb7135e2a8f9ceaca5e531100b925"),
    "ETHUSDT": (Path("data/binance_spot_ETHUSDT_1d_20170817_20261101.csv"),
                "ea1cbef7312ec82476904cd4243b9a871b9159f5e4e7b7b810c763036e63c308"),
}


def load(path: Path, sha256: str) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} SHA-256 mismatch: {digest}")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    return frame.reset_index(drop=True)


def positions(close: pd.Series) -> pd.Series:
    """Position decided after the close of day t: mean of three close > SMA votes."""
    votes = [(close > close.rolling(n, min_periods=n).mean()).astype(float) for n in SMA_LENGTHS]
    pos = sum(votes) / len(SMA_LENGTHS)
    pos[close.rolling(max(SMA_LENGTHS), min_periods=max(SMA_LENGTHS)).mean().isna()] = np.nan
    return pos


def daily_returns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (strategy, buy_and_hold, position) net returns indexed by holding day j.

    The position decided at close t is held from open[t+1] to open[t+2]; that
    holding period is labeled j = t+1. Evaluation days run from WARMUP+1 to N-2.
    """
    opens = frame["open"].astype(float)
    pos = positions(frame["close"].astype(float))
    open_to_open = opens.shift(-1) / opens - 1.0              # labeled by j: open[j] -> open[j+1]
    held = pos.shift(1)                                        # pos decided at t = j-1
    trades = held.diff().abs()
    trades.iloc[WARMUP + 1] = abs(held.iloc[WARMUP + 1])        # first entry from flat
    days = pd.RangeIndex(WARMUP + 1, len(frame) - 1)
    strategy = (held * open_to_open - FEE_RATE * trades).loc[days]
    buy_hold = open_to_open.loc[days].copy()
    buy_hold.iloc[0] -= FEE_RATE
    stamps = frame["timestamp"].loc[days]
    strategy.index = buy_hold.index = stamps
    held = held.loc[days]
    held.index = stamps
    return strategy, buy_hold, held


def metrics(returns: pd.Series, days_per_year: int = DAYS_PER_YEAR) -> dict[str, Any]:
    equity = (1.0 + returns).cumprod()
    n = len(returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])
    drawdown = np.r_[1.0, equity.to_numpy()] / peak - 1.0
    std = returns.std(ddof=1)
    cagr = equity.iloc[-1] ** (days_per_year / n) - 1.0
    max_dd = float(drawdown.min())
    return {
        "days": n, "start": str(returns.index[0]), "end": str(returns.index[-1]),
        "total_return": float(equity.iloc[-1] - 1.0), "cagr": float(cagr),
        "annualized_volatility": float(std * math.sqrt(days_per_year)),
        "sharpe": float(returns.mean() / std * math.sqrt(days_per_year)) if std > 0 else None,
        "max_drawdown": max_dd, "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else None,
    }


def evaluate(frame: pd.DataFrame, days_per_year: int = DAYS_PER_YEAR) -> dict[str, Any]:
    strategy, buy_hold, held = daily_returns(frame)
    mid = len(strategy) // 2
    windows = {"full": slice(None), "first_half": slice(0, mid), "second_half": slice(mid, None)}
    result: dict[str, Any] = {name: {"strategy": metrics(strategy.iloc[s], days_per_year),
                                     "buy_and_hold": metrics(buy_hold.iloc[s], days_per_year)}
                              for name, s in windows.items()}
    changes = held.diff().fillna(held.iloc[0]).abs()
    result["activity"] = {"time_in_market": float(held.mean()), "position_changes": int((changes > 0).sum()),
                          "total_fees_fraction_traded": float(changes.sum() * FEE_RATE)}
    years = strategy.index.year
    result["calendar_years"] = {
        str(y): {"strategy": float((1 + strategy[years == y]).prod() - 1),
                 "buy_and_hold": float((1 + buy_hold[years == y]).prod() - 1),
                 "average_position": float(held[years == y].mean())}
        for y in sorted(set(years))
    }
    return result


def verdict(btc: dict[str, Any], eth: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    s, b = btc["full"]["strategy"], btc["full"]["buy_and_hold"]
    checks = {
        "btc_full_cagr_positive": s["cagr"] > 0,
        "btc_full_sharpe_beats_buy_hold": s["sharpe"] > b["sharpe"],
        "btc_full_drawdown_within_60pct_of_buy_hold":
            abs(s["max_drawdown"]) <= DRAWDOWN_RATIO_LIMIT * abs(b["max_drawdown"]),
        "btc_first_half_sharpe_beats_buy_hold":
            btc["first_half"]["strategy"]["sharpe"] > btc["first_half"]["buy_and_hold"]["sharpe"],
        "btc_second_half_sharpe_beats_buy_hold":
            btc["second_half"]["strategy"]["sharpe"] > btc["second_half"]["buy_and_hold"]["sharpe"],
        "eth_full_sharpe_beats_buy_hold": eth["full"]["strategy"]["sharpe"] > eth["full"]["buy_and_hold"]["sharpe"],
    }
    return ("TREND_CANDIDATE" if all(checks.values()) else "TREND_REJECT"), checks


def run() -> dict[str, Any]:
    results = {symbol: evaluate(load(path, sha)) for symbol, (path, sha) in DATASETS.items()}
    label, checks = verdict(results["BTCUSDT"], results["ETHUSDT"])
    return {"hypothesis_id": HYPOTHESIS_ID,
            "frozen": {"sma_lengths": list(SMA_LENGTHS), "fee_rate": FEE_RATE, "execution": "next_open",
                       "cash_yield": 0.0, "warmup_days": WARMUP},
            "datasets": {s: {"path": str(p), "sha256": h} for s, (p, h) in DATASETS.items()},
            "results": results, "verdict_checks": checks, "verdict": label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {args.output}")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "checks": report["verdict_checks"]}, indent=2))


if __name__ == "__main__":
    main()
