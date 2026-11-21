"""Descriptive economics of a delta-neutral BTCUSDT funding carry on Binance.

Position: long spot BTC + short USD-M BTCUSDT perpetual, equal notional, held
continuously. A positive funding rate means longs pay shorts, so the short
perpetual leg receives `rate * notional` at each settlement.

This is not a strategy backtest: there are no entry signals or tunable
parameters. Basis PnL, spot/perp price divergence, slippage and funding on
the margin balance are not modeled; see the report for limitations.

    python -m app.research.funding_carry_economics --output <json path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATASET = Path("data/binance_usdm_BTCUSDT_funding_20190901_20261101.csv")
DATASET_SHA256 = "1b1779e108fe23ab5c56e74aa6c24855c8d4400fa56fb4efbe20aeeecb167e8b"
SETTLEMENTS_PER_DAY = 3

# Fee assumptions (Binance regular user VIP0, per fill, as fractions). Not verified
# against the user's account tier; the report shows each scenario separately.
FEE_SCENARIOS: dict[str, dict[str, float]] = {
    "taker_all_fills": {"spot": 0.0010, "perp": 0.0005},
    "maker_all_fills_bnb_discount": {"spot": 0.00075, "perp": 0.00018},
}
LEVERAGES = (1, 2, 3)


def load(path: Path = DATASET) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATASET_SHA256:
        raise SystemExit(f"Funding dataset SHA-256 mismatch: {digest}")
    frame = pd.read_csv(path)
    frame["funding_time"] = pd.to_datetime(frame["funding_time_ms"], unit="ms", utc=True)
    return frame.sort_values("funding_time").reset_index(drop=True)


def round_trip_cost(fees: dict[str, float]) -> float:
    """Four fills: buy spot, sell perp at entry; sell spot, buy perp at exit."""
    return 2 * fees["spot"] + 2 * fees["perp"]


def capital_per_notional(leverage: int) -> float:
    """Spot leg fully paid plus perp initial margin."""
    return 1.0 + 1.0 / leverage


def longest_negative_streak(rates: pd.Series) -> int:
    best = run = 0
    for value in rates:
        run = run + 1 if value < 0 else 0
        best = max(best, run)
    return best


def period_stats(rates: pd.Series) -> dict[str, Any]:
    n = len(rates)
    window = 30 * SETTLEMENTS_PER_DAY
    rolling = rates.rolling(window).sum() if n >= window else pd.Series(dtype=float)
    return {
        "settlements": n,
        "sum_funding": float(rates.sum()),
        "annualized_funding": float(rates.mean() * SETTLEMENTS_PER_DAY * 365) if n else None,
        "mean_rate": float(rates.mean()) if n else None,
        "median_rate": float(rates.median()) if n else None,
        "fraction_positive": float((rates > 0).mean()) if n else None,
        "fraction_negative": float((rates < 0).mean()) if n else None,
        "longest_negative_streak_settlements": longest_negative_streak(rates),
        "worst_30d_sum": float(rolling.min()) if len(rolling.dropna()) else None,
        "best_30d_sum": float(rolling.max()) if len(rolling.dropna()) else None,
    }


def net_returns(sum_funding: float, years: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fees in FEE_SCENARIOS.items():
        net_on_notional = sum_funding - round_trip_cost(fees)
        out[name] = {
            "round_trip_cost": round_trip_cost(fees),
            "net_on_notional": net_on_notional,
            "annualized_net_on_notional": net_on_notional / years if years > 0 else None,
            "annualized_on_capital": {
                f"{lev}x": (net_on_notional / years) / capital_per_notional(lev) if years > 0 else None
                for lev in LEVERAGES
            },
        }
    return out


def rolling_365d(frame: pd.DataFrame) -> dict[str, Any]:
    window = 365 * SETTLEMENTS_PER_DAY
    sums = frame["funding_rate"].rolling(window).sum().dropna()
    costs = {name: round_trip_cost(f) for name, f in FEE_SCENARIOS.items()}
    return {
        "windows": int(len(sums)),
        "min": float(sums.min()), "p10": float(sums.quantile(0.10)), "median": float(sums.median()),
        "p90": float(sums.quantile(0.90)), "max": float(sums.max()),
        "fraction_below_zero": float((sums < 0).mean()),
        "fraction_below_round_trip_cost": {name: float((sums < cost).mean()) for name, cost in costs.items()},
        "worst_window_end": str(frame.loc[sums.idxmin(), "funding_time"]),
        "best_window_end": str(frame.loc[sums.idxmax(), "funding_time"]),
    }


def breakeven_days(annualized_funding: float) -> dict[str, float | None]:
    daily = annualized_funding / 365
    return {name: (round_trip_cost(f) / daily if daily > 0 else None) for name, f in FEE_SCENARIOS.items()}


def analyze(frame: pd.DataFrame) -> dict[str, Any]:
    rates = frame["funding_rate"]
    t0, t1 = frame["funding_time"].iat[0], frame["funding_time"].iat[-1]
    total_years = (t1 - t0).total_seconds() / (365 * 86400)
    by_year: dict[str, Any] = {}
    for year, group in frame.groupby(frame["funding_time"].dt.year):
        stats = period_stats(group["funding_rate"])
        span_years = len(group) / (SETTLEMENTS_PER_DAY * 365)
        stats["net_if_held_whole_year"] = net_returns(stats["sum_funding"], span_years)
        by_year[str(year)] = stats

    recent: dict[str, Any] = {}
    for label, days in (("last_365d", 365), ("last_90d", 90)):
        cutoff = t1 - pd.to_timedelta(days, unit="D")
        part = frame[frame["funding_time"] > cutoff]["funding_rate"]
        stats = period_stats(part)
        stats["net_if_held"] = net_returns(stats["sum_funding"], days / 365)
        stats["breakeven_days_at_this_rate"] = breakeven_days(stats["annualized_funding"])
        recent[label] = stats

    whole = period_stats(rates)
    whole["net_if_held"] = net_returns(whole["sum_funding"], total_years)
    return {
        "dataset": {"path": str(DATASET), "sha256": DATASET_SHA256, "first": str(t0), "last": str(t1),
                    "settlements": int(len(frame)), "years": total_years},
        "assumptions": {"fee_scenarios": FEE_SCENARIOS, "leverages": list(LEVERAGES),
                        "capital_per_notional": {f"{lev}x": capital_per_notional(lev) for lev in LEVERAGES},
                        "not_modeled": ["basis PnL at entry/exit", "slippage and spread", "margin top-ups",
                                        "liquidation of the short leg", "interest on idle collateral",
                                        "taxes", "ARS/USDT conversion costs"]},
        "whole_period": whole,
        "by_year": by_year,
        "recent": recent,
        "rolling_365d_funding_sum": rolling_365d(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(load())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
