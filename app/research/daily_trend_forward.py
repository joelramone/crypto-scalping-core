"""Forward paper test of HYP-DAILY-TREND-BTC-FWD-001 (self-contained, re-runnable).

Downloads Binance spot daily klines (public data mirror, fallback to the main
API) from WARMUP_START to the last completed UTC day, applies the frozen rule
of HYP-DAILY-TREND-BTC-001, and evaluates only holding days from
FORWARD_START. Writes status.json and report.md under OUTPUT_DIR.

    python -m app.research.daily_trend_forward [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from app.research.daily_trend import FEE_RATE, SMA_LENGTHS, metrics, positions
from app.research.download_daily import DAY_MS, MAX_LIMIT, download_rows, rows_to_frame, validate_frame
from app.research.download_funding import to_ms

HYPOTHESIS_ID = "HYP-DAILY-TREND-BTC-FWD-001"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
WARMUP_START = "2026-03-01T00:00:00Z"
FORWARD_START = pd.Timestamp("2026-11-21T00:00:00Z")
VERDICT_DATE = pd.Timestamp("2028-11-21T00:00:00Z")
MIN_BENCHMARK_DRAWDOWN = 0.20
MIN_DAYS_FOR_SHARPE = 30
URLS = ("https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines")
OUTPUT_DIR = Path("research/forward/daily_trend_btc_fwd_v1")
TIMEOUT_SECONDS = 30


def fetch_page(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    params = {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": MAX_LIMIT}
    errors = []
    for url in URLS:
        for attempt in range(3):
            try:
                response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                raise ValueError(f"Unexpected response: {payload}")
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {exc}")
                time.sleep(2 ** attempt)
    raise RuntimeError("All Binance endpoints failed: " + " | ".join(errors[-2:]))


def load_market(symbol: str, end_ms: int, fetch: Callable[..., list[list[Any]]] = fetch_page) -> pd.DataFrame:
    start_ms = to_ms(WARMUP_START)
    frame = rows_to_frame(download_rows(symbol, start_ms, end_ms, fetch))
    validate_frame(frame, start_ms, end_ms)
    return frame


def forward_returns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strategy and buy-and-hold net returns for holding days j >= FORWARD_START.

    The position decided at close t is held open[t+1] -> open[t+2], labeled j = t+1.
    Both portfolios start flat, so each pays its entry fee on the first forward day.
    """
    opens = frame["open"].astype(float)
    open_to_open = opens.shift(-1) / opens - 1.0
    held = positions(frame["close"].astype(float)).shift(1)
    mask = (frame["timestamp"] >= FORWARD_START) & open_to_open.notna()
    if mask.any() and held[mask].isna().any():
        raise ValueError("Warm-up history is too short for the SMAs")
    days = frame.index[mask]
    held_f = held.loc[days]
    trades = held_f.diff()
    if len(days):
        trades.iloc[0] = held_f.iloc[0]
    strategy = held_f * open_to_open.loc[days] - FEE_RATE * trades.abs()
    buy_hold = open_to_open.loc[days].copy()
    if len(days):
        buy_hold.iloc[0] -= FEE_RATE
    stamps = frame["timestamp"].loc[days]
    for series in (strategy, buy_hold, held_f):
        series.index = stamps
    return strategy, buy_hold, held_f


def summary(returns: pd.Series) -> dict[str, Any] | None:
    if returns.empty:
        return None
    result = metrics(returns) if len(returns) > 1 else {
        "days": 1, "total_return": float(returns.iloc[0]), "max_drawdown": float(min(returns.iloc[0], 0.0))}
    if len(returns) < MIN_DAYS_FOR_SHARPE:
        result["sharpe"] = None
        result["cagr"] = None
        result["calmar"] = None
        result["note"] = f"Sharpe and CAGR not reported before {MIN_DAYS_FOR_SHARPE} days"
    return result


def verdict(as_of: pd.Timestamp, results: dict[str, Any]) -> str:
    if as_of < VERDICT_DATE:
        return "FORWARD_IN_PROGRESS"
    btc, eth = results["BTCUSDT"], results["ETHUSDT"]
    if abs(btc["buy_and_hold"]["max_drawdown"]) < MIN_BENCHMARK_DRAWDOWN:
        return "FORWARD_INCONCLUSIVE"
    ok = (btc["strategy"]["sharpe"] > btc["buy_and_hold"]["sharpe"]
          and abs(btc["strategy"]["max_drawdown"]) <= abs(btc["buy_and_hold"]["max_drawdown"])
          and eth["strategy"]["sharpe"] > eth["buy_and_hold"]["sharpe"])
    return "FORWARD_CONFIRMED" if ok else "FORWARD_REJECTED"


def check_no_revisions(previous: dict[str, Any] | None, daily: dict[str, list[dict[str, Any]]]) -> None:
    if not previous:
        return
    for symbol, rows in previous.get("daily", {}).items():
        new = {r["date"]: r for r in daily.get(symbol, [])}
        for row in rows:
            match = new.get(row["date"])
            if match is None:
                raise ValueError(f"{symbol} {row['date']} disappeared from the forward record")
            for key in ("strategy", "buy_and_hold", "position"):
                if not math.isclose(match[key], row[key], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{symbol} {row['date']} {key} was revised: {row[key]} -> {match[key]}")


def run(as_of: pd.Timestamp, fetch: Callable[..., list[list[Any]]] = fetch_page,
        output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    end_ms = int(as_of.value // 1_000_000)
    results: dict[str, Any] = {}
    daily: dict[str, list[dict[str, Any]]] = {}
    for symbol in SYMBOLS:
        frame = load_market(symbol, end_ms, fetch)
        strategy, buy_hold, held = forward_returns(frame)
        latest_signal = positions(frame["close"].astype(float)).iloc[-1]
        results[symbol] = {
            "strategy": summary(strategy), "buy_and_hold": summary(buy_hold),
            "current_position_for_next_open": float(latest_signal),
            "last_close_date": str(frame["timestamp"].iloc[-1].date()),
            "average_position": float(held.mean()) if len(held) else None,
        }
        daily[symbol] = [{"date": str(ts.date()), "position": float(p), "strategy": float(s), "buy_and_hold": float(b)}
                         for ts, p, s, b in zip(strategy.index, held, strategy, buy_hold)]
    status_path = output_dir / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    check_no_revisions(previous, daily)
    forward_days = len(daily["BTCUSDT"])
    status = {
        "hypothesis_id": HYPOTHESIS_ID, "as_of_exclusive": str(as_of.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forward_start": str(FORWARD_START.date()), "verdict_date": str(VERDICT_DATE.date()),
        "forward_days": forward_days, "frozen": {"sma_lengths": list(SMA_LENGTHS), "fee_rate": FEE_RATE},
        "results": results, "verdict": verdict(as_of, results) if forward_days else "FORWARD_IN_PROGRESS",
        "daily": daily,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(status), encoding="utf-8")
    return status


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_report(status: dict[str, Any]) -> str:
    lines = [
        "# Daily Trend BTC — Forward Paper Test",
        "",
        f"- Hypothesis: `{status['hypothesis_id']}` "
        "([protocol](../../hypotheses/daily_trend_btc_forward_v1.md))",
        f"- Data through: {status['as_of_exclusive']} (exclusive); forward start {status['forward_start']}; "
        f"verdict date {status['verdict_date']}",
        f"- Forward days evaluated: {status['forward_days']}",
        f"- **Verdict: `{status['verdict']}`**",
        "",
        "Paper only. Interim numbers are descriptive and must not change the rule or the criteria.",
        "",
        "| Market | Portfolio | Total return | CAGR | Sharpe | Max drawdown | Position for next open |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for symbol, r in status["results"].items():
        for name in ("strategy", "buy_and_hold"):
            m = r[name] or {}
            position = num(r["current_position_for_next_open"]) if name == "strategy" else "1.00"
            lines.append(f"| {symbol} | {name.replace('_', ' ')} | {pct(m.get('total_return'))} | {pct(m.get('cagr'))} | "
                         f"{num(m.get('sharpe'))} | {pct(m.get('max_drawdown'))} | {position} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="Exclusive end date (UTC, YYYY-MM-DD). Default: today UTC.")
    args = parser.parse_args()
    today = pd.Timestamp.now(tz="UTC").normalize()
    as_of = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else today
    if as_of > today:
        raise SystemExit("--as-of cannot be in the future")
    status = run(as_of)
    print(json.dumps({"verdict": status["verdict"], "forward_days": status["forward_days"],
                      "results": {s: {"strategy_total": (r["strategy"] or {}).get("total_return"),
                                      "buy_hold_total": (r["buy_and_hold"] or {}).get("total_return"),
                                      "position_for_next_open": r["current_position_for_next_open"]}
                                  for s, r in status["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
