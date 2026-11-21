"""Owner-authorized download of daily adjusted ETF prices from Yahoo Finance (TASK-036B).

Unofficial public endpoint: results can be revised, so each file is hashed in
its manifest and consumers must verify the hash. No-overwrite publication.

    python -m app.research.download_yahoo --dry-run
    python -m app.research.download_yahoo --execute-authorized-download
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from app.research.download_funding import manifest_path, publish, to_ms

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = "Mozilla/5.0 (research-data-download)"
TIMEOUT_SECONDS = 30
RETRIES = 4
MAX_CALENDAR_GAP_DAYS = 6
MIN_ROWS_PER_YEAR = 245
CSV_COLUMNS = ["date", "open", "high", "low", "close", "adjclose", "volume", "adj_open"]

START = "2005-01-01T00:00:00Z"
END = "2026-11-01T00:00:00Z"
AUTHORIZED_SCOPES: dict[str, list[tuple[str, Path]]] = {
    "TASK-036B": [(t, Path(f"data/yahoo_{t}_1d_20050101_20261101.csv")) for t in ("SPY", "EFA", "TLT", "GLD")],
}
SCOPE = "TASK-036B"


def fetch_chart(ticker: str, start_s: int, end_s: int) -> dict[str, Any]:
    params = {"period1": start_s, "period2": end_s, "interval": "1d", "events": "div,split",
              "includeAdjustedClose": "true"}
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(CHART_URL.format(ticker=ticker), params=params,
                                    headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            result = payload["chart"]["result"]
            if not result:
                raise ValueError(f"Empty chart result: {payload['chart'].get('error')}")
            return result[0]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Yahoo chart request failed for {ticker}") from last_error


def chart_to_frame(chart: dict[str, Any]) -> pd.DataFrame:
    quote = chart["indicators"]["quote"][0]
    adj = chart["indicators"]["adjclose"][0]["adjclose"]
    tz = chart["meta"].get("exchangeTimezoneName", "America/New_York")
    dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_convert(tz).date
    frame = pd.DataFrame({"date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
                          "open": quote["open"], "high": quote["high"], "low": quote["low"],
                          "close": quote["close"], "adjclose": adj, "volume": quote["volume"]})
    frame["adj_open"] = frame["open"] * frame["adjclose"] / frame["close"]
    return frame[CSV_COLUMNS]


def validate_frame(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("No rows returned")
    dates = pd.to_datetime(frame["date"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if dates.iloc[0] < pd.Timestamp(start[:10]) or dates.iloc[-1] >= pd.Timestamp(end[:10]):
        raise ValueError("Rows outside the authorized window")
    prices = frame[["open", "high", "low", "close", "adjclose", "adj_open"]]
    if prices.isna().any().any() or not prices.map(math.isfinite).all().all() or not (prices > 0).all().all():
        raise ValueError("Prices must be present, finite and strictly positive")
    if ((frame["high"] < frame[["open", "close"]].max(axis=1) * (1 - 1e-6)) |
            (frame["low"] > frame[["open", "close"]].min(axis=1) * (1 + 1e-6))).any():
        raise ValueError("High/low inconsistent with open/close")
    gaps = dates.diff().dt.days.iloc[1:]
    if (gaps > MAX_CALENDAR_GAP_DAYS).any():
        raise ValueError(f"Calendar gap over {MAX_CALENDAR_GAP_DAYS} days")
    if (dates.iloc[0] - pd.Timestamp(start[:10])).days > 7 or (pd.Timestamp(end[:10]) - dates.iloc[-1]).days > 7:
        raise ValueError("Coverage does not reach the window edges")
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if len(frame) < MIN_ROWS_PER_YEAR * years:
        raise ValueError("Too few trading days for the window")
    return {"row_count": int(len(frame)), "first_date": str(dates.iloc[0].date()),
            "last_date": str(dates.iloc[-1].date()), "max_calendar_gap_days": int(gaps.max())}


def plan(scope: str = SCOPE) -> dict[str, Any]:
    return {"scope": scope, "url": CHART_URL, "start": START, "end_exclusive": END,
            "targets": [{"ticker": t, "output": str(o)} for t, o in AUTHORIZED_SCOPES[scope]]}


def execute(scope: str = SCOPE, fetch: Callable[[str, int, int], dict[str, Any]] = fetch_chart,
            start: str = START, end: str = END) -> list[dict[str, Any]]:
    targets = AUTHORIZED_SCOPES[scope]
    if (Path.cwd() / "data").is_symlink():
        raise ValueError("data/ must not be a symlink")
    for _, output in targets:
        if output.exists() or manifest_path(output).exists():
            raise FileExistsError(f"Refusing to overwrite {output} or its manifest")
    start_s, end_s = to_ms(start) // 1000, to_ms(end) // 1000
    manifests = []
    for ticker, output in targets:
        frame = chart_to_frame(fetch(ticker, start_s, end_s))
        frame = frame[frame["date"] < end[:10]].reset_index(drop=True)
        checks = validate_frame(frame, start, end)
        metadata = {"status": "DATASET_VALIDATED", "authorization_scope": scope, "source": "yahoo_finance_chart_api",
                    "source_url": CHART_URL.format(ticker=ticker), "ticker": ticker, "interval": "1d",
                    "start_inclusive": start, "end_exclusive": end, "csv_path": str(output),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                    "semantics": {"adjclose": "split- and dividend-adjusted close",
                                  "adj_open": "open * adjclose / close", "date": "exchange-local trading date"},
                    "limitations": "unofficial endpoint; values can be revised retroactively",
                    "integrity_checks": checks}
        publish(frame, output, metadata)
        manifests.append(json.loads(manifest_path(output).read_text()))
        time.sleep(1)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-authorized-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan() if args.dry_run else execute(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
