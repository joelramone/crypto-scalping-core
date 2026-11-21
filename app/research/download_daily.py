"""Owner-authorized download of Binance spot daily klines (TASK-035B).

Only the exact authorized scopes can execute. Each CSV and its manifest are
published without overwriting, reusing the funding downloader's publisher.

    python -m app.research.download_daily --dry-run
    python -m app.research.download_daily --execute-authorized-download
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

KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
DAY_MS = 86_400_000
REQUEST_SLEEP_SECONDS = 0.3
TIMEOUT_SECONDS = 30
RETRIES = 4
CSV_COLUMNS = ["timestamp", "open_time_ms", "open", "high", "low", "close", "volume", "quote_volume", "trades"]

START = "2017-08-17T00:00:00Z"
END = "2026-11-01T00:00:00Z"
# Exact owner-authorized scopes: scope -> list of (symbol, output).
AUTHORIZED_SCOPES: dict[str, list[tuple[str, Path]]] = {
    "TASK-035B": [
        ("BTCUSDT", Path("data/binance_spot_BTCUSDT_1d_20170817_20261101.csv")),
        ("ETHUSDT", Path("data/binance_spot_ETHUSDT_1d_20170817_20261101.csv")),
    ],
}
SCOPE = "TASK-035B"


def fetch_page(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    params = {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": MAX_LIMIT}
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(KLINES_URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected Binance response: {payload}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Binance klines request failed after {RETRIES} attempts") from last_error


def download_rows(symbol: str, start_ms: int, end_ms: int,
                  fetch: Callable[..., list[list[Any]]] = fetch_page) -> list[list[Any]]:
    rows: list[list[Any]] = []
    cursor = start_ms
    with requests.Session() as session:
        while cursor < end_ms:
            page = fetch(session, symbol, cursor, end_ms - 1)
            if not page:
                break
            rows.extend(page)
            last = int(page[-1][0])
            if last < cursor:
                raise ValueError("Binance pagination moved backwards")
            cursor = last + DAY_MS
            if len(page) < MAX_LIMIT:
                break
            time.sleep(REQUEST_SLEEP_SECONDS)
    return rows


def rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    raw = pd.DataFrame([r[:9] for r in rows], columns=["open_time_ms", "open", "high", "low", "close", "volume",
                                                       "close_time_ms", "quote_volume", "trades"])
    out = raw.drop(columns="close_time_ms")
    out["open_time_ms"] = out["open_time_ms"].astype("int64")
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce")
    out.insert(0, "timestamp", pd.to_datetime(out["open_time_ms"], unit="ms", utc=True))
    return out[CSV_COLUMNS]


def validate_frame(frame: pd.DataFrame, start_ms: int, end_ms: int) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("No daily rows returned")
    times = frame["open_time_ms"]
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Daily open times must be unique and sorted")
    if times.iloc[0] != start_ms or times.iloc[-1] != end_ms - DAY_MS:
        raise ValueError("Daily rows do not cover the exact authorized scope")
    if not (times.diff().iloc[1:] == DAY_MS).all():
        raise ValueError("Daily rows must be contiguous with exact 1-day spacing")
    prices = frame[["open", "high", "low", "close"]]
    if not prices.map(math.isfinite).all().all() or not (prices > 0).all().all():
        raise ValueError("OHLC must be finite and strictly positive")
    if ((frame["high"] < prices.max(axis=1)) | (frame["low"] > prices.min(axis=1))).any():
        raise ValueError("High/low inconsistent with open/close")
    if not frame["volume"].map(math.isfinite).all() or (frame["volume"] < 0).any():
        raise ValueError("Volume must be finite and non-negative")
    expected = (end_ms - start_ms) // DAY_MS
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(frame)}")
    return {"row_count": int(len(frame)), "expected_rows": int(expected), "spacing_ms": DAY_MS,
            "zero_volume_days": int((frame["volume"] == 0).sum())}


def plan(scope: str = SCOPE) -> dict[str, Any]:
    return {"scope": scope, "url": KLINES_URL, "interval": "1d", "start_inclusive_utc": START,
            "end_exclusive_utc": END, "expected_rows_per_symbol": (to_ms(END) - to_ms(START)) // DAY_MS,
            "targets": [{"symbol": s, "output": str(o)} for s, o in AUTHORIZED_SCOPES[scope]]}


def execute(scope: str = SCOPE, fetch: Callable[..., list[list[Any]]] = fetch_page,
            start: str = START, end: str = END) -> list[dict[str, Any]]:
    targets = AUTHORIZED_SCOPES[scope]
    if (Path.cwd() / "data").is_symlink():
        raise ValueError("data/ must not be a symlink")
    for _, output in targets:
        if output.exists() or manifest_path(output).exists():
            raise FileExistsError(f"Refusing to overwrite {output} or its manifest")
    start_ms, end_ms = to_ms(start), to_ms(end)
    manifests = []
    for symbol, output in targets:
        started = time.perf_counter()
        frame = rows_to_frame(download_rows(symbol, start_ms, end_ms, fetch))
        checks = validate_frame(frame, start_ms, end_ms)
        metadata = {
            "status": "DATASET_VALIDATED", "authorization_scope": scope, "source": "binance",
            "product": "spot", "symbol": symbol, "interval": "1d", "source_url": KLINES_URL,
            "start_inclusive_utc": start, "end_exclusive_utc": end, "csv_path": str(output),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition_duration_seconds": time.perf_counter() - started,
            "semantics": {"timestamp": "candle open time, UTC", "volume": "base asset",
                          "quote_volume": "USDT"},
            "integrity_checks": checks,
        }
        publish(frame, output, metadata)
        manifests.append(json.loads(manifest_path(output).read_text()))
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
