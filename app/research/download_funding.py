"""Owner-authorized download of Binance USD-M funding-rate history (TASK-034B).

Only the exact authorized scope can execute. Output and manifest are published
without overwriting, with the same hard-link pattern as download_data.py.

    python -m app.research.download_funding --dry-run
    python -m app.research.download_funding --execute-authorized-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
MAX_LIMIT = 1000
REQUEST_SLEEP_SECONDS = 0.3
TIMEOUT_SECONDS = 30
RETRIES = 4
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
SPACING_TOLERANCE_MS = 60_000
CSV_COLUMNS = ["funding_time", "funding_time_ms", "symbol", "funding_rate", "mark_price"]

# Exact owner-authorized scope: (symbol, start inclusive, end exclusive, output).
AUTHORIZED_SCOPES: dict[str, tuple[str, str, str, Path]] = {
    "TASK-034B": ("BTCUSDT", "2019-09-01T00:00:00Z", "2026-11-01T00:00:00Z",
                  Path("data/binance_usdm_BTCUSDT_funding_20190901_20261101.csv")),
}
SCOPE = "TASK-034B"


def to_ms(value: str) -> int:
    return int(pd.Timestamp(value).tz_convert("UTC").value // 1_000_000)


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def fetch_page(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": MAX_LIMIT}
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(FUNDING_URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected Binance response: {payload}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Binance funding request failed after {RETRIES} attempts") from last_error


def download_rows(symbol: str, start_ms: int, end_ms: int,
                  fetch: Callable[..., list[dict[str, Any]]] = fetch_page) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    with requests.Session() as session:
        while cursor < end_ms:
            page = fetch(session, symbol, cursor, end_ms - 1)
            if not page:
                break
            rows.extend(page)
            last = int(page[-1]["fundingTime"])
            if last < cursor:
                raise ValueError("Binance pagination moved backwards")
            cursor = last + 1
            if len(page) < MAX_LIMIT:
                break
            time.sleep(REQUEST_SLEEP_SECONDS)
    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)
    out = pd.DataFrame({
        "funding_time_ms": frame["fundingTime"].astype("int64"),
        "symbol": frame["symbol"].astype(str),
        "funding_rate": pd.to_numeric(frame["fundingRate"], errors="coerce"),
        "mark_price": pd.to_numeric(frame.get("markPrice"), errors="coerce") if "markPrice" in frame else float("nan"),
    })
    out.insert(0, "funding_time", pd.to_datetime(out["funding_time_ms"], unit="ms", utc=True))
    return out[CSV_COLUMNS]


def validate_frame(frame: pd.DataFrame, symbol: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("No funding rows returned")
    times = frame["funding_time_ms"]
    if times.duplicated().any():
        raise ValueError("Duplicate funding times")
    if not times.is_monotonic_increasing:
        raise ValueError("Funding times are not sorted")
    if times.iloc[0] < start_ms or times.iloc[-1] >= end_ms:
        raise ValueError("Funding times outside the authorized scope")
    if (frame["symbol"] != symbol).any():
        raise ValueError("Unexpected symbol in response")
    if not frame["funding_rate"].map(math.isfinite).all():
        raise ValueError("Funding rates must be finite")
    if (frame["funding_rate"].abs() >= 0.05).any():
        raise ValueError("Implausible funding rate magnitude (>= 5% per settlement)")
    spacing = times.diff().iloc[1:]
    gaps = spacing[spacing > EIGHT_HOURS_MS + SPACING_TOLERANCE_MS]
    if not gaps.empty:
        raise ValueError(f"{len(gaps)} gaps longer than 8h between settlements")
    return {
        "row_count": int(len(frame)), "unique_times": int(times.nunique()),
        "max_spacing_ms": int(spacing.max()) if len(spacing) else None,
        "min_spacing_ms": int(spacing.min()) if len(spacing) else None,
        "settlements_off_8h_grid": int((~((times % EIGHT_HOURS_MS) < SPACING_TOLERANCE_MS)).sum()),
        "missing_mark_price": int(frame["mark_price"].isna().sum()),
        "max_abs_rate": float(frame["funding_rate"].abs().max()),
    }


def publish(frame: pd.DataFrame, output: Path, metadata: dict[str, Any]) -> None:
    manifest = manifest_path(output)
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                         prefix=".funding-", suffix=".tmp", delete=False) as handle:
            csv_temp = Path(handle.name)
            temporary.append(csv_temp)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        metadata = metadata | {"file_size_bytes": csv_temp.stat().st_size,
                               "sha256": hashlib.sha256(csv_temp.read_bytes()).hexdigest()}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                         prefix=".funding-", suffix=".tmp", delete=False) as handle:
            json_temp = Path(handle.name)
            temporary.append(json_temp)
            json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(csv_temp, output)
        published.append(output)
        os.link(json_temp, manifest)
        published.append(manifest)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def plan(scope: str = SCOPE) -> dict[str, Any]:
    symbol, start, end, output = AUTHORIZED_SCOPES[scope]
    return {"scope": scope, "symbol": symbol, "url": FUNDING_URL, "start_inclusive_utc": start,
            "end_exclusive_utc": end, "output": str(output), "method": "GET", "limit": MAX_LIMIT}


def execute(scope: str = SCOPE, fetch: Callable[..., list[dict[str, Any]]] = fetch_page) -> dict[str, Any]:
    symbol, start, end, output = AUTHORIZED_SCOPES[scope]
    if (Path.cwd() / "data").is_symlink():
        raise ValueError("data/ must not be a symlink")
    start_ms, end_ms = to_ms(start), to_ms(end)
    started = time.perf_counter()
    frame = rows_to_frame(download_rows(symbol, start_ms, end_ms, fetch))
    checks = validate_frame(frame, symbol, start_ms, end_ms)
    metadata = {
        "status": "DATASET_VALIDATED", "authorization_scope": scope, "source": "binance",
        "product": "usdm-futures", "symbol": symbol, "source_url": FUNDING_URL,
        "start_inclusive_utc": start, "end_exclusive_utc": end,
        "first_funding_time_utc": frame["funding_time"].iat[0].isoformat(),
        "last_funding_time_utc": frame["funding_time"].iat[-1].isoformat(),
        "csv_path": str(output), "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_duration_seconds": time.perf_counter() - started,
        "semantics": {"funding_rate": "settled rate per settlement; positive means longs pay shorts",
                      "funding_time": "settlement time, UTC", "mark_price": "as returned; may be empty for old rows"},
        "integrity_checks": checks,
    }
    publish(frame, output, metadata)
    return json.loads(manifest_path(output).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-authorized-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan() if args.dry_run else execute(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
