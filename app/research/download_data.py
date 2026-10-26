from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator, model_validator

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_LIMIT = 1500
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30
REQUEST_SLEEP_SECONDS = 0.2
AUTHORIZED_START = "2024-10-01T00:00:00Z"
AUTHORIZED_END = "2024-10-08T00:00:00Z"
AUTHORIZED_OUTPUT = Path("data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv")

INTERVAL_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}

CSV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


class DownloadConfig(BaseModel):
    symbol: str = Field(min_length=1)
    interval: str
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    output: Path
    retries: int = Field(default=DEFAULT_RETRIES, ge=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, value: str) -> str:
        if value not in INTERVAL_TO_MS:
            valid_intervals = ", ".join(INTERVAL_TO_MS)
            raise ValueError(f"Unsupported interval '{value}'. Valid intervals: {valid_intervals}")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> DownloadConfig:
        if parse_datetime_to_utc_ms(self.end) <= parse_datetime_to_utc_ms(self.start):
            raise ValueError("--end must be later than --start")
        return self


def parse_datetime_to_utc_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO date/datetime '{value}'. Example: 2025-01-01 or 2025-01-01T00:00:00+00:00"
        ) from exc

    if parsed.tzinfo is None:
        if len(value) != 10:
            raise ValueError("Datetime must include an explicit UTC offset")
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    if parsed.microsecond % 1000:
        raise ValueError("Sub-millisecond date precision is unsupported")
    return int(parsed.timestamp() * 1000)


def fetch_klines_page(
    session: requests.Session,
    config: DownloadConfig,
    start_time_ms: int,
    end_time_ms: int,
) -> list[list[Any]]:
    params: dict[str, Any] = {
        "symbol": config.symbol,
        "interval": config.interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": MAX_LIMIT,
    }

    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            response = session.get(BINANCE_FUTURES_KLINES_URL, params=params, timeout=config.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected Binance response: {payload}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= config.retries:
                break
            sleep_seconds = min(2 ** (attempt - 1), 8)
            logging.warning("Binance request failed on attempt %s/%s: %s", attempt, config.retries, exc)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Binance request failed after {config.retries} attempts") from last_error


def rows_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)

    dataframe = pd.DataFrame(
        (
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "close_time": row[6],
                "quote_volume": row[7],
                "trades": row[8],
                "taker_buy_base_volume": row[9],
                "taker_buy_quote_volume": row[10],
            }
            for row in rows
        ),
        columns=CSV_COLUMNS,
    )
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms", utc=True)
    dataframe["close_time"] = pd.to_datetime(dataframe["close_time"], unit="ms", utc=True)
    return dataframe


def validate_preparation(config: DownloadConfig) -> tuple[int, int]:
    """Fail closed on known research restrictions; not a research authorization."""
    start = parse_datetime_to_utc_ms(config.start)
    end = parse_datetime_to_utc_ms(config.end)
    if config.symbol != "BTCUSDT" or config.interval != "1m":
        raise ValueError("Preparation supports BTCUSDT 1m only")
    # Existing protocols: 2025 discovery-used, partial 2026 consumed,
    # and all dates after 2026-08-05 reserved. No release is inferred.
    if end > parse_datetime_to_utc_ms("2025-01-01"):
        raise ValueError("Interval overlaps consumed/reserved research dates (2025 onward)")
    if start % 60_000 or end % 60_000:
        raise ValueError("Start and exclusive end must align to whole UTC minutes")
    for destination in (config.output, manifest_path(config.output)):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing overwrite: {destination}")
    return start, end


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def validate_execution(config: DownloadConfig, exchange: str, product: str) -> None:
    start, end = validate_preparation(config)
    if (exchange != "binance" or product != "usdm-futures" or
            start != parse_datetime_to_utc_ms(AUTHORIZED_START) or
            end != parse_datetime_to_utc_ms(AUTHORIZED_END) or
            config.output.resolve() != (Path.cwd() / AUTHORIZED_OUTPUT).resolve() or
            (Path.cwd() / "data").is_symlink()):
        raise ValueError("Execution scope does not match the exact TASK-019B authorization")


def preparation_plan(config: DownloadConfig) -> dict[str, Any]:
    start, end = validate_preparation(config)
    stamp = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "mode": "dry_run", "download_authorized": False,
        "source": "binance", "product": "usdm-futures",
        "method": "GET", "url": BINANCE_FUTURES_KLINES_URL,
        "params": {"symbol": config.symbol, "interval": config.interval,
                   "startTime": start, "endTime": end - 1, "limit": MAX_LIMIT},
        "destination": str(config.output),
        "suggested_filename": f"binance_usdm_{config.symbol}_{config.interval}_{stamp(start)}_{stamp(end)}_exclusive.csv",
        "expected_candles": (end - start) // 60_000,
        "notice": "Technical preparation only; pre-2025 dates are not automatically authorized for research. Separate human download authorization required.",
    }


def validate_download_frame(dataframe: pd.DataFrame, start: int, end: int) -> None:
    from app.research.data_validation.validate_ohlcv import validate_ohlcv

    # Validator accepts serialized ISO timestamps, as in its CSV interface.
    validation = validate_ohlcv(dataframe.assign(timestamp=dataframe["timestamp"].astype(str)))
    if not validation.structurally_valid or dataframe.empty:
        raise ValueError(f"Invalid OHLCV: {validation.errors}")
    numeric = dataframe[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if not numeric.map(math.isfinite).all().all():
        raise ValueError("OHLCV must contain finite numeric values")
    expected_count = (end - start) // 60_000
    expected_start = pd.to_datetime(start, unit="ms", utc=True)
    expected_end = pd.to_datetime(end - 60_000, unit="ms", utc=True)
    if (len(dataframe) != expected_count or
            dataframe.iloc[0]["timestamp"] != expected_start or
            dataframe.iloc[-1]["timestamp"] != expected_end):
        raise ValueError("Missing candles or timestamps outside the requested interval")


def download_binance_futures_klines(config: DownloadConfig) -> pd.DataFrame:
    """Existing acquisition engine; CLI authorization is checked before entry."""
    acquisition_started = time.perf_counter()
    start_time_ms, exclusive_end_ms = validate_preparation(config)
    end_time_ms = exclusive_end_ms - 1
    interval_ms = INTERVAL_TO_MS[config.interval]
    current_start_ms = start_time_ms
    rows: list[list[Any]] = []

    with requests.Session() as session:
        while current_start_ms <= end_time_ms:
            page = fetch_klines_page(session, config, current_start_ms, end_time_ms)
            if not page:
                break
            rows.extend(page)
            last_open_time_ms = int(page[-1][0])
            next_start_ms = last_open_time_ms + interval_ms
            if next_start_ms <= current_start_ms:
                raise RuntimeError("Binance pagination did not advance")
            current_start_ms = next_start_ms
            # A short page is not proof that the bounded interval is complete.
            if current_start_ms <= end_time_ms:
                time.sleep(REQUEST_SLEEP_SECONDS)

    dataframe = rows_to_dataframe(rows)
    validate_download_frame(dataframe, start_time_ms, exclusive_end_ms)
    publish_validated_dataset(config, dataframe, acquisition_started)
    return dataframe


def publish_validated_dataset(config: DownloadConfig, dataframe: pd.DataFrame, started: float) -> None:
    """Stage both files, publish without replacement; manifest is the completion marker.

    Same-directory hard links provide atomic, no-overwrite publication on NTFS
    and POSIX filesystems. Unsupported filesystems fail closed. Two files cannot
    be committed atomically together: handled failures roll back our publications;
    after a process/power interruption, a CSV without a manifest is not success.
    """
    start, end = validate_preparation(config)
    validate_download_frame(dataframe, start, end)
    output, manifest = config.output, manifest_path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output.parent, prefix=".ohlcv-", suffix=".tmp", delete=False) as handle:
            csv_temp = Path(handle.name)
            temporary.append(csv_temp)
            dataframe.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        # Validate the actual serialized bytes before any final name is visible.
        serialized = pd.read_csv(csv_temp)
        serialized["timestamp"] = pd.to_datetime(serialized["timestamp"], utc=True, errors="raise")
        validate_download_frame(serialized, start, end)
        metadata = {
            "status": "DATASET_VALIDATED", "purpose": "technical ingestion and validation only",
            "source": "binance", "product": "usdm-futures", "source_url": BINANCE_FUTURES_KLINES_URL,
            "symbol": config.symbol, "timeframe": config.interval,
            "start_inclusive_utc": datetime.fromtimestamp(start / 1000, timezone.utc).isoformat(),
            "end_exclusive_utc": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
            "first_timestamp_utc": serialized.iloc[0]["timestamp"].isoformat(),
            "last_timestamp_utc": serialized.iloc[-1]["timestamp"].isoformat(),
            "row_count": len(serialized), "file_size_bytes": csv_temp.stat().st_size,
            "sha256": hashlib.sha256(csv_temp.read_bytes()).hexdigest(),
            "acquisition_duration_seconds": time.perf_counter() - started,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(output),
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output.parent, prefix=".ohlcv-", suffix=".tmp", delete=False) as handle:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute the exact locally authorized BTCUSDT acquisition.")
    parser.add_argument("--exchange", required=True, choices=["binance"])
    parser.add_argument("--product", required=True, choices=["usdm-futures"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True, choices=["1m"])
    parser.add_argument("--start", required=True, help="Inclusive UTC ISO date or offset-aware datetime")
    parser.add_argument("--end", required=True, help="Exclusive UTC ISO date or offset-aware datetime")
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show plan only, without network or writes")
    mode.add_argument("--execute-authorized-download", action="store_true", help="Execute only the exact TASK-019B scope on the operator computer")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        config = DownloadConfig(symbol=args.symbol, interval=args.interval,
                                start=args.start, end=args.end, output=args.output)
        if args.dry_run:
            print(json.dumps(preparation_plan(config), sort_keys=True, indent=2))
        else:
            validate_execution(config, args.exchange, args.product)
            download_binance_futures_klines(config)
            print(manifest_path(config.output).read_text(encoding="utf-8"))
    except (ValueError, OSError, RuntimeError, requests.RequestException, IndexError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
