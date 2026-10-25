from __future__ import annotations

import argparse
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
    if config.output.exists() or config.output.is_symlink():
        raise FileExistsError(f"Refusing overwrite: {config.output}")
    return start, end


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
    expected_count = (end - start) // 60_000
    expected_start = pd.to_datetime(start, unit="ms", utc=True)
    expected_end = pd.to_datetime(end - 60_000, unit="ms", utc=True)
    if (len(dataframe) != expected_count or
            dataframe.iloc[0]["timestamp"] != expected_start or
            dataframe.iloc[-1]["timestamp"] != expected_end):
        raise ValueError("Missing candles or timestamps outside the requested interval")


def download_binance_futures_klines(config: DownloadConfig) -> pd.DataFrame:
    # Kept for future separately authorized callers; the CLI below never calls it.
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
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x", encoding="utf-8", newline="") as handle:
        dataframe.to_csv(handle, index=False)
    return dataframe


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a bounded Binance USD-M request; no download is performed.")
    parser.add_argument("--exchange", required=True, choices=["binance"])
    parser.add_argument("--product", required=True, choices=["usdm-futures"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True, choices=["1m"])
    parser.add_argument("--start", required=True, help="Inclusive UTC ISO date or offset-aware datetime")
    parser.add_argument("--end", required=True, help="Exclusive UTC ISO date or offset-aware datetime")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", required=True, action="store_true", help="Mandatory; show plan only, without network or writes")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        config = DownloadConfig(symbol=args.symbol, interval=args.interval,
                                start=args.start, end=args.end, output=args.output)
        print(json.dumps(preparation_plan(config), sort_keys=True, indent=2))
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
