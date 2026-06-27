from __future__ import annotations

import argparse
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


def download_binance_futures_klines(config: DownloadConfig) -> pd.DataFrame:
    start_time_ms = parse_datetime_to_utc_ms(config.start)
    end_time_ms = parse_datetime_to_utc_ms(config.end) - 1
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
            if len(page) < MAX_LIMIT:
                break

            time.sleep(REQUEST_SLEEP_SECONDS)

    dataframe = rows_to_dataframe(rows)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(config.output, index=False)
    return dataframe


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance USD-M Futures klines into CSV.")
    parser.add_argument("--symbol", required=True, help="Binance Futures symbol, for example BTCUSDT.")
    parser.add_argument("--interval", required=True, help="Kline interval, for example 1m, 5m, 1h, or 1d.")
    parser.add_argument("--start", required=True, help="Inclusive UTC start date or datetime, for example 2025-01-01.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end date or datetime, for example 2026-01-01.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path, for example data/BTCUSDT_1m.csv.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry attempts per Binance request.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()
    config = DownloadConfig(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        output=args.output,
        retries=args.retries,
    )

    try:
        dataframe = download_binance_futures_klines(config)
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        logging.error("Failed to download Binance Futures klines: %s", exc)
        raise SystemExit(1) from exc

    logging.info("Downloaded %s candles to %s", len(dataframe), config.output)


if __name__ == "__main__":
    main()
