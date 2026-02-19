from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
REQUEST_SLEEP_SECONDS = 0.5
INTERVAL_TO_MS = {
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


def _parse_datetime_to_utc_ms(value: str) -> int:
    """Parse a date/datetime string and return epoch milliseconds in UTC."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Formato de fecha inválido: '{value}'. Usa ISO (ej: 2024-01-01 o 2024-01-01T00:00:00+00:00)."
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return int(dt.timestamp() * 1000)


def _validate_interval(interval: str) -> None:
    if interval not in INTERVAL_TO_MS:
        valid = ", ".join(INTERVAL_TO_MS.keys())
        raise ValueError(f"Intervalo '{interval}' no soportado. Valores válidos: {valid}")


def _fetch_klines_page(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int | None,
    session: requests.Session,
) -> list[list[Any]]:
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": MAX_LIMIT,
    }
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"Error HTTP al consultar Binance ({response.status_code}): {response.text}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f"Respuesta JSON inválida de Binance: {response.text}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Respuesta inesperada de Binance: {data}")

    return data


def download_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    output_csv_path: str = "data/binance_klines.csv",
) -> pd.DataFrame:
    """Download historical klines from Binance public endpoint and save to CSV."""
    _validate_interval(interval)

    start_time_ms = _parse_datetime_to_utc_ms(start_date)
    end_time_ms = _parse_datetime_to_utc_ms(end_date) if end_date else None

    if end_time_ms is not None and end_time_ms < start_time_ms:
        raise ValueError("end_date debe ser posterior o igual a start_date")

    all_rows: list[list[Any]] = []
    current_start_ms = start_time_ms
    interval_ms = INTERVAL_TO_MS[interval]

    session = requests.Session()

    try:
        while True:
            page = _fetch_klines_page(
                symbol=symbol,
                interval=interval,
                start_time_ms=current_start_ms,
                end_time_ms=end_time_ms,
                session=session,
            )

            if not page:
                break

            all_rows.extend(page)

            if len(page) < MAX_LIMIT:
                break

            last_open_time = int(page[-1][0])
            next_start_ms = last_open_time + interval_ms

            if end_time_ms is not None and next_start_ms > end_time_ms:
                break

            current_start_ms = next_start_ms
            time.sleep(REQUEST_SLEEP_SECONDS)
    finally:
        session.close()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    ]

    if not all_rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(
            (
                {
                    "open_time": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                    "close_time": row[6],
                }
                for row in all_rows
            ),
            columns=columns,
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if df.empty:
        logging.info("No se descargaron velas para %s %s en el rango solicitado.", symbol, interval)
    else:
        logging.info("Total candles descargadas: %d", len(df))
        logging.info("Fecha mínima: %s", df["open_time"].min())
        logging.info("Fecha máxima: %s", df["open_time"].max())
        logging.info("Primeras 5 filas:\n%s", df.head().to_string(index=False))
        logging.info("Últimas 5 filas:\n%s", df.tail().to_string(index=False))
        logging.info("CSV guardado en: %s", output_path)

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Descarga datos históricos de Binance (klines).")
    parser.add_argument("--symbol", default="BTCUSDT", help="Símbolo de mercado (default: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Intervalo de velas (default: 1m)")
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Fecha inicial ISO (ej: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Fecha final ISO opcional (ej: 2024-02-01)",
    )
    parser.add_argument(
        "--output-csv-path",
        default="data/binance_klines.csv",
        help="Ruta de salida CSV (default: data/binance_klines.csv)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        download_binance_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            output_csv_path=args.output_csv_path,
        )
    except (requests.RequestException, ValueError) as exc:
        logging.error("Fallo en descarga de klines: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
