from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd


logger = get_logger(__name__)


def load_real_market_data(csv_path: str | Path) -> "pd.DataFrame":
    import pandas as pd

    path = Path(csv_path)
    source_type = "csv_cache"
    if not path.exists():
        logger.error(
            "data_source_missing",
            extra={"event_name": "data_source_missing", "parameters": {"path": str(path), "source_type": source_type}},
        )
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "open_time" not in df.columns:
        raise ValueError("CSV must include open_time column")

    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    df = df.sort_values("open_time").dropna(subset=["open", "high", "low", "close", "volume", "open_time"]).reset_index(drop=True)

    invalid_ohlc = (~((df["high"] >= df["open"]) & (df["high"] >= df["close"]) & (df["low"] <= df["open"]) & (df["low"] <= df["close"]))).sum()
    if invalid_ohlc:
        logger.warning(
            "ohlc_integrity_warning",
            extra={"event_name": "ohlc_integrity_warning", "parameters": {"invalid_rows": int(invalid_ohlc)}},
        )

    timestamp_monotonic = bool(df["open_time"].is_monotonic_increasing)
    logger.info(
        "data_source_loaded",
        extra={
            "event_name": "data_source_loaded",
            "parameters": {
                "source_type": source_type,
                "path": str(path),
                "rows": int(len(df)),
                "start": str(df["open_time"].min()),
                "end": str(df["open_time"].max()),
                "is_cached": True,
                "timestamp_monotonic": timestamp_monotonic,
                "synthetic_data": False,
            },
        },
    )
    return df
