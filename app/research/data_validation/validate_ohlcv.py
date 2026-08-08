"""Validate raw OHLCV CSV data without changing its contents."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
INTERVALS = {"1m": "1min"}


class ValidationResult(BaseModel):
    """Persistable validation outcome and dataset summary."""

    model_config = ConfigDict(frozen=True)

    structurally_valid: bool
    errors: list[str]
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    row_count: int
    unique_timestamps: int
    duplicate_timestamps: int
    invalid_timestamps: int
    missing_candles: int
    first_missing_timestamps: list[datetime]
    expected_candle_count: int
    completeness_percentage: float
    monotonic_ordering: bool
    min_close: float | None
    max_close: float | None
    total_volume: float | None


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse ISO-like or Unix second/millisecond timestamps as UTC."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = series.notna()
    if non_null.any() and numeric[non_null].notna().all():
        maximum = numeric[non_null].abs().max()
        unit = "ms" if maximum >= 1_000_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def validate_ohlcv(
    dataframe: pd.DataFrame,
    interval: str = "1m",
    *,
    allow_missing: bool = False,
) -> ValidationResult:
    """Validate a raw OHLCV frame and return a summary without mutating it."""
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported intervals: 1m")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")

    timestamps = _parse_timestamps(dataframe["timestamp"])
    invalid_timestamps = int(timestamps.isna().sum())
    valid_timestamps = timestamps.dropna()
    duplicate_timestamps = int(valid_timestamps.duplicated().sum())
    monotonic_ordering = invalid_timestamps == 0 and timestamps.is_monotonic_increasing

    numeric = dataframe.loc[:, NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    null_required = dataframe.loc[:, REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_numeric = numeric.isna().any(axis=1)
    comparable = numeric.dropna()
    invalid_high = comparable["high"] < comparable[["open", "close"]].max(axis=1)
    invalid_low = comparable["low"] > comparable[["open", "close"]].min(axis=1)
    negative_volume = comparable["volume"] < 0

    unique_sorted = pd.DatetimeIndex(valid_timestamps.drop_duplicates().sort_values())
    if unique_sorted.empty:
        expected = pd.DatetimeIndex([])
    else:
        expected = pd.date_range(unique_sorted[0], unique_sorted[-1], freq=INTERVALS[interval])
    missing = expected.difference(unique_sorted)
    expected_count = len(expected)
    unique_count = len(unique_sorted)
    completeness = (unique_count / expected_count * 100.0) if expected_count else 0.0

    errors: list[str] = []
    if invalid_timestamps:
        errors.append(f"{invalid_timestamps} invalid timestamp(s)")
    if not monotonic_ordering:
        errors.append("timestamps are not sorted ascending")
    if duplicate_timestamps:
        errors.append(f"{duplicate_timestamps} duplicate timestamp(s)")
    if null_required.any():
        errors.append(f"{int(null_required.sum())} row(s) contain null required fields")
    if invalid_numeric.any():
        errors.append(f"{int(invalid_numeric.sum())} row(s) contain non-numeric OHLCV values")
    if invalid_high.any():
        errors.append(f"{int(invalid_high.sum())} row(s) have high below open or close")
    if invalid_low.any():
        errors.append(f"{int(invalid_low.sum())} row(s) have low above open or close")
    if negative_volume.any():
        errors.append(f"{int(negative_volume.sum())} row(s) have negative volume")
    if len(missing) and not allow_missing:
        errors.append(f"{len(missing)} missing candle(s)")

    closes = numeric["close"].dropna()
    volumes = numeric["volume"].dropna()
    return ValidationResult(
        structurally_valid=not errors,
        errors=errors,
        first_timestamp=unique_sorted[0].to_pydatetime() if unique_count else None,
        last_timestamp=unique_sorted[-1].to_pydatetime() if unique_count else None,
        row_count=len(dataframe),
        unique_timestamps=unique_count,
        duplicate_timestamps=duplicate_timestamps,
        invalid_timestamps=invalid_timestamps,
        missing_candles=len(missing),
        first_missing_timestamps=[value.to_pydatetime() for value in missing[:20]],
        expected_candle_count=expected_count,
        completeness_percentage=completeness,
        monotonic_ordering=monotonic_ordering,
        min_close=float(closes.min()) if not closes.empty else None,
        max_close=float(closes.max()) if not closes.empty else None,
        total_volume=float(volumes.sum()) if not volumes.empty else None,
    )


def _display(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return "N/A" if value is None else str(value)


def print_validation_result(result: ValidationResult) -> None:
    """Print the validation result in a compact human-readable form."""
    print("Raw OHLCV validation summary")
    print(f"First timestamp: {_display(result.first_timestamp)}")
    print(f"Last timestamp: {_display(result.last_timestamp)}")
    print(f"Row count: {result.row_count}")
    print(f"Unique timestamps: {result.unique_timestamps}")
    print(f"Duplicates: {result.duplicate_timestamps}")
    print(f"Invalid timestamps: {result.invalid_timestamps}")
    print(f"Expected candle count: {result.expected_candle_count}")
    print(f"Missing candles: {result.missing_candles}")
    print(f"Completeness: {result.completeness_percentage:.6f}%")
    print(f"Monotonic ordering: {result.monotonic_ordering}")
    print(f"Min close: {_display(result.min_close)}")
    print(f"Max close: {_display(result.max_close)}")
    print(f"Total volume: {_display(result.total_volume)}")
    print("First missing timestamps:")
    for timestamp in result.first_missing_timestamps:
        print(f"  {timestamp.isoformat()}")
    print(f"Structurally valid: {result.structurally_valid}")
    for error in result.errors:
        print(f"ERROR: {error}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a raw OHLCV research CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Raw OHLCV CSV path.")
    parser.add_argument("--interval", required=True, choices=sorted(INTERVALS))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing candles without failing validation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        dataframe = pd.read_csv(args.input)
        result = validate_ohlcv(dataframe, args.interval, allow_missing=args.allow_missing)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        raise SystemExit(1) from exc
    print_validation_result(result)
    raise SystemExit(0 if result.structurally_valid else 1)


if __name__ == "__main__":
    main()
