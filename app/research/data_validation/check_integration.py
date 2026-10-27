"""Read-only TASK-020 loader/resampling check; no strategies are executed."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd

from app.research.backtester import load_ohlcv_csv
from app.research.data_utils import OPTIONAL_SUM_COLUMNS, resample_ohlcv
from app.research.data_validation.validate_ohlcv import validate_ohlcv

EXPECTED_PATH = Path("data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv")
EXPECTED_SHA256 = "f04bbaf8b8fe5a333bd5e67891cec081fea4b9dba8850a3166bbbfb9e3893c5f"
START = pd.Timestamp("2024-10-01T00:00:00Z")
END = pd.Timestamp("2024-10-08T00:00:00Z")


def check_integration(input_path: Path, manifest_path: Path) -> dict:
    """Check only the locally authorized bytes, without modifying any file."""
    if input_path.resolve() != (Path.cwd() / EXPECTED_PATH).resolve():
        raise ValueError("Input must be the TASK-020 authorized local dataset path")
    if manifest_path.resolve() != input_path.with_suffix(".csv.manifest.json").resolve():
        raise ValueError("Manifest must be the authorized CSV's adjacent manifest")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Manifest must be a JSON object")
    expected = {"status": "DATASET_VALIDATED", "source": "binance", "product": "usdm-futures",
                "symbol": "BTCUSDT", "timeframe": "1m", "row_count": 10080,
                "sha256": EXPECTED_SHA256,
                "start_inclusive_utc": START.isoformat(), "end_exclusive_utc": END.isoformat(),
                "first_timestamp_utc": START.isoformat(),
                "last_timestamp_utc": (END - pd.Timedelta(minutes=1)).isoformat()}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Manifest mismatch: {field}")
    payload = input_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256 or digest != metadata["sha256"]:
        raise ValueError("CSV SHA-256 does not match the validated dataset and manifest")
    if len(payload) != metadata.get("file_size_bytes"):
        raise ValueError("CSV byte size does not match manifest")
    # The existing backtester loader sees exactly the bytes that were hashed.
    frame = load_ohlcv_csv(io.BytesIO(payload))
    validation = validate_ohlcv(frame)
    if not validation.structurally_valid:
        raise ValueError(f"Invalid 1m OHLCV: {validation.errors}")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    numeric_columns += [c for c in OPTIONAL_SUM_COLUMNS if c in frame]
    if not frame[numeric_columns].map(math.isfinite).all().all():
        raise ValueError("Nonfinite OHLCV or optional aggregates")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    expected_index = pd.date_range(START, END, freq="1min", inclusive="left")
    if not pd.DatetimeIndex(frame["timestamp"]).equals(expected_index):
        raise ValueError("Source coverage/count does not match the exact authorized UTC interval")
    summaries = {}
    for interval, minutes, count in (("1m", 1, 10080), ("5m", 5, 2016), ("15m", 15, 672)):
        result = resample_ohlcv(frame, interval)
        index = pd.date_range(START, END, freq=f"{minutes}min", inclusive="left")
        counts = frame.set_index("timestamp").resample(f"{minutes}min", label="left", closed="left").size()
        if len(result) != count or not pd.DatetimeIndex(result["timestamp"]).equals(index):
            raise ValueError(f"Unexpected resampled coverage/count: {interval}")
        if not counts.eq(minutes).all():
            raise ValueError(f"Incomplete input bucket: {interval}")
        if not result[numeric_columns].map(math.isfinite).all().all():
            raise ValueError(f"Invalid resampled values: {interval}")
        summaries[interval] = {"row_count": len(result), "complete_buckets": True,
                               "first_open_utc": result.iloc[0]["timestamp"].isoformat(),
                               "last_open_utc": result.iloc[-1]["timestamp"].isoformat(),
                               "last_available_at_utc": (result.iloc[-1]["timestamp"] + pd.Timedelta(minutes=minutes)).isoformat()}
    return {"status": "INTEGRATION_VALIDATED", "input": str(input_path), "manifest": str(manifest_path),
            "source": "binance", "product": "usdm-futures", "symbol": "BTCUSDT",
            "sha256": digest, "sha256_verified": True, "file_size_bytes": len(payload),
            "schema_compatible": True, "timestamps_utc": True, "continuous_1m": True,
            "duplicate_timestamps": validation.duplicate_timestamps, "missing_candles": validation.missing_candles,
            "resampled": summaries, "strategy_executed": False,
            "resampling_convention": "UTC left-closed, left-labeled; aggregates available only at bucket end; no fill/interpolation/indicators"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = check_integration(args.input, args.manifest)
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        print(json.dumps({"status": "INTEGRATION_BLOCKED", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
