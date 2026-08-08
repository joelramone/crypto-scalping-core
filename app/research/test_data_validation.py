from __future__ import annotations

import pandas as pd

from app.research.data_validation.validate_ohlcv import validate_ohlcv


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC").astype(str),
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1, 2, 3],
        }
    )


def test_valid_continuous_data() -> None:
    result = validate_ohlcv(_candles())
    assert result.structurally_valid
    assert result.expected_candle_count == 3
    assert result.completeness_percentage == 100.0


def test_one_missing_candle() -> None:
    result = validate_ohlcv(_candles().drop(index=1))
    assert not result.structurally_valid
    assert result.missing_candles == 1
    assert result.first_missing_timestamps[0].minute == 1


def test_duplicate_timestamp() -> None:
    frame = pd.concat([_candles(), _candles().iloc[[1]]], ignore_index=True)
    result = validate_ohlcv(frame)
    assert not result.structurally_valid
    assert result.duplicate_timestamps == 1


def test_unsorted_timestamps() -> None:
    result = validate_ohlcv(_candles().iloc[[1, 0, 2]])
    assert not result.structurally_valid
    assert not result.monotonic_ordering


def test_invalid_ohlc_relationship() -> None:
    frame = _candles()
    frame.loc[1, "high"] = 100
    result = validate_ohlcv(frame)
    assert not result.structurally_valid
    assert any("high" in error for error in result.errors)


def test_negative_volume() -> None:
    frame = _candles()
    frame.loc[0, "volume"] = -1
    result = validate_ohlcv(frame)
    assert not result.structurally_valid
    assert any("negative volume" in error for error in result.errors)


def test_timestamps_are_parsed_as_utc() -> None:
    result = validate_ohlcv(_candles())
    assert result.first_timestamp is not None
    assert str(result.first_timestamp.tzinfo) == "UTC"


def test_allow_missing_only_ignores_missing_candles() -> None:
    missing = validate_ohlcv(_candles().drop(index=1), allow_missing=True)
    assert missing.structurally_valid
    assert missing.missing_candles == 1

    invalid = _candles().drop(index=1)
    invalid.loc[0, "volume"] = -1
    assert not validate_ohlcv(invalid, allow_missing=True).structurally_valid
