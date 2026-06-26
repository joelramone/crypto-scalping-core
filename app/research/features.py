from __future__ import annotations

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv_columns(candles: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in candles.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")


def add_basic_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV candles with deterministic research features attached."""
    validate_ohlcv_columns(candles)
    frame = candles.copy()
    for column in REQUIRED_OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["returns"] = frame["close"].pct_change().fillna(0.0)
    frame["candle_body_size"] = (frame["close"] - frame["open"]).abs()
    frame["candle_range"] = frame["high"] - frame["low"]
    frame["atr"] = calculate_atr(frame)
    frame["rsi"] = calculate_rsi(frame["close"])
    frame["rsi_7"] = calculate_rsi(frame["close"], period=7)
    frame["ema_20"] = frame["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    frame["ema_50"] = frame["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    frame["ema_200"] = frame["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    frame["ema_34"] = frame["close"].ewm(span=34, adjust=False, min_periods=34).mean()
    frame["ema_slope"] = frame["ema_20"].diff()
    volume_average = frame["volume"].rolling(window=20, min_periods=20).mean()
    frame["volume_ratio"] = frame["volume"] / volume_average
    frame["volatility"] = frame["returns"].rolling(window=20, min_periods=20).std()
    frame["distance_from_ema20"] = frame["close"] - frame["ema_20"]
    frame["distance_from_ema200"] = frame["close"] - frame["ema_200"]
    frame["atr_avg"] = frame["atr"].rolling(window=20, min_periods=20).mean()
    return frame


def calculate_atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = candles["close"].shift(1)
    true_range = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - previous_close).abs(),
            (candles["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.rolling(window=period, min_periods=period).mean()
    average_loss = losses.rolling(window=period, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.fillna(50.0)
