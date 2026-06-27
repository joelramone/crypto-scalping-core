"""Technical feature engineering for research backtesting."""

import pandas as pd

FEATURE_COLUMNS = (
    "returns",
    "candle_body",
    "candle_range",
    "atr14",
    "rsi14",
    "ema20",
    "ema50",
    "ema200",
    "ema20_slope",
    "volume_sma20",
    "volume_ratio",
    "volatility_20",
    "distance_from_ema20",
    "distance_from_ema200",
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for each OHLCV candle.

    The input DataFrame must include open, high, low, close, and volume columns.
    A copy is returned so callers can safely keep the raw candle data unchanged.
    """
    features = df.copy()

    close = features["close"]
    high = features["high"]
    low = features["low"]
    open_ = features["open"]
    volume = features["volume"]

    features["returns"] = close.pct_change()
    features["candle_body"] = close - open_
    features["candle_range"] = high - low

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr14"] = true_range.rolling(window=14, min_periods=14).mean()

    close_delta = close.diff()
    gains = close_delta.clip(lower=0.0)
    losses = -close_delta.clip(upper=0.0)
    average_gain = gains.rolling(window=14, min_periods=14).mean()
    average_loss = losses.rolling(window=14, min_periods=14).mean()
    relative_strength = average_gain / average_loss
    features["rsi14"] = 100.0 - (100.0 / (1.0 + relative_strength))

    features["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    features["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    features["ema200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    features["ema20_slope"] = features["ema20"].diff()

    features["volume_sma20"] = volume.rolling(window=20, min_periods=20).mean()
    features["volume_ratio"] = volume / features["volume_sma20"]
    features["volatility_20"] = features["returns"].rolling(window=20, min_periods=20).std()
    features["distance_from_ema20"] = (close - features["ema20"]) / features["ema20"]
    features["distance_from_ema200"] = (close - features["ema200"]) / features["ema200"]

    return features
