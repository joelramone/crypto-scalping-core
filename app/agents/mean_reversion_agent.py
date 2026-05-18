from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeanReversionConfig:
    rsi_period: int = 7
    ema_period: int = 20
    atr_period: int = 14
    volume_avg_period: int = 20
    extension_atr_multiplier: float = 0.3
    volume_spike_multiplier: float = 1.2
    min_range_to_atr_ratio: float = 0.8
    min_atr_ratio: float = 0.85
    min_take_profit_r: float = 1.4
    atr_sl_multiplier: float = 1.2
    atr_tp_multiplier: float = 1.8


class MeanReversionAgent:
    name = "MeanReversionAgent"

    def __init__(self, config: Any | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def generate_signal(self, data: dict[str, list[float]], regime: str | None = None):
        _ = regime

        close = data.get("close") or []
        open_prices = data.get("open") or []
        high = data.get("high") or []
        low = data.get("low") or []
        volume = data.get("volume") or []
        atr = data.get("atr") or []

        min_required = max(self.config.ema_period, self.config.volume_avg_period + 1, self.config.atr_period + 2)
        if min(len(close), len(open_prices), len(high), len(low), len(volume), len(atr)) < min_required:
            return None

        atr_value = float(atr[-1])
        if atr_value <= 0.0:
            return None

        if not self._passes_volatility_filter(atr, high, low):
            return None

        if not self._passes_volume_spike_filter(volume):
            return None

        ema_value = self._ema(close, self.config.ema_period)
        if ema_value is None:
            return None

        long_rsi = self._resolve_rsi(data, side="LONG")
        short_rsi = self._resolve_rsi(data, side="SHORT")

        if long_rsi is None or short_rsi is None:
            return None

        extension = atr_value * self.config.extension_atr_multiplier

        if (
            long_rsi < 20.0
            and float(close[-2]) < (ema_value - extension)
            and self._is_bearish_exhaustion_candle(open_prices, high, low, close, -2)
            and self._is_bullish_confirmation(open_prices, close)
        ):
            return self._build_signal(side="LONG", entry=float(close[-1]), atr_value=atr_value)

        if (
            short_rsi > 80.0
            and float(close[-2]) > (ema_value + extension)
            and self._is_bullish_exhaustion_candle(open_prices, high, low, close, -2)
            and self._is_bearish_confirmation(open_prices, close)
        ):
            return self._build_signal(side="SHORT", entry=float(close[-1]), atr_value=atr_value)

        return None

    def _passes_volatility_filter(self, atr: list[float], high: list[float], low: list[float]) -> bool:
        period = self.config.atr_period
        if len(atr) < period:
            return False

        current_atr = float(atr[-1])
        atr_avg = sum(float(value) for value in atr[-period:]) / period
        if atr_avg <= 0.0 or (current_atr / atr_avg) < self.config.min_atr_ratio:
            return False

        candle_range = float(high[-1]) - float(low[-1])
        return (candle_range / current_atr) >= self.config.min_range_to_atr_ratio

    def _passes_volume_spike_filter(self, volume: list[float]) -> bool:
        period = self.config.volume_avg_period
        if len(volume) < period + 1:
            return False

        current_volume = float(volume[-1])
        avg_volume = sum(float(value) for value in volume[-(period + 1):-1]) / period
        return current_volume >= (avg_volume * self.config.volume_spike_multiplier)

    @staticmethod
    def _is_bearish_exhaustion_candle(open_prices, high, low, close, index: int) -> bool:
        return MeanReversionAgent._is_exhaustion(open_prices, high, low, close, index, bullish=False)

    @staticmethod
    def _is_bullish_exhaustion_candle(open_prices, high, low, close, index: int) -> bool:
        return MeanReversionAgent._is_exhaustion(open_prices, high, low, close, index, bullish=True)

    @staticmethod
    def _is_exhaustion(open_prices, high, low, close, index: int, bullish: bool) -> bool:
        open_price = float(open_prices[index])
        close_price = float(close[index])
        high_price = float(high[index])
        low_price = float(low[index])

        candle_range = high_price - low_price
        if candle_range <= 0.0:
            return False

        body = abs(close_price - open_price)
        body_ratio = body / candle_range
        if body_ratio < 0.2:
            return False

        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price

        wick_ratio_min = 0.45
        if bullish:
            return close_price > open_price and (upper_wick / candle_range) >= wick_ratio_min
        return close_price < open_price and (lower_wick / candle_range) >= wick_ratio_min

    @staticmethod
    def _is_bullish_confirmation(open_prices, close) -> bool:
        return float(close[-1]) > float(open_prices[-1])

    @staticmethod
    def _is_bearish_confirmation(open_prices, close) -> bool:
        return float(close[-1]) < float(open_prices[-1])

    @staticmethod
    def _ema(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None

        multiplier = 2.0 / (period + 1)
        ema = sum(float(value) for value in values[:period]) / period
        for value in values[period:]:
            ema = ((float(value) - ema) * multiplier) + ema
        return ema

    def _resolve_rsi(self, data: dict[str, list[float]], side: str) -> float | None:
        if data.get("rsi_7"):
            return float(data["rsi_7"][-2])
        if data.get("rsi"):
            return float(data["rsi"][-2])
        return self._rsi(data.get("close") or [], self.config.rsi_period, -2, side)

    @staticmethod
    def _rsi(values: list[float], period: int, index: int, side: str) -> float | None:
        if len(values) < period + 2:
            return None

        end = len(values) + index + 1
        if end <= period:
            return None

        window = values[end - period - 1:end]
        gains = []
        losses = []
        for i in range(1, len(window)):
            delta = float(window[i]) - float(window[i - 1])
            gains.append(max(delta, 0.0))
            losses.append(abs(min(delta, 0.0)))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0.0:
            return 100.0 if side == "SHORT" else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _build_signal(self, side: str, entry: float, atr_value: float) -> dict[str, float | str | bool]:
        sl_distance = atr_value * self.config.atr_sl_multiplier
        min_tp_distance = sl_distance * self.config.min_take_profit_r
        atr_tp_distance = atr_value * self.config.atr_tp_multiplier
        tp_distance = max(min_tp_distance, atr_tp_distance)

        if side == "LONG":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance

        return {
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": sl_distance,
            "strategy_name": self.name,
        }
