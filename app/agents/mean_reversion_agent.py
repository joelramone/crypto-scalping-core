from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.utils.signal_rejections import SignalRejectionEvent, build_rejection_event, safe_ratio


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
        self._last_rejection_event: SignalRejectionEvent | None = None

    def generate_signal(self, data: dict[str, list[float]], regime: str | None = None):
        self._last_rejection_event = None
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
            self._set_rejection(data, regime, ["atr_invalid"], atr=atr_value, atr_status="invalid")
            return None

        if not self._passes_volatility_filter(atr, high, low):
            self._set_rejection(data, regime, ["atr_too_low"], atr=atr_value, atr_status="low")
            return None

        if not self._passes_volume_spike_filter(volume):
            self._set_rejection(data, regime, ["volume_too_low"], atr=atr_value)
            return None

        ema_value = self._ema(close, self.config.ema_period)
        if ema_value is None:
            return None

        long_rsi = self._resolve_rsi(data, side="LONG")
        short_rsi = self._resolve_rsi(data, side="SHORT")

        if long_rsi is None or short_rsi is None:
            return None

        extension = atr_value * self.config.extension_atr_multiplier

        min_score_threshold = int(getattr(self.config, "signal_score_threshold", 6))

        if long_rsi < 20.0 and float(close[-2]) < (ema_value - extension):
            long_components = self._score_components(
                side="LONG",
                rsi_value=long_rsi,
                open_prices=open_prices,
                high=high,
                low=low,
                close=close,
                volume=volume,
                atr=atr,
                atr_value=atr_value,
                ema_value=ema_value,
                regime=regime,
            )
            long_score = sum(long_components.values())
            if long_score >= min_score_threshold:
                return self._build_signal(side="LONG", entry=float(close[-1]), atr_value=atr_value, signal_quality_score=long_score, score_components=long_components, regime=regime)
            self._set_rejection(data, regime, ["score_too_low"], score=float(long_score), required_score=float(min_score_threshold), atr=atr_value, ema_value=ema_value)

        if short_rsi > 80.0 and float(close[-2]) > (ema_value + extension):
            short_components = self._score_components(
                side="SHORT",
                rsi_value=short_rsi,
                open_prices=open_prices,
                high=high,
                low=low,
                close=close,
                volume=volume,
                atr=atr,
                atr_value=atr_value,
                ema_value=ema_value,
                regime=regime,
            )
            short_score = sum(short_components.values())
            if short_score >= min_score_threshold:
                return self._build_signal(side="SHORT", entry=float(close[-1]), atr_value=atr_value, signal_quality_score=short_score, score_components=short_components, regime=regime)
            self._set_rejection(data, regime, ["score_too_low"], score=float(short_score), required_score=float(min_score_threshold), atr=atr_value, ema_value=ema_value)

        if self._last_rejection_event is None:
            self._set_rejection(data, regime, ["score_too_low"], score=0.0, required_score=float(min_score_threshold), atr=atr_value, ema_value=ema_value)
        return None

    def consume_last_rejection_event(self) -> SignalRejectionEvent | None:
        event = self._last_rejection_event
        self._last_rejection_event = None
        return event

    def _set_rejection(
        self,
        data: dict[str, list[float]],
        regime: str | None,
        reasons: list[str],
        *,
        score: float | None = None,
        required_score: float | None = None,
        atr: float | None = None,
        atr_status: str | None = None,
        ema_value: float | None = None,
    ) -> None:
        self._last_rejection_event = build_rejection_event(
            strategy=self.name,
            regime=regime,
            reasons=reasons,
            score=score,
            required_score=required_score,
            atr=atr if atr is not None else self._current_atr(data),
            atr_status=atr_status,
            volume_ratio=self._volume_ratio(data),
            ema_distance=self._ema_distance(data, ema_value),
        )

    @staticmethod
    def _current_atr(data: dict[str, list[float]]) -> float | None:
        atr = data.get("atr") or []
        return float(atr[-1]) if atr else None

    def _volume_ratio(self, data: dict[str, list[float]]) -> float | None:
        volume = data.get("volume") or []
        period = int(getattr(self.config, "volume_avg_period", 20))
        if len(volume) < period + 1:
            return None
        avg_volume = sum(float(value) for value in volume[-(period + 1):-1]) / period
        return safe_ratio(volume[-1], avg_volume)

    def _ema_distance(self, data: dict[str, list[float]], ema_value: float | None) -> float | None:
        close = data.get("close") or []
        if not close:
            return None
        if ema_value is None:
            ema_value = self._ema(close, int(getattr(self.config, "ema_period", 20)))
        if ema_value is None:
            return None
        return float(close[-1]) - ema_value

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

    def _score_components(
        self,
        side: str,
        rsi_value: float,
        open_prices: list[float],
        high: list[float],
        low: list[float],
        close: list[float],
        volume: list[float],
        atr: list[float],
        atr_value: float,
        ema_value: float,
        regime: str | None,
    ) -> dict[str, int]:
        components = {
            "rsi_extreme_confirmation": 0,
            "strong_rejection_candle": 0,
            "volume_spike": 0,
            "atr_above_threshold": 0,
            "distance_from_ema": 0,
            "trend_alignment": 0,
        }
        components["rsi_extreme_confirmation"] = 3 if (side == "LONG" and rsi_value < 20.0) or (side == "SHORT" and rsi_value > 80.0) else 0
        is_rejection = self._is_bearish_exhaustion_candle(open_prices, high, low, close, -2) if side == "LONG" else self._is_bullish_exhaustion_candle(open_prices, high, low, close, -2)
        if is_rejection:
            components["strong_rejection_candle"] = 2
        if self._passes_volume_spike_filter(volume):
            components["volume_spike"] = 1
        if self._passes_volatility_filter(atr, high, low):
            components["atr_above_threshold"] = 1
        extension = atr_value * self.config.extension_atr_multiplier
        if abs(float(close[-2]) - ema_value) >= extension:
            components["distance_from_ema"] = 1
        if regime in {"RANGING", "HIGH_VOLATILITY", None}:
            components["trend_alignment"] = 1
        return components

    def _build_signal(self, side: str, entry: float, atr_value: float, signal_quality_score: int, score_components: dict[str, int], regime: str | None) -> dict[str, float | str | bool | int | dict[str, int] | None]:
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
            "signal_quality_score": signal_quality_score,
            "score_components": score_components,
            "regime": regime,
        }
