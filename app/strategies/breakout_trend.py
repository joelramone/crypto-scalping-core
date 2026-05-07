from __future__ import annotations

from .rsi_mean_reversion import BaseStrategy


class BreakoutTrendStrategy(BaseStrategy):
    name = "BreakoutTrendStrategy"

    def generate_signal(self, data, regime=None):
        _ = regime

        close = data.get("close") or []
        high = data.get("high") or []
        low = data.get("low") or []
        open_prices = data.get("open") or []
        atr = data.get("atr") or []
        volume = data.get("volume") or []

        if not close or not high or not low or not open_prices or not atr or not volume:
            return None

        ema_period = int(getattr(self.config, "ema_period", 20))
        volume_avg_period = int(getattr(self.config, "volume_avg_period", 20))
        atr_avg_period = int(getattr(self.config, "atr_avg_period", 20))

        min_required = max(ema_period, volume_avg_period + 1, atr_avg_period)
        if len(close) < min_required:
            return None

        atr_value = float(atr[-1])
        if atr_value <= 0:
            return None

        if not self._passes_volatility_filter(atr):
            return None

        ema_value = self._ema(close, ema_period)
        if ema_value is None:
            return None

        if not self._passes_volume_spike_filter(volume):
            return None

        if not self._is_bullish_confirmation(open_prices, close):
            return self._maybe_short(data, ema_value, atr_value)

        long_rsi = self._resolve_rsi(data, side="LONG")
        if long_rsi is None:
            return None

        extension = atr_value * float(getattr(self.config, "extension_atr_multiplier", 0.2))
        if (
            long_rsi < float(getattr(self.config, "rsi_oversold", 20.0))
            and float(close[-2]) < (ema_value - extension)
            and self._is_bearish_exhaustion_candle(open_prices, high, low, close, -2)
        ):
            entry = float(close[-1])
            return self._build_signal(side="LONG", entry=entry, atr_value=atr_value)

        return self._maybe_short(data, ema_value, atr_value)

    def _maybe_short(self, data, ema_value: float, atr_value: float):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_prices = data["open"]

        if not self._is_bearish_confirmation(open_prices, close):
            return None

        short_rsi = self._resolve_rsi(data, side="SHORT")
        if short_rsi is None:
            return None

        extension = atr_value * float(getattr(self.config, "extension_atr_multiplier", 0.2))
        if (
            short_rsi > float(getattr(self.config, "rsi_overbought", 80.0))
            and float(close[-2]) > (ema_value + extension)
            and self._is_bullish_exhaustion_candle(open_prices, high, low, close, -2)
        ):
            entry = float(close[-1])
            return self._build_signal(side="SHORT", entry=entry, atr_value=atr_value)

        return None

    def _resolve_rsi(self, data, side: str) -> float | None:
        series_key = "rsi_7"
        if data.get(series_key):
            return float(data[series_key][-2])

        if data.get("rsi"):
            return float(data["rsi"][-2])

        close = data.get("close") or []
        period = int(getattr(self.config, "rsi_period", 7))
        return self._rsi(close, period, -2, side)

    def _passes_volatility_filter(self, atr) -> bool:
        atr_avg_period = int(getattr(self.config, "atr_avg_period", 20))
        if len(atr) < atr_avg_period:
            return False

        current_atr = float(atr[-1])
        rolling_avg = sum(float(value) for value in atr[-atr_avg_period:]) / atr_avg_period
        min_ratio = float(getattr(self.config, "min_atr_ratio", 1.0))
        return rolling_avg > 0 and (current_atr / rolling_avg) >= min_ratio

    def _passes_volume_spike_filter(self, volume) -> bool:
        volume_avg_period = int(getattr(self.config, "volume_avg_period", 20))
        if len(volume) < volume_avg_period + 1:
            return False

        current_volume = float(volume[-1])
        avg_volume = sum(float(value) for value in volume[-(volume_avg_period + 1):-1]) / volume_avg_period
        volume_multiplier = float(getattr(self.config, "volume_multiplier", 1.2))
        return current_volume >= (avg_volume * volume_multiplier)

    def _is_bearish_exhaustion_candle(self, open_prices, high, low, close, index: int) -> bool:
        return self._is_exhaustion(open_prices, high, low, close, index, bullish=False)

    def _is_bullish_exhaustion_candle(self, open_prices, high, low, close, index: int) -> bool:
        return self._is_exhaustion(open_prices, high, low, close, index, bullish=True)

    def _is_exhaustion(self, open_prices, high, low, close, index: int, bullish: bool) -> bool:
        open_price = float(open_prices[index])
        close_price = float(close[index])
        high_price = float(high[index])
        low_price = float(low[index])

        candle_range = high_price - low_price
        if candle_range <= 0:
            return False

        body = abs(close_price - open_price)
        body_ratio = body / candle_range
        min_body_ratio = float(getattr(self.config, "min_body_ratio", 0.2))
        if body_ratio < min_body_ratio:
            return False

        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        wick_ratio_min = float(getattr(self.config, "exhaustion_wick_ratio_min", 0.45))

        if bullish:
            return close_price > open_price and upper_wick / candle_range >= wick_ratio_min
        return close_price < open_price and lower_wick / candle_range >= wick_ratio_min

    @staticmethod
    def _is_bullish_confirmation(open_prices, close) -> bool:
        return float(close[-1]) > float(open_prices[-1])

    @staticmethod
    def _is_bearish_confirmation(open_prices, close) -> bool:
        return float(close[-1]) < float(open_prices[-1])

    @staticmethod
    def _ema(values, period: int) -> float | None:
        if len(values) < period:
            return None

        multiplier = 2.0 / (period + 1)
        ema = sum(float(value) for value in values[:period]) / period
        for value in values[period:]:
            ema = ((float(value) - ema) * multiplier) + ema
        return ema

    @staticmethod
    def _rsi(values, period: int, index: int, side: str) -> float | None:
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
        if avg_loss == 0:
            return 100.0 if side == "SHORT" else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _build_signal(self, side: str, entry: float, atr_value: float) -> dict[str, float | str | bool]:
        sl_distance = atr_value * float(self.config.atr_sl_multiplier)
        risk_per_trade = sl_distance
        min_tp_distance = risk_per_trade * float(getattr(self.config, "min_take_profit_r", 1.5))
        atr_tp_distance = atr_value * float(getattr(self.config, "atr_tp_multiplier", 2.0))
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
            "risk": risk_per_trade,
            "trailing_stop_enabled": bool(getattr(self.config, "trailing_stop_enabled", False)),
            "trailing_stop_atr_multiplier": float(getattr(self.config, "trailing_stop_atr_multiplier", 1.0)),
        }
