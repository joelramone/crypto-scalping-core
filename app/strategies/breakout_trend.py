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

        lookback_period = int(getattr(self.config, "lookback_period", 20))
        trend_ema_period = int(getattr(self.config, "trend_ema_period", 200))
        atr_avg_period = int(getattr(self.config, "atr_avg_period", 20))
        min_sequence_bars = int(getattr(self.config, "confirmation_sequence_bars", 3))
        volume_avg_period = int(getattr(self.config, "volume_avg_period", 20))

        min_required = max(
            lookback_period + min_sequence_bars,
            trend_ema_period,
            atr_avg_period,
            volume_avg_period,
        )
        if len(close) < min_required:
            return None

        atr_value = float(atr[-1])
        if atr_value <= 0:
            return None

        if not self._passes_volatility_filter(atr):
            return None

        if not self._passes_volume_filter(volume):
            return None

        trend_ema = self._ema(close, trend_ema_period)
        if trend_ema is None:
            return None

        breakout_index = -3
        pullback_index = -2
        confirmation_index = -1

        breakout_close = float(close[breakout_index])
        pullback_close = float(close[pullback_index])
        confirmation_close = float(close[confirmation_index])
        confirmation_open = float(open_prices[confirmation_index])

        breakout_buffer = atr_value * float(getattr(self.config, "breakout_buffer_atr", 0.0))
        pullback_tolerance = atr_value * float(getattr(self.config, "pullback_tolerance_atr", 0.20))
        breakout_window = close[-(lookback_period + min_sequence_bars):-min_sequence_bars]
        if not breakout_window:
            return None

        breakout_high = max(float(value) for value in breakout_window)
        breakout_low = min(float(value) for value in breakout_window)

        if (
            breakout_close >= (breakout_high + breakout_buffer)
            and pullback_close >= (breakout_high - pullback_tolerance)
            and pullback_close <= breakout_close
            and confirmation_close > confirmation_open
            and confirmation_close > pullback_close
            and confirmation_close > trend_ema
            and self._passes_candle_strength_filter(open_prices, high, low, close, confirmation_index)
        ):
            return self._build_signal(side="LONG", entry=confirmation_close, atr_value=atr_value)

        if (
            breakout_close <= (breakout_low - breakout_buffer)
            and pullback_close <= (breakout_low + pullback_tolerance)
            and pullback_close >= breakout_close
            and confirmation_close < confirmation_open
            and confirmation_close < pullback_close
            and confirmation_close < trend_ema
            and self._passes_candle_strength_filter(open_prices, high, low, close, confirmation_index)
        ):
            return self._build_signal(side="SHORT", entry=confirmation_close, atr_value=atr_value)

        return None

    def _passes_candle_strength_filter(self, open_prices, high, low, close, index: int = -1) -> bool:
        body_ratio_min = float(getattr(self.config, "breakout_body_ratio_min", 0.70))

        open_price = float(open_prices[index])
        close_price = float(close[index])
        high_price = float(high[index])
        low_price = float(low[index])

        candle_range = high_price - low_price
        if candle_range <= 0:
            return False

        body = abs(close_price - open_price)
        body_ratio = body / candle_range
        if body_ratio < body_ratio_min:
            return False

        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        wick_ratio = (upper_wick + lower_wick) / candle_range

        return wick_ratio <= (1.0 - body_ratio_min)

    def _passes_volatility_filter(self, atr) -> bool:
        atr_avg_period = int(getattr(self.config, "atr_avg_period", 20))
        if len(atr) < atr_avg_period:
            return False

        current_atr = float(atr[-1])
        rolling_avg = sum(float(value) for value in atr[-atr_avg_period:]) / atr_avg_period
        return current_atr > rolling_avg

    def _passes_volume_filter(self, volume) -> bool:
        volume_avg_period = int(getattr(self.config, "volume_avg_period", 20))
        if len(volume) < volume_avg_period:
            return False

        current_volume = float(volume[-1])
        avg_volume = sum(float(value) for value in volume[-volume_avg_period:]) / volume_avg_period
        volume_multiplier = float(getattr(self.config, "volume_multiplier", 1.0))
        return current_volume >= (avg_volume * volume_multiplier)

    @staticmethod
    def _ema(values, period: int) -> float | None:
        if len(values) < period:
            return None

        multiplier = 2.0 / (period + 1)
        ema = sum(float(value) for value in values[:period]) / period
        for value in values[period:]:
            ema = ((float(value) - ema) * multiplier) + ema
        return ema

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
