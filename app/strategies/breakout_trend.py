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

        if not close or not high or not low or not open_prices or not atr:
            return None

        lookback_period = int(getattr(self.config, "lookback_period", 20))
        trend_ema_period = int(getattr(self.config, "trend_ema_period", 200))
        atr_avg_period = int(getattr(self.config, "atr_avg_period", 20))
        min_required = max(lookback_period + 1, trend_ema_period, atr_avg_period)
        if len(close) < min_required:
            return None

        price = float(close[-1])
        atr_value = float(atr[-1])
        if atr_value <= 0:
            return None

        if not self._passes_volatility_filter(atr):
            return None

        if not self._passes_candle_strength_filter(open_prices, high, low, close):
            return None

        window = close[-lookback_period:-1]
        if not window:
            return None

        breakout_buffer = atr_value * float(getattr(self.config, "breakout_buffer_atr", 0.0))
        window_high = max(window)
        window_low = min(window)

        trend_ema = self._ema(close, trend_ema_period)
        if trend_ema is None:
            return None

        if price >= (window_high - breakout_buffer) and price > trend_ema:
            return self._build_signal(side="LONG", entry=price, atr_value=atr_value)

        if price <= (window_low + breakout_buffer) and price < trend_ema:
            return self._build_signal(side="SHORT", entry=price, atr_value=atr_value)

        return None

    def _passes_candle_strength_filter(self, open_prices, high, low, close) -> bool:
        body_ratio_min = float(getattr(self.config, "breakout_body_ratio_min", 0.70))

        open_price = float(open_prices[-1])
        close_price = float(close[-1])
        high_price = float(high[-1])
        low_price = float(low[-1])

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
