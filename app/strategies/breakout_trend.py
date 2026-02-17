from __future__ import annotations

from .rsi_mean_reversion import BaseStrategy


class BreakoutTrendStrategy(BaseStrategy):
    name = "BreakoutTrendStrategy"

    def generate_signal(self, data, regime=None):
        _ = regime

        close = data.get("close") or []
        atr = data.get("atr") or []

        if not close or not atr:
            return None

        lookback_period = getattr(self.config, "lookback_period", 20)
        if len(close) <= lookback_period:
            return None

        price = float(close[-1])
        window = close[-lookback_period:-1]
        if not window:
            return None

        atr_value = float(atr[-1])
        if atr_value <= 0:
            return None

        breakout_buffer = atr_value * float(getattr(self.config, "breakout_buffer_atr", 0.0))
        window_high = max(window)
        window_low = min(window)

        if price >= (window_high - breakout_buffer):
            return self._build_signal(side="LONG", entry=price, atr_value=atr_value)

        if price <= (window_low + breakout_buffer):
            return self._build_signal(side="SHORT", entry=price, atr_value=atr_value)

        return None

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
