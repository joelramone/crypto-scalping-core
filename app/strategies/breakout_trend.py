from __future__ import annotations

from .rsi_mean_reversion import BaseStrategy


class BreakoutTrendStrategy(BaseStrategy):
    def generate_signal(self, data, regime=None):
        _ = regime

        close = data.get("close")
        atr = data.get("atr")

        if not close or not atr:
            return None

        lookback_period = getattr(self.config, "lookback_period", 20)
        if len(close) <= lookback_period:
            return None

        price = close[-1]
        window = close[-lookback_period:-1]

        if not window:
            return None

        atr_value = atr[-1]

        if price > max(window):
            sl = price - atr_value * self.config.atr_sl_multiplier
            tp = price + atr_value * self.config.atr_tp_multiplier
            return {"side": "LONG", "entry": price, "sl": sl, "tp": tp}

        if price < min(window):
            sl = price + atr_value * self.config.atr_sl_multiplier
            tp = price - atr_value * self.config.atr_tp_multiplier
            return {"side": "SHORT", "entry": price, "sl": sl, "tp": tp}

        return None
