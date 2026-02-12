from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseStrategy:
    config: Any


class RSIMeanReversionStrategy(BaseStrategy):
    def generate_signal(self, data, regime=None):
        _ = regime

        if not data.get("rsi") or not data.get("atr") or not data.get("close"):
            return None

        rsi = data["rsi"][-1]
        atr = data["atr"][-1]
        price = data["close"][-1]

        if rsi < self.config.rsi_oversold:
            sl = price - atr * self.config.atr_sl_multiplier
            tp = price + atr * self.config.atr_tp_multiplier
            return {"side": "LONG", "entry": price, "sl": sl, "tp": tp}

        if rsi > self.config.rsi_overbought:
            sl = price + atr * self.config.atr_sl_multiplier
            tp = price - atr * self.config.atr_tp_multiplier
            return {"side": "SHORT", "entry": price, "sl": sl, "tp": tp}

        return None
