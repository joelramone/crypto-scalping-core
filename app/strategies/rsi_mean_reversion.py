from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseStrategy:
    config: Any


class RSIMeanReversionStrategy(BaseStrategy):
    name = "RSIMeanReversionStrategy"

    def generate_signal(self, data, regime=None):
        if not data.get("rsi") or not data.get("atr") or not data.get("close"):
            return None

        rsi = float(data["rsi"][-1])
        atr = float(data["atr"][-1])
        price = float(data["close"][-1])
        ema_series = data.get("ema_20") or []
        volume = data.get("volume") or []
        min_score_threshold = int(getattr(self.config, "signal_score_threshold", 6))

        if atr <= 0.0:
            return None

        if not ema_series and len(data["close"]) >= int(getattr(self.config, "ema_period", 20)):
            ema_series = [sum(float(v) for v in data["close"][-20:]) / 20.0]
        if not ema_series:
            return None

        ema_value = float(ema_series[-1])
        distance_to_ema = abs(price - ema_value)
        min_ema_distance = atr * float(getattr(self.config, "extension_atr_multiplier", 0.2))
        atr_min_ratio = float(getattr(self.config, "min_atr_ratio", 0.8))
        score_components = {
            "rsi_extreme_confirmation": 0,
            "strong_rejection_candle": 0,
            "volume_spike": 0,
            "atr_above_threshold": 0,
            "distance_from_ema": 0,
            "trend_alignment": 0,
        }

        if atr >= atr * atr_min_ratio:
            score_components["atr_above_threshold"] = 1
        if distance_to_ema >= min_ema_distance:
            score_components["distance_from_ema"] = 1

        if volume and len(volume) >= 21:
            avg_volume = sum(float(v) for v in volume[-21:-1]) / 20.0
            if avg_volume > 0 and float(volume[-1]) >= avg_volume * float(getattr(self.config, "volume_multiplier", 1.1)):
                score_components["volume_spike"] = 1

        if rsi < self.config.rsi_oversold:
            score_components["rsi_extreme_confirmation"] = 3
            score_components["strong_rejection_candle"] = 2
            score_components["trend_alignment"] = 1 if price >= ema_value else 0
            signal_quality_score = sum(score_components.values())
            if signal_quality_score < min_score_threshold:
                return None
            sl = price - atr * self.config.atr_sl_multiplier
            tp = price + atr * self.config.atr_tp_multiplier
            return {"side": "LONG", "entry": price, "sl": sl, "tp": tp, "strategy_name": self.name, "signal_quality_score": signal_quality_score, "score_components": score_components, "regime": regime}

        if rsi > self.config.rsi_overbought:
            score_components["rsi_extreme_confirmation"] = 3
            score_components["strong_rejection_candle"] = 2
            score_components["trend_alignment"] = 1 if price <= ema_value else 0
            signal_quality_score = sum(score_components.values())
            if signal_quality_score < min_score_threshold:
                return None
            sl = price + atr * self.config.atr_sl_multiplier
            tp = price - atr * self.config.atr_tp_multiplier
            return {"side": "SHORT", "entry": price, "sl": sl, "tp": tp, "strategy_name": self.name, "signal_quality_score": signal_quality_score, "score_components": score_components, "regime": regime}

        return None
