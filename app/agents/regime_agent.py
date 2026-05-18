from __future__ import annotations


class RegimeAgent:
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        low_activity_atr_ratio: float = 0.75,
        high_vol_atr_ratio: float = 1.5,
        range_compression_threshold: float = 0.8,
    ) -> None:
        self.adx_trending_threshold = adx_trending_threshold
        self.low_activity_atr_ratio = low_activity_atr_ratio
        self.high_vol_atr_ratio = high_vol_atr_ratio
        self.range_compression_threshold = range_compression_threshold

    def classify(self, market_data: dict[str, list[float]]) -> str | None:
        close = market_data.get("close") or []
        high = market_data.get("high") or []
        low = market_data.get("low") or []
        atr = market_data.get("atr") or []
        adx = market_data.get("adx") or []

        if not close or not high or not low or len(atr) < 20:
            return None

        current_atr = float(atr[-1])
        atr_avg = sum(float(v) for v in atr[-20:]) / 20
        atr_ratio = (current_atr / atr_avg) if atr_avg > 0 else 0.0

        recent_ranges = [float(h) - float(l) for h, l in zip(high[-20:], low[-20:])]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        range_to_atr = (avg_range / current_atr) if current_atr > 0 else 0.0

        if atr_ratio >= self.high_vol_atr_ratio:
            return self.HIGH_VOLATILITY

        if atr_ratio <= self.low_activity_atr_ratio and range_to_atr <= self.range_compression_threshold:
            return self.LOW_ACTIVITY

        if adx:
            if float(adx[-1]) >= self.adx_trending_threshold:
                return self.TRENDING
            return self.RANGING

        momentum = abs((float(close[-1]) - float(close[-10])) / float(close[-10])) if len(close) >= 10 and float(close[-10]) else 0.0
        if momentum >= 0.003 and atr_ratio >= 1.0:
            return self.TRENDING

        return self.RANGING
