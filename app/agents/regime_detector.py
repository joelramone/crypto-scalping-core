from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import pstdev


@dataclass(frozen=True)
class RegimeState:
    high_vol_expansion: bool
    sideways: bool
    std_short: float
    std_long: float
    roc_20: float
    vol_threshold: float
    momentum_threshold: float


class RegimeDetector:
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRENDING = "TRENDING"

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 100,
        momentum_window: int = 20,
        momentum_threshold: float = 0.003,
        vol_percentile: float = 0.50,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_window = momentum_window
        self.momentum_threshold = momentum_threshold
        self.vol_percentile = vol_percentile
        self.std_history: list[float] = []

    def evaluate(self, price_history: list[float]) -> RegimeState | None:
        minimum_history = max(self.long_window, self.momentum_window + 1)
        if len(price_history) < minimum_history:
            return None

        last_short_prices = price_history[-self.short_window:]
        last_long_prices = price_history[-self.long_window:]
        std_short = pstdev(last_short_prices)
        std_long = pstdev(last_long_prices)

        previous_price = price_history[-(self.momentum_window + 1)]
        roc_20 = ((price_history[-1] - previous_price) / previous_price) if previous_price else 0.0

        self.std_history.append(std_short)
        vol_threshold = self._percentile(self.std_history, self.vol_percentile)

        high_vol_expansion = (
            std_short > std_long
            and std_short > vol_threshold
            and abs(roc_20) > self.momentum_threshold
        )
        sideways = not high_vol_expansion

        return RegimeState(
            high_vol_expansion=high_vol_expansion,
            sideways=sideways,
            std_short=std_short,
            std_long=std_long,
            roc_20=roc_20,
            vol_threshold=vol_threshold,
            momentum_threshold=self.momentum_threshold,
        )

    def detect(self, market_data: dict[str, list[float]]) -> str | None:
        close_prices = market_data.get("close", [])
        regime_state = self.evaluate(close_prices)
        if regime_state is None:
            return None

        if regime_state.high_vol_expansion:
            return self.HIGH_VOLATILITY

        if abs(regime_state.roc_20) <= regime_state.momentum_threshold:
            return self.SIDEWAYS

        return self.TRENDING

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, ceil(percentile * len(ordered)) - 1))
        return ordered[index]
