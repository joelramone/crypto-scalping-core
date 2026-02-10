from __future__ import annotations

from dataclasses import dataclass

from app.data.market_stream import MarketTick


@dataclass(frozen=True)
class FeatureSnapshot:
    symbol: str
    price: float
    momentum: float


class FeatureBuilder:
    def __init__(self) -> None:
        self._last_price: float | None = None

    def build(self, tick: MarketTick) -> FeatureSnapshot:
        momentum = 0.0
        if self._last_price is not None and self._last_price != 0:
            momentum = (tick.price - self._last_price) / self._last_price
        self._last_price = tick.price
        return FeatureSnapshot(symbol=tick.symbol, price=tick.price, momentum=momentum)
