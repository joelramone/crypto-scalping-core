from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Iterator


@dataclass(frozen=True)
class MarketTick:
    symbol: str
    price: float


class MarketStream:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._prices = cycle([50000.0, 50120.0, 49980.0, 50250.0, 50300.0])

    def stream(self) -> Iterator[MarketTick]:
        while True:
            yield MarketTick(symbol=self.symbol, price=next(self._prices))
