from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class TradeRecord:
    timestamp: datetime
    symbol: str
    signal: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float


class TradesRepository:
    def __init__(self) -> None:
        self._trades: List[TradeRecord] = []

    def add(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def all(self) -> list[TradeRecord]:
        return list(self._trades)
