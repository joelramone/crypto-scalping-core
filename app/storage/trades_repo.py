from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from app.utils.datetime_utils import ensure_utc


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
        self._trades.append(
            TradeRecord(
                timestamp=ensure_utc(trade.timestamp),
                symbol=trade.symbol,
                signal=trade.signal,
                size=trade.size,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
            )
        )

    def all(self) -> list[TradeRecord]:
        return list(self._trades)
