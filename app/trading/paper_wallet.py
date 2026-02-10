from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["buy", "sell"]


@dataclass
class PaperWallet:
    balance: float

    def apply_trade(self, direction: Direction, size: float, entry_price: float, exit_price: float) -> float:
        price_diff = exit_price - entry_price
        pnl = price_diff * size if direction == "buy" else (-price_diff * size)
        self.balance += pnl
        return pnl
