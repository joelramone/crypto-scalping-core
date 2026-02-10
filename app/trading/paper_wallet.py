from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class Trade:
    symbol: str
    side: str  # "buy" | "sell"
    price: float
    quantity: float
    fee: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PaperWallet:
    def __init__(self, initial_balance: float = 250.0, fee_rate: float = 0.001):
        self.fee_rate = fee_rate
        self.balances: Dict[str, float] = {
            "USDT": initial_balance
        }
        self.trades: List[Trade] = []

    def get_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    def buy(self, symbol: str, price: float, quantity: float):
        base, quote = symbol.split("/")
        cost = price * quantity
        fee = cost * self.fee_rate
        total_cost = cost + fee

        if self.get_balance(quote) < total_cost:
            raise ValueError("Insufficient balance")

        self.balances[quote] -= total_cost
        self.balances[base] = self.get_balance(base) + quantity

        self.trades.append(
            Trade(symbol, "buy", price, quantity, fee)
        )

    def sell(self, symbol: str, price: float, quantity: float):
        base, quote = symbol.split("/")
        if self.get_balance(base) < quantity:
            raise ValueError("Insufficient asset balance")

        revenue = price * quantity
        fee = revenue * self.fee_rate
        net_revenue = revenue - fee

        self.balances[base] -= quantity
        self.balances[quote] = self.get_balance(quote) + net_revenue

        self.trades.append(
            Trade(symbol, "sell", price, quantity, fee)
        )

    def total_pnl(self, current_prices: Dict[str, float]) -> float:
        total = self.get_balance("USDT")
        for asset, qty in self.balances.items():
            if asset == "USDT":
                continue
            price = current_prices.get(asset)
            if price:
                total += qty * price
        return total
