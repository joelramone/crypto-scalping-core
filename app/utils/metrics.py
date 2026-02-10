from typing import List
from app.trading.paper_wallet import Trade


def total_trades(trades: List[Trade]) -> int:
    return len(trades)


def total_fees(trades: List[Trade]) -> float:
    return sum(t.fee for t in trades)


def profit_net(trades: List[Trade]) -> float:
    profit = 0.0
    for i in range(0, len(trades), 2):
        buy = trades[i]
        sell = trades[i + 1]
        profit += (sell.price - buy.price) * buy.quantity
    return profit


def average_profit_per_trade(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return profit_net(trades) / (len(trades) / 2)
