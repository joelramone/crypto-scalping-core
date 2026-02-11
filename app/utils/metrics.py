from typing import Iterable, List, Tuple

from app.trading.paper_wallet import Trade


def total_trades(trades: List[Trade]) -> int:
    return len(trades)


def total_fees(trades: List[Trade]) -> float:
    fees = 0.0
    for trade in trades:
        try:
            fees += float(getattr(trade, "fee", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return fees


def _iter_closed_cycles(trades: Iterable[Trade]) -> Iterable[Tuple[Trade, Trade]]:
    pending_buy = None

    for trade in trades:
        side = str(getattr(trade, "side", "")).lower()

        if side == "buy":
            pending_buy = trade
            continue

        if side == "sell" and pending_buy is not None:
            buy_symbol = getattr(pending_buy, "symbol", None)
            sell_symbol = getattr(trade, "symbol", None)

            if buy_symbol == sell_symbol:
                yield pending_buy, trade
                pending_buy = None


def _trade_amount(trade: Trade) -> float:
    raw_amount = getattr(trade, "amount", None)
    if raw_amount is None:
        raw_amount = getattr(trade, "quantity", 0.0)

    try:
        return float(raw_amount or 0.0)
    except (TypeError, ValueError):
        return 0.0


def profit_gross(trades: List[Trade]) -> float:
    gross = 0.0

    for buy, sell in _iter_closed_cycles(trades):
        try:
            buy_price = float(getattr(buy, "price", 0.0) or 0.0)
            sell_price = float(getattr(sell, "price", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        amount = _trade_amount(buy)
        gross += (sell_price - buy_price) * amount

    return gross


def profit_net(trades: List[Trade]) -> float:
    return profit_gross(trades) - total_fees(trades)


def avg_profit_per_trade(trades: List[Trade]) -> float:
    closed_trades = sum(1 for _ in _iter_closed_cycles(trades))
    if closed_trades == 0:
        return 0.0
    return profit_net(trades) / closed_trades


def average_profit_per_trade(trades: List[Trade]) -> float:
    """Backward-compatible alias."""
    return avg_profit_per_trade(trades)
