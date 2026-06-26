from __future__ import annotations

from pydantic import BaseModel, Field


class TradeResult(BaseModel):
    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    strategy_name: str = "unknown"


class BacktestSummary(BaseModel):
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_drawdown: float = 0.0
    gross_pnl: float = 0.0
    estimated_fees: float = 0.0
    net_pnl: float = 0.0
    trades: list[TradeResult] = Field(default_factory=list)


def summarize_trades(trades: list[TradeResult]) -> BacktestSummary:
    if not trades:
        return BacktestSummary()

    net_values = [trade.net_pnl for trade in trades]
    wins = [value for value in net_values if value > 0]
    losses = [value for value in net_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in net_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return BacktestSummary(
        total_trades=len(trades),
        win_rate=len(wins) / len(trades),
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0.0,
        expectancy=sum(net_values) / len(trades),
        average_win=sum(wins) / len(wins) if wins else 0.0,
        average_loss=sum(losses) / len(losses) if losses else 0.0,
        max_drawdown=max_drawdown,
        gross_pnl=sum(trade.gross_pnl for trade in trades),
        estimated_fees=sum(trade.fees for trade in trades),
        net_pnl=sum(net_values),
        trades=trades,
    )
