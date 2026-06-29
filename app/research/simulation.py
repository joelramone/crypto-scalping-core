"""Simple rule-based trade simulation for research backtesting."""

from __future__ import annotations

from math import inf
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, computed_field

from app.research.config import DEFAULT_FEE_RATE
from app.research.strategies.base import BaseStrategy

TAKE_PROFIT_PCT = 0.004
STOP_LOSS_PCT = 0.0025
MAX_HOLDING_CANDLES = 30
FIXED_NOTIONAL_USDT = 100.0

ExitReason = Literal["take_profit", "stop_loss", "strategy_exit", "max_holding"]


class BacktestTrade(BaseModel):
    """A completed simulated long trade."""

    entry_index: int = Field(ge=0)
    exit_index: int = Field(ge=0)
    entry_timestamp: object
    exit_timestamp: object
    entry_price: float = Field(gt=0.0)
    exit_price: float = Field(gt=0.0)
    notional: float = Field(gt=0.0)
    gross_pnl: float
    fees: float = Field(ge=0.0)
    net_pnl: float
    exit_reason: ExitReason

    @computed_field
    @property
    def holding_candles(self) -> int:
        return self.exit_index - self.entry_index

    @computed_field
    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0.0


class BacktestMetrics(BaseModel):
    """Aggregate metrics for a completed simulation."""

    total_trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_rate: float
    gross_pnl: float
    estimated_fees: float
    net_pnl: float
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    max_drawdown: float


class BacktestResult(BaseModel):
    """Full simulation result including trades and summary metrics."""

    trades: list[BacktestTrade]
    metrics: BacktestMetrics


def simulate_strategy(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    fee_rate: float = DEFAULT_FEE_RATE,
    fixed_notional: float = FIXED_NOTIONAL_USDT,
) -> BacktestResult:
    """Run a long-only research strategy on featured candles."""
    simulation_df = df.copy()
    entry_signals = strategy.generate_entries(simulation_df).reindex(
        simulation_df.index,
        fill_value=False,
    )
    exit_signals = strategy.generate_exits(simulation_df).reindex(
        simulation_df.index,
        fill_value=False,
    )

    trades: list[BacktestTrade] = []
    index = 0
    last_entry_index = len(simulation_df) - 2

    while index <= last_entry_index:
        row = simulation_df.iloc[index]
        if not bool(entry_signals.iloc[index]):
            index += 1
            continue

        entry_index = index
        entry_row = row
        entry_price = float(entry_row["close"])
        take_profit_price = entry_price * (1.0 + strategy.take_profit_pct())
        stop_loss_price = entry_price * (1.0 - strategy.stop_loss_pct())
        max_exit_index = min(
            entry_index + strategy.max_holding_candles(),
            len(simulation_df) - 1,
        )

        exit_index = max_exit_index
        exit_price = float(simulation_df.iloc[exit_index]["close"])
        exit_reason: ExitReason = "max_holding"

        for candidate_index in range(entry_index + 1, max_exit_index + 1):
            candidate = simulation_df.iloc[candidate_index]
            if float(candidate["low"]) <= stop_loss_price:
                exit_index = candidate_index
                exit_price = stop_loss_price
                exit_reason = "stop_loss"
                break
            if float(candidate["high"]) >= take_profit_price:
                exit_index = candidate_index
                exit_price = take_profit_price
                exit_reason = "take_profit"
                break
            if bool(exit_signals.iloc[candidate_index]):
                exit_index = candidate_index
                exit_price = float(candidate["close"])
                exit_reason = "strategy_exit"
                break

        quantity = fixed_notional / entry_price
        gross_pnl = (exit_price - entry_price) * quantity
        fees = fixed_notional * fee_rate + (quantity * exit_price) * fee_rate
        net_pnl = gross_pnl - fees
        exit_row = simulation_df.iloc[exit_index]

        trades.append(
            BacktestTrade(
                entry_index=entry_index,
                exit_index=exit_index,
                entry_timestamp=entry_row["timestamp"],
                exit_timestamp=exit_row["timestamp"],
                entry_price=entry_price,
                exit_price=exit_price,
                notional=fixed_notional,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_pnl,
                exit_reason=exit_reason,
            )
        )
        index = exit_index + 1

    return BacktestResult(trades=trades, metrics=calculate_metrics(trades))


def calculate_metrics(trades: list[BacktestTrade]) -> BacktestMetrics:
    """Calculate aggregate metrics from completed trades."""
    total_trades = len(trades)
    winning_trades = [trade for trade in trades if trade.net_pnl > 0.0]
    losing_trades = [trade for trade in trades if trade.net_pnl <= 0.0]

    gross_pnl = sum(trade.gross_pnl for trade in trades)
    estimated_fees = sum(trade.fees for trade in trades)
    net_pnl = sum(trade.net_pnl for trade in trades)
    gross_profit = sum(trade.net_pnl for trade in winning_trades)
    gross_loss = abs(sum(trade.net_pnl for trade in losing_trades))

    return BacktestMetrics(
        total_trades=total_trades,
        wins=len(winning_trades),
        losses=len(losing_trades),
        win_rate=(len(winning_trades) / total_trades) if total_trades else 0.0,
        gross_pnl=gross_pnl,
        estimated_fees=estimated_fees,
        net_pnl=net_pnl,
        profit_factor=(
            (gross_profit / gross_loss)
            if gross_loss
            else (inf if gross_profit > 0.0 else 0.0)
        ),
        expectancy=(net_pnl / total_trades) if total_trades else 0.0,
        average_win=(gross_profit / len(winning_trades)) if winning_trades else 0.0,
        average_loss=(
            (sum(trade.net_pnl for trade in losing_trades) / len(losing_trades))
            if losing_trades
            else 0.0
        ),
        max_drawdown=calculate_max_drawdown(trades),
    )


def calculate_max_drawdown(trades: list[BacktestTrade]) -> float:
    """Calculate max drawdown in USDT from the cumulative net PnL equity curve."""
    peak = 0.0
    equity = 0.0
    max_drawdown = 0.0

    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)

    return abs(max_drawdown)
