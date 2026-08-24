"""Reusable diagnostics computed from official completed backtest trades."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import inf
from statistics import mean, median
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.research.simulation import BacktestTrade, ExitReason


class ExitDiagnostic(BaseModel):
    count: int = Field(ge=0)
    percentage: float
    net_pnl: float


class MonthlyDiagnostic(BaseModel):
    month: str
    trades: int = Field(ge=0)
    profit_factor: float
    expectancy: float
    net_pnl: float
    profitable: bool


class TradeDiagnostics(BaseModel):
    """Permanent, strategy-independent research diagnostics."""

    gross_pnl_before_fees: float
    total_fees: float
    net_pnl: float
    gross_expectancy: float
    fee_expectancy: float
    net_expectancy: float
    gross_profit_before_fees: float
    gross_loss_before_fees: float
    gross_profit_factor: float
    net_profit_factor: float
    wins: int
    losses: int
    flats: int
    win_rate: float
    average_winner: float
    median_winner: float
    average_loser: float
    median_loser: float
    payoff_ratio: float
    break_even_win_rate: float
    actual_minus_break_even_win_rate: float
    exits: dict[ExitReason, ExitDiagnostic]
    average_holding: float
    median_holding: float
    holding_p25: float
    holding_p75: float
    holding_p95: float
    raw_entry_signals: int | None = None
    completed_trades: int
    suppressed_signals: int | None = None
    suppression_rate: float | None = None
    raw_signals_per_opened_trade: float | None = None
    monthly: list[MonthlyDiagnostic]
    positive_months: int
    negative_months: int
    profitable_month_percentage: float
    best_month: str | None
    worst_month: str | None
    positive_pnl_concentration_top_2_months: float


def _profit_factor(values: list[float]) -> float:
    profit = sum(value for value in values if value > 0.0)
    loss = abs(sum(value for value in values if value < 0.0))
    return profit / loss if loss else (inf if profit else 0.0)


def _month(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def calculate_trade_diagnostics(
    trades: list[BacktestTrade],
    raw_entry_signal_indices: list[int] | None = None,
) -> TradeDiagnostics:
    """Calculate diagnostics directly from official trade records and raw signals."""
    count = len(trades)
    gross = [trade.gross_pnl for trade in trades]
    net = [trade.net_pnl for trade in trades]
    fees = [trade.fees for trade in trades]
    winners = [value for value in net if value > 0.0]
    losers = [value for value in net if value < 0.0]
    flats = [value for value in net if value == 0.0]
    gross_profit = sum(value for value in gross if value > 0.0)
    gross_loss = abs(sum(value for value in gross if value < 0.0))
    average_winner = mean(winners) if winners else 0.0
    average_loser = mean(losers) if losers else 0.0
    payoff = average_winner / abs(average_loser) if average_loser else (inf if average_winner else 0.0)
    break_even = 1.0 / (1.0 + payoff) if payoff not in (0.0, inf) else (0.0 if payoff == inf else 1.0)
    win_rate = len(winners) / count if count else 0.0

    exit_counts = Counter(trade.exit_reason for trade in trades)
    exit_pnl: dict[str, float] = defaultdict(float)
    for trade in trades:
        exit_pnl[trade.exit_reason] += trade.net_pnl
    reasons: tuple[ExitReason, ...] = ("take_profit", "stop_loss", "max_holding", "strategy_exit")
    exits = {
        reason: ExitDiagnostic(
            count=exit_counts[reason],
            percentage=exit_counts[reason] / count if count else 0.0,
            net_pnl=exit_pnl[reason],
        )
        for reason in reasons
    }

    holdings = [trade.holding_candles for trade in trades]
    percentiles = pd.Series(holdings, dtype=float).quantile([0.25, 0.75, 0.95]) if holdings else None
    grouped: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        grouped[_month(trade.exit_timestamp)].append(trade.net_pnl)
    monthly = [
        MonthlyDiagnostic(
            month=month,
            trades=len(values),
            profit_factor=_profit_factor(values),
            expectancy=sum(values) / len(values),
            net_pnl=sum(values),
            profitable=sum(values) > 0.0,
        )
        for month, values in sorted(grouped.items())
    ]
    positive = [item for item in monthly if item.net_pnl > 0.0]
    negative = [item for item in monthly if item.net_pnl < 0.0]
    total_positive = sum(item.net_pnl for item in positive)
    top_two = sum(sorted((item.net_pnl for item in positive), reverse=True)[:2])

    raw_count: int | None = None
    suppressed: int | None = None
    suppression_rate: float | None = None
    signals_per_trade: float | None = None
    if raw_entry_signal_indices is not None:
        raw_count = len(raw_entry_signal_indices)
        suppressed = sum(
            1
            for signal in raw_entry_signal_indices
            if any(trade.entry_index < signal <= trade.exit_index for trade in trades)
        )
        suppression_rate = suppressed / raw_count if raw_count else 0.0
        signals_per_trade = raw_count / count if count else 0.0

    return TradeDiagnostics(
        gross_pnl_before_fees=sum(gross), total_fees=sum(fees), net_pnl=sum(net),
        gross_expectancy=sum(gross) / count if count else 0.0,
        fee_expectancy=sum(fees) / count if count else 0.0,
        net_expectancy=sum(net) / count if count else 0.0,
        gross_profit_before_fees=gross_profit, gross_loss_before_fees=gross_loss,
        gross_profit_factor=_profit_factor(gross), net_profit_factor=_profit_factor(net),
        wins=len(winners), losses=len(losers), flats=len(flats), win_rate=win_rate,
        average_winner=average_winner, median_winner=median(winners) if winners else 0.0,
        average_loser=average_loser, median_loser=median(losers) if losers else 0.0,
        payoff_ratio=payoff, break_even_win_rate=break_even,
        actual_minus_break_even_win_rate=win_rate - break_even, exits=exits,
        average_holding=mean(holdings) if holdings else 0.0,
        median_holding=median(holdings) if holdings else 0.0,
        holding_p25=float(percentiles.loc[0.25]) if percentiles is not None else 0.0,
        holding_p75=float(percentiles.loc[0.75]) if percentiles is not None else 0.0,
        holding_p95=float(percentiles.loc[0.95]) if percentiles is not None else 0.0,
        raw_entry_signals=raw_count, completed_trades=count, suppressed_signals=suppressed,
        suppression_rate=suppression_rate, raw_signals_per_opened_trade=signals_per_trade,
        monthly=monthly, positive_months=len(positive), negative_months=len(negative),
        profitable_month_percentage=len(positive) / len(monthly) if monthly else 0.0,
        best_month=max(monthly, key=lambda item: item.net_pnl).month if monthly else None,
        worst_month=min(monthly, key=lambda item: item.net_pnl).month if monthly else None,
        positive_pnl_concentration_top_2_months=top_two / total_positive if total_positive else 0.0,
    )
