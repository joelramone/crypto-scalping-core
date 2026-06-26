from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateSignalEvaluation(BaseModel):
    timestamp: str
    close: float
    regime: str
    signal_side: str
    signal_detected: bool
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    score: float
    atr: float | None = None
    rsi: float | None = None
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    ema_slope: float | None = None
    volume_ratio: float | None = None
    volatility: float | None = None
    distance_from_ema20: float | None = None
    distance_from_ema200: float | None = None
    future_return_5m: float | None = None
    future_return_10m: float | None = None
    future_return_15m: float | None = None
    future_max_up_15m: float | None = None
    future_max_down_15m: float | None = None


class CandidateSignalSummary(BaseModel):
    total_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    average_future_return_accepted: float = 0.0
    average_future_return_rejected: float = 0.0
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    best_rejection_reason_by_future_return: str | None = None
    best_rejection_reason_average_future_return: float | None = None
    worst_rejection_reason_by_future_return: str | None = None
    worst_rejection_reason_average_future_return: float | None = None


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
    candidate_signal_summary: CandidateSignalSummary = Field(default_factory=CandidateSignalSummary)
    trades: list[TradeResult] = Field(default_factory=list)


def summarize_candidate_signals(evaluations: list[CandidateSignalEvaluation]) -> CandidateSignalSummary:
    accepted = [evaluation for evaluation in evaluations if evaluation.accepted]
    rejected = [evaluation for evaluation in evaluations if not evaluation.accepted]
    reason_counts: dict[str, int] = {}
    reason_returns: dict[str, list[float]] = {}

    for evaluation in rejected:
        reasons = evaluation.rejection_reasons or ["unknown"]
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if evaluation.future_return_15m is not None:
                reason_returns.setdefault(reason, []).append(evaluation.future_return_15m)

    reason_averages = {
        reason: sum(values) / len(values)
        for reason, values in reason_returns.items()
        if values
    }
    best_reason = max(reason_averages, key=reason_averages.get) if reason_averages else None
    worst_reason = min(reason_averages, key=reason_averages.get) if reason_averages else None

    return CandidateSignalSummary(
        total_candidates=len(evaluations),
        accepted_candidates=len(accepted),
        rejected_candidates=len(rejected),
        average_future_return_accepted=_average_future_return_15m(accepted),
        average_future_return_rejected=_average_future_return_15m(rejected),
        rejection_reason_counts=reason_counts,
        best_rejection_reason_by_future_return=best_reason,
        best_rejection_reason_average_future_return=reason_averages.get(best_reason) if best_reason else None,
        worst_rejection_reason_by_future_return=worst_reason,
        worst_rejection_reason_average_future_return=reason_averages.get(worst_reason) if worst_reason else None,
    )


def summarize_trades(trades: list[TradeResult], candidate_signal_summary: CandidateSignalSummary | None = None) -> BacktestSummary:
    if not trades:
        return BacktestSummary(candidate_signal_summary=candidate_signal_summary or CandidateSignalSummary())

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
        candidate_signal_summary=candidate_signal_summary or CandidateSignalSummary(),
        trades=trades,
    )


def _average_future_return_15m(evaluations: list[CandidateSignalEvaluation]) -> float:
    values = [evaluation.future_return_15m for evaluation in evaluations if evaluation.future_return_15m is not None]
    return sum(values) / len(values) if values else 0.0
