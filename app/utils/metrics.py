from math import sqrt
from typing import Iterable, List, Sequence, Tuple

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


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0

    mean_value = _safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (n - 1)
    return sqrt(variance)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    q_clamped = max(0.0, min(1.0, q))
    rank = (len(sorted_values) - 1) * q_clamped
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)

    if low == high:
        return sorted_values[low]

    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _build_equity_curve(r_values: Sequence[float]) -> List[float]:
    equity_curve = [1.0]
    current = 1.0

    for r_value in r_values:
        current *= max(0.0, 1.0 + r_value)
        equity_curve.append(current)

    return equity_curve


def _drawdown_series(equity_curve: Sequence[float]) -> List[float]:
    if not equity_curve:
        return []

    peak = equity_curve[0]
    drawdowns: List[float] = []

    for value in equity_curve:
        peak = max(peak, value)
        if peak <= 0.0:
            drawdowns.append(0.0)
        else:
            drawdowns.append((peak - value) / peak)

    return drawdowns


def _sharpe_annualized(r_values: Sequence[float], periods_per_year: int) -> float:
    std = _sample_std(r_values)
    if std == 0.0:
        if _safe_mean(r_values) > 0.0:
            return float("inf")
        return 0.0

    return sqrt(periods_per_year) * (_safe_mean(r_values) / std)


def _sortino_annualized(r_values: Sequence[float], periods_per_year: int) -> float:
    if not r_values:
        return 0.0

    downside = [min(0.0, value) for value in r_values]
    downside_deviation = sqrt(sum(value * value for value in downside) / len(downside))
    avg_return = _safe_mean(r_values)

    if downside_deviation == 0.0:
        if avg_return > 0.0:
            return float("inf")
        return 0.0

    return sqrt(periods_per_year) * (avg_return / downside_deviation)


def _kelly_fraction(r_values: Sequence[float]) -> float:
    if not r_values:
        return 0.0

    wins = [value for value in r_values if value > 0.0]
    losses = [value for value in r_values if value < 0.0]

    if not wins and not losses:
        return 0.0
    if wins and not losses:
        return 1.0
    if losses and not wins:
        return 0.0

    win_rate = len(wins) / len(r_values)
    loss_rate = 1.0 - win_rate

    avg_win = _safe_mean(wins)
    avg_loss_abs = abs(_safe_mean(losses))
    if avg_loss_abs == 0.0:
        return 1.0

    payoff_ratio = avg_win / avg_loss_abs
    if payoff_ratio == 0.0:
        return 0.0

    return win_rate - (loss_rate / payoff_ratio)


def _skewness(r_values: Sequence[float]) -> float:
    n = len(r_values)
    if n < 3:
        return 0.0

    mean_value = _safe_mean(r_values)
    m2 = _safe_mean([(value - mean_value) ** 2 for value in r_values])
    if m2 == 0.0:
        return 0.0

    m3 = _safe_mean([(value - mean_value) ** 3 for value in r_values])
    return m3 / (m2 ** 1.5)


def _kurtosis(r_values: Sequence[float]) -> float:
    n = len(r_values)
    if n < 4:
        return 0.0

    mean_value = _safe_mean(r_values)
    m2 = _safe_mean([(value - mean_value) ** 2 for value in r_values])
    if m2 == 0.0:
        return 0.0

    m4 = _safe_mean([(value - mean_value) ** 4 for value in r_values])
    return m4 / (m2 ** 2)


def compute_trade_metrics(
    r_values: Sequence[float],
    equity_curve: Sequence[float] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | dict[str, float]]:
    """Compute strategy-performance metrics from per-trade returns.

    Args:
        r_values: Sequence of trade returns (R-multiples or fractional returns).
        equity_curve: Optional portfolio-equity series. If omitted, a synthetic
            curve is generated from r_values.
        periods_per_year: Annualization factor used in Sharpe/Sortino.
    """

    cleaned_r_values = [float(value) for value in r_values]

    if equity_curve is None:
        curve = _build_equity_curve(cleaned_r_values)
    else:
        curve = [float(value) for value in equity_curve]

    positive = [value for value in cleaned_r_values if value > 0.0]
    negative = [value for value in cleaned_r_values if value < 0.0]

    gross_profit = sum(positive)
    gross_loss = abs(sum(negative))

    if gross_loss == 0.0:
        profit_factor = float("inf") if gross_profit > 0.0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    drawdowns = _drawdown_series(curve)
    ulcer_index = sqrt(_safe_mean([drawdown * drawdown for drawdown in drawdowns])) if drawdowns else 0.0

    percentiles = {
        "p01": _percentile(cleaned_r_values, 0.01),
        "p05": _percentile(cleaned_r_values, 0.05),
        "p50": _percentile(cleaned_r_values, 0.50),
        "p95": _percentile(cleaned_r_values, 0.95),
        "p99": _percentile(cleaned_r_values, 0.99),
    }

    return {
        "trades": float(len(cleaned_r_values)),
        "expectancy": _safe_mean(cleaned_r_values),
        "profit_factor": profit_factor,
        "winrate": (len(positive) / len(cleaned_r_values)) if cleaned_r_values else 0.0,
        "sharpe_annualized": _sharpe_annualized(cleaned_r_values, periods_per_year),
        "sortino": _sortino_annualized(cleaned_r_values, periods_per_year),
        "max_drawdown": max(drawdowns) if drawdowns else 0.0,
        "ulcer_index": ulcer_index,
        "kelly_fraction": _kelly_fraction(cleaned_r_values),
        "skewness": _skewness(cleaned_r_values),
        "kurtosis": _kurtosis(cleaned_r_values),
        "percentiles": percentiles,
    }
