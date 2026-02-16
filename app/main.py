import csv
import json
import random
from statistics import mean, median, pstdev
from typing import Any, Callable, Dict, List
import os
import time
from datetime import UTC, datetime

from importlib import util as importlib_util
from pathlib import Path

from app.agents.regime_detector import RegimeDetector
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.multi_strategy_engine import MultiStrategyEngine
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from app.trading.paper_wallet import PaperWallet, Trade
from app.utils.strategy_performance_tracker import StrategyPerformanceTracker
from app.utils.logger import configure_logging, get_logger
from app.utils.datetime_utils import ensure_utc

TRADE_COMMISSION_RATE = 0.001
BACKTEST_TICK_HARD_LIMIT = 2_000_000
DEBUG_METRICS = os.getenv("DEBUG_METRICS", "0") == "1"
TRADES_AUDIT_CSV_PATH = "trades_audit.csv"
TRADES_AUDIT_HEADERS = [
    "timestamp",
    "strategy_name",
    "regime",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "fee",
    "net_pnl",
    "drawdown_at_trade",
]


MarketGenerator = Callable[[float], float]


class TradeAuditLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._file = None
        self._writer = None

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return

        file_exists = os.path.exists(self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        should_write_header = True
        if file_exists:
            should_write_header = os.path.getsize(self.path) == 0

        if should_write_header:
            self._writer.writerow(TRADES_AUDIT_HEADERS)
            self._file.flush()

    def record_trade(
        self,
        *,
        timestamp: datetime,
        strategy_name: str,
        regime: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        gross_pnl: float,
        fee: float,
        net_pnl: float,
        drawdown_at_trade: float,
    ) -> None:
        try:
            self._ensure_writer()
            if self._writer is None or self._file is None:
                return

            self._writer.writerow(
                [
                    timestamp.isoformat(),
                    strategy_name,
                    regime,
                    side,
                    entry_price,
                    exit_price,
                    quantity,
                    gross_pnl,
                    fee,
                    net_pnl,
                    drawdown_at_trade,
                ]
            )
            self._file.flush()
        except OSError as exc:
            get_logger(__name__).warning("Unable to write trade audit row: %s", exc)


TRADE_AUDIT_LOGGER = TradeAuditLogger(TRADES_AUDIT_CSV_PATH)


def _load_breakout_strategy_config_class():
    config_path = Path(__file__).resolve().parent / "config" / "breakout_config.py"
    spec = importlib_util.spec_from_file_location("app.config.breakout_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load BreakoutStrategyConfig")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BreakoutStrategyConfig


def _load_rsi_strategy_config_class():
    config_path = Path(__file__).resolve().parent / "config" / "rsi_config.py"
    spec = importlib_util.spec_from_file_location("app.config.rsi_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load RSIStrategyConfig")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RSIStrategyConfig


def _compute_atr(closes: List[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None

    true_ranges = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period


def _compute_rsi(closes: List[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None

    gains: List[float] = []
    losses: List[float] = []
    for index in range(len(closes) - period, len(closes)):
        delta = closes[index] - closes[index - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _closed_trade_pnls(trades: List[Trade]) -> List[float]:
    pnls: List[float] = []
    pending_buy: Trade | None = None

    for trade in trades:
        side = str(getattr(trade, "side", "")).lower()

        if side == "buy":
            pending_buy = trade
            continue

        if side == "sell" and pending_buy is not None:
            if getattr(pending_buy, "symbol", None) != getattr(trade, "symbol", None):
                continue

            pnl = _trade_pnl_from_pair(pending_buy, trade)
            pnls.append(pnl)
            pending_buy = None

    return pnls

def _trade_pnl_from_pair(buy_trade: Trade, sell_trade: Trade) -> float:
    buy_price = float(getattr(buy_trade, "price", 0.0) or 0.0)
    sell_price = float(getattr(sell_trade, "price", 0.0) or 0.0)
    quantity = float(getattr(buy_trade, "quantity", 0.0) or 0.0)
    gross_profit = (sell_price - buy_price) * quantity

    buy_fee = float(
        getattr(buy_trade, "fee", buy_price * quantity * TRADE_COMMISSION_RATE) or 0.0
    )
    sell_fee = float(
        getattr(sell_trade, "fee", sell_price * quantity * TRADE_COMMISSION_RATE) or 0.0
    )
    return gross_profit - (buy_fee + sell_fee)


def _resolve_signal_context(strategy: Any) -> dict[str, str] | None:
    consume_context = getattr(strategy, "consume_last_signal_context", None)
    if callable(consume_context):
        context = consume_context()
        if isinstance(context, dict):
            return context
    return None


def _record_trade_outcome(strategy: Any, trade_result: str, pnl: float, context: dict[str, str] | None) -> None:
    record_outcome = getattr(strategy, "record_trade_outcome", None)
    if callable(record_outcome):
        record_outcome(trade_result=trade_result, pnl=pnl, context=context)


def _audit_closed_trade(
    *,
    strategy: Any,
    context: dict[str, str] | None,
    buy_trade: Trade,
    sell_trade: Trade,
    running_peak_equity: float,
    current_equity: float,
) -> None:
    entry_price = float(getattr(buy_trade, "price", 0.0) or 0.0)
    exit_price = float(getattr(sell_trade, "price", 0.0) or 0.0)
    quantity = float(getattr(buy_trade, "quantity", 0.0) or 0.0)
    gross_pnl = (exit_price - entry_price) * quantity
    buy_fee = float(getattr(buy_trade, "fee", entry_price * quantity * TRADE_COMMISSION_RATE) or 0.0)
    sell_fee = float(getattr(sell_trade, "fee", exit_price * quantity * TRADE_COMMISSION_RATE) or 0.0)
    total_fee = buy_fee + sell_fee
    net_pnl = gross_pnl - total_fee
    drawdown_at_trade = max(0.0, running_peak_equity - current_equity)

    TRADE_AUDIT_LOGGER.record_trade(
        timestamp=ensure_utc(getattr(sell_trade, "timestamp", datetime.now(UTC))),
        strategy_name=strategy.__class__.__name__,
        regime=(context or {}).get("regime_detected", "unknown"),
        side="LONG",
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        gross_pnl=gross_pnl,
        fee=total_fee,
        net_pnl=net_pnl,
        drawdown_at_trade=drawdown_at_trade,
    )




def _trending_market_generator(initial_price: float) -> MarketGenerator:
    direction = random.choice((-1.0, 1.0))
    drift = 0.0007 * direction
    trend_anchor = initial_price

    def next_price(price: float) -> float:
        nonlocal trend_anchor
        trend_anchor = (trend_anchor * 0.995) + (price * 0.005)
        reversion_component = -0.01 * ((price - trend_anchor) / trend_anchor)
        random_component = random.uniform(-0.004, 0.004)
        price_change = drift + random_component + reversion_component
        return max(100.0, price * (1 + price_change))

    return next_price


def _sideways_market_generator(initial_price: float) -> MarketGenerator:
    mean_price = initial_price

    def next_price(price: float) -> float:
        nonlocal mean_price
        mean_price = (mean_price * 0.995) + (price * 0.005)
        drift = 0.0
        reversion_component = -0.08 * ((price - mean_price) / mean_price)
        random_component = random.uniform(-0.003, 0.003)
        price_change = drift + reversion_component + random_component
        return max(100.0, price * (1 + price_change))

    return next_price


def _high_volatility_market_generator(initial_price: float) -> MarketGenerator:
    _ = initial_price
    direction = random.choice((-1.0, 1.0))

    def next_price(price: float) -> float:
        nonlocal direction
        if random.random() < 0.35:
            direction *= -1.0

        shock_size = random.uniform(0.001, 0.015)
        random_component = random.uniform(-0.012, 0.012)
        drift = 0.0
        price_change = drift + (direction * shock_size) + random_component
        return max(100.0, price * (1 + price_change))

    return next_price


def run_single_backtest(
    market_generator_factory: Callable[[float], MarketGenerator],
    strategy,
    ticks: int = 1000,
    initial_price: float = 50000.0,
) -> Dict[str, float]:
    logger = get_logger(__name__)
    run_start = time.time()

    if ticks <= 0:
        raise ValueError(f"ticks must be > 0, got {ticks}")
    if ticks > BACKTEST_TICK_HARD_LIMIT:
        logger.warning("ticks=%s exceeds hard limit=%s, capping to avoid runaway loops", ticks, BACKTEST_TICK_HARD_LIMIT)
        ticks = BACKTEST_TICK_HARD_LIMIT

    wallet = PaperWallet()
    strategy_performance_tracker = StrategyPerformanceTracker()

    price = initial_price
    market_generator = market_generator_factory(initial_price)
    equity_curve: List[float] = []
    close_history: List[float] = [price]
    atr_history: List[float] = []
    rsi_history: List[float] = []

    in_position = False
    active_sl: float | None = None
    active_tp: float | None = None
    active_trade_context: dict[str, str] | None = None
    active_buy_trade: Trade | None = None
    running_peak_equity = wallet.total_pnl({"BTC": price})

    for tick_index in range(1, ticks + 1):
        price = market_generator(price)
        signal_close_history = close_history.copy()
        close_history.append(price)

        atr = _compute_atr(signal_close_history)
        rsi = _compute_rsi(signal_close_history)

        if atr is None or rsi is None:
            current_equity = wallet.total_pnl({"BTC": price})
            equity_curve.append(current_equity)
            running_peak_equity = max(running_peak_equity, current_equity)
            continue

        atr_history.append(atr)
        rsi_history.append(rsi)

        signal = strategy.generate_signal({"atr": atr_history, "close": close_history, "rsi": rsi_history})

        if in_position:
            should_close_position = False
            if (active_sl is not None and price <= active_sl) or (active_tp is not None and price >= active_tp):
                should_close_position = True
            elif signal and signal["side"] == "SHORT":
                should_close_position = True

            if should_close_position:
                qty = wallet.get_balance("BTC")
                if qty > 0:
                    wallet.sell("BTC/USDT", price, qty)
                    sell_trade = wallet.trades[-1]
                    if active_buy_trade is not None:
                        trade_pnl = _trade_pnl_from_pair(active_buy_trade, sell_trade)
                        trade_result = "WIN" if trade_pnl > 0 else "LOSS" if trade_pnl < 0 else "BREAKEVEN"
                        if active_trade_context is not None:
                            strategy_performance_tracker.record_trade(
                                strategy_name=active_trade_context.get("strategy_name", "unknown"),
                                pnl=trade_pnl,
                            )
                        _record_trade_outcome(
                            strategy=strategy,
                            trade_result=trade_result,
                            pnl=trade_pnl,
                            context=active_trade_context,
                        )
                        current_equity = wallet.total_pnl({"BTC": price})
                        _audit_closed_trade(
                            strategy=strategy,
                            context=active_trade_context,
                            buy_trade=active_buy_trade,
                            sell_trade=sell_trade,
                            running_peak_equity=running_peak_equity,
                            current_equity=current_equity,
                        )
                in_position = False
                active_sl = None
                active_tp = None
                active_trade_context = None
                active_buy_trade = None
        else:
            if signal and signal["side"] == "LONG":
                trade_usdt = min(wallet.get_balance("USDT") * 0.95, 50.0)
                if trade_usdt > 0:
                    qty = trade_usdt / price
                    wallet.buy("BTC/USDT", price, qty)
                    in_position = True
                    active_sl = float(signal["sl"])
                    active_tp = float(signal["tp"])
                    active_trade_context = _resolve_signal_context(strategy)
                    active_buy_trade = wallet.trades[-1]

        current_equity = wallet.total_pnl({"BTC": price})
        equity_curve.append(current_equity)
        running_peak_equity = max(running_peak_equity, current_equity)

    btc_balance = wallet.get_balance("BTC")
    if btc_balance > 0:
        wallet.sell("BTC/USDT", price, btc_balance)
        sell_trade = wallet.trades[-1]
        if active_buy_trade is not None:
            trade_pnl = _trade_pnl_from_pair(active_buy_trade, sell_trade)
            trade_result = "WIN" if trade_pnl > 0 else "LOSS" if trade_pnl < 0 else "BREAKEVEN"
            if active_trade_context is not None:
                strategy_performance_tracker.record_trade(
                    strategy_name=active_trade_context.get("strategy_name", "unknown"),
                    pnl=trade_pnl,
                )
            _record_trade_outcome(
                strategy=strategy,
                trade_result=trade_result,
                pnl=trade_pnl,
                context=active_trade_context,
            )
            current_equity = wallet.total_pnl({"BTC": price})
            _audit_closed_trade(
                strategy=strategy,
                context=active_trade_context,
                buy_trade=active_buy_trade,
                sell_trade=sell_trade,
                running_peak_equity=running_peak_equity,
                current_equity=current_equity,
            )

    trade_pnls = _closed_trade_pnls(wallet.trades)
    total_trades = len(trade_pnls)
    total_wins = sum(1 for pnl in trade_pnls if pnl > 0)
    total_losses = sum(1 for pnl in trade_pnls if pnl < 0)
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = sum(abs(pnl) for pnl in trade_pnls if pnl < 0)
    net_profit = sum(trade_pnls)
    average_trade_pnl = mean(trade_pnls) if trade_pnls else 0.0
    median_trade_pnl = median(trade_pnls) if trade_pnls else 0.0
    profit_factor = StrategyPerformanceTracker.compute_profit_factor({"gross_profit": gross_profit, "gross_loss": gross_loss})

    running_peak = float("-inf")
    max_drawdown = 0.0
    for equity in equity_curve:
        running_peak = max(running_peak, equity)
        drawdown = running_peak - equity
        max_drawdown = max(max_drawdown, drawdown)


    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "average_trade_pnl": average_trade_pnl,
        "median_trade_pnl": median_trade_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "strategy_breakdown": strategy_performance_tracker.export(),
    }



def _aggregate_strategy_breakdown(results: List[Dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}

    for result in results:
        strategy_breakdown = result.get("strategy_breakdown", {})
        if not isinstance(strategy_breakdown, dict):
            continue

        for strategy_name, stats in strategy_breakdown.items():
            if not isinstance(stats, dict):
                continue

            current = aggregated.setdefault(
                strategy_name,
                {
                    "trades_count": 0,
                    "wins": 0,
                    "losses": 0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "net_profit": 0.0,
                },
            )
            current["trades_count"] += int(stats.get("trades_count", 0))
            current["wins"] += int(stats.get("wins", 0))
            current["losses"] += int(stats.get("losses", 0))
            current["gross_profit"] += float(stats.get("gross_profit", 0.0))
            current["gross_loss"] += float(stats.get("gross_loss", 0.0))
            current["net_profit"] += float(stats.get("net_profit", 0.0))

    return aggregated


def _format_profit_factor(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.6f}"


def _print_strategy_breakdown(results: List[Dict[str, Any]]) -> None:
    aggregated = _aggregate_strategy_breakdown(results)
    if not aggregated:
        return

    print("--- STRATEGY BREAKDOWN ---")
    for strategy_name in sorted(aggregated.keys()):
        stats = aggregated[strategy_name]
        win_rate = StrategyPerformanceTracker.compute_win_rate(stats)
        profit_factor = StrategyPerformanceTracker.compute_profit_factor(stats)
        expectancy = StrategyPerformanceTracker.compute_expectancy(stats)

        print(f"{strategy_name}:")
        print(f"    Trades: {int(stats['trades_count'])}")
        print(f"    Win rate: {win_rate:.2f}%")
        print(f"    Profit factor: {_format_profit_factor(profit_factor)}")
        print(f"    Expectancy: {expectancy:.6f}")

        if DEBUG_METRICS:
            print(
                "    [DEBUG_METRICS] "
                f"gross_profit={float(stats.get('gross_profit', 0.0)):.6f} "
                f"gross_loss={float(stats.get('gross_loss', 0.0)):.6f} "
                f"total_trades={int(stats.get('trades_count', 0))} "
                f"total_wins={int(stats.get('wins', 0))} "
                f"total_losses={int(stats.get('losses', 0))}"
            )

def _compute_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def _build_monte_carlo_summary(results: List[Dict[str, Any]], simulations: int) -> Dict[str, Any]:
    total_runs = len(results)
    net_profits = [result["net_profit"] for result in results]
    total_net_profit = sum(net_profits)
    total_gross_profit = sum(result["gross_profit"] for result in results)
    total_gross_loss = sum(result["gross_loss"] for result in results)
    total_trades_all_runs = sum(result["total_trades"] for result in results)
    total_wins_all_runs = sum(result["total_wins"] for result in results)
    total_losses_all_runs = sum(result["total_losses"] for result in results)

    average_net_profit = total_net_profit / total_runs if total_runs else 0.0
    average_gross_profit = total_gross_profit / total_runs if total_runs else 0.0
    average_gross_loss = total_gross_loss / total_runs if total_runs else 0.0
    average_win_rate = (
        sum(
            (result["total_wins"] / result["total_trades"]) * 100
            for result in results
            if result["total_trades"] > 0
        )
        / total_runs
        if total_runs
        else 0.0
    )
    weighted_win_rate = (
        (total_wins_all_runs / total_trades_all_runs) * 100 if total_trades_all_runs > 0 else 0.0
    )
    average_trades_per_run = total_trades_all_runs / total_runs if total_runs else 0.0
    average_profit_factor = mean(
        StrategyPerformanceTracker.compute_profit_factor(result) for result in results
    ) if total_runs else 0.0
    aggregate_profit_factor = StrategyPerformanceTracker.compute_profit_factor(
        {"gross_profit": total_gross_profit, "gross_loss": total_gross_loss}
    )
    average_trade_pnl = mean(result["average_trade_pnl"] for result in results) if total_runs else 0.0
    median_trade_pnl = median(result["median_trade_pnl"] for result in results) if total_runs else 0.0
    average_max_drawdown = mean(result["max_drawdown"] for result in results) if total_runs else 0.0
    worst_max_drawdown = max((result["max_drawdown"] for result in results), default=0.0)
    average_expectancy = mean(
        StrategyPerformanceTracker.compute_expectancy(result) for result in results
    ) if total_runs else 0.0
    expectancy_per_trade = (
        total_net_profit / total_trades_all_runs if total_trades_all_runs > 0 else 0.0
    )
    net_profit_std = pstdev(net_profits) if len(net_profits) > 1 else 0.0
    median_net_profit = median(net_profits) if net_profits else 0.0
    percentile_5_net_profit = _compute_percentile(net_profits, 0.05)
    percentile_95_net_profit = _compute_percentile(net_profits, 0.95)
    best_run = max(net_profits) if net_profits else 0.0
    worst_run = min(net_profits) if net_profits else 0.0
    profitable_runs = sum(1 for value in net_profits if value > 0)
    profitable_run_pct = ((profitable_runs / total_runs) * 100) if total_runs else 0.0
    sharpe_approx = (average_net_profit / net_profit_std) if net_profit_std > 0 else 0.0
    coefficient_of_variation = (net_profit_std / average_net_profit) if average_net_profit != 0 else 0.0
    drawdowns = [float(result.get("max_drawdown", 0.0)) for result in results]
    trades_per_run = [int(result.get("total_trades", 0)) for result in results]

    return {
        "simulations": simulations,
        "total_runs": total_runs,
        "avg_net_profit": average_net_profit,
        "median_net_profit": median_net_profit,
        "net_profit_std_dev": net_profit_std,
        "p5_net_profit": percentile_5_net_profit,
        "p95_net_profit": percentile_95_net_profit,
        "best_run_net_profit": best_run,
        "worst_run_net_profit": worst_run,
        "profitable_runs": profitable_runs,
        "profitable_run_pct": profitable_run_pct,
        "avg_gross_profit": average_gross_profit,
        "avg_gross_loss": average_gross_loss,
        "avg_win_rate": average_win_rate,
        "weighted_win_rate": weighted_win_rate,
        "avg_trades_per_run": average_trades_per_run,
        "total_trades": total_trades_all_runs,
        "total_wins": total_wins_all_runs,
        "total_losses": total_losses_all_runs,
        "avg_profit_factor": average_profit_factor,
        "aggregate_profit_factor": aggregate_profit_factor,
        "avg_expectancy": average_expectancy,
        "expectancy_per_trade": expectancy_per_trade,
        "avg_trade_pnl": average_trade_pnl,
        "median_trade_pnl": median_trade_pnl,
        "avg_max_drawdown": average_max_drawdown,
        "worst_max_drawdown": worst_max_drawdown,
        "sharpe_ratio_approx": sharpe_approx,
        "coefficient_of_variation": coefficient_of_variation,
        "_net_profits": [float(value) for value in net_profits],
        "_max_drawdowns": drawdowns,
        "_trades_per_run": trades_per_run,
    }


def _build_final_monte_carlo_report(summary: Dict[str, float]) -> Dict[str, Any]:
    total_simulations = int(summary.get("total_runs", 0))
    std_pnl = float(summary.get("net_profit_std_dev", 0.0))
    avg_trades = float(summary.get("avg_trades_per_run", 0.0))

    warnings: List[str] = []
    if std_pnl == 0.0:
        warnings.append("WARNING: PnL std deviation is zero; Sharpe ratio is not informative.")
    if avg_trades < 10:
        warnings.append("WARNING: Low sample size (<10 trades per simulation on average).")

    return {
        "total_simulations": total_simulations,
        "mean_pnl": float(summary.get("avg_net_profit", 0.0)),
        "median_pnl": float(summary.get("median_net_profit", 0.0)),
        "std_pnl": std_pnl,
        "sharpe": float(summary.get("sharpe_ratio_approx", 0.0)),
        "worst_drawdown": float(summary.get("worst_max_drawdown", 0.0)),
        "avg_drawdown": float(summary.get("avg_max_drawdown", 0.0)),
        "winrate": float(summary.get("weighted_win_rate", 0.0)),
        "profitable_simulations_pct": float(summary.get("profitable_run_pct", 0.0)),
        "expectancy_per_trade": float(summary.get("expectancy_per_trade", 0.0)),
        "avg_trades_per_simulation": avg_trades,
        "fees_included": True,
        "slippage_included": True,
        "equity_reset_per_simulation": True,
        "lookahead_bias_prevention": "Signals use rolling historical data available up to current tick only.",
        "warnings": warnings,
    }


def _print_final_monte_carlo_report(report: Dict[str, Any]) -> None:
    print("===== MONTECARLO FINAL REPORT =====")
    print(f"Total simulations: {report['total_simulations']}")
    print(f"Mean PnL: {report['mean_pnl']:.6f}")
    print(f"Median PnL: {report['median_pnl']:.6f}")
    print(f"Std PnL: {report['std_pnl']:.6f}")
    print(f"Sharpe: {report['sharpe']:.6f}")
    print(f"Worst DD: {report['worst_drawdown']:.6f}")
    print(f"Avg DD: {report['avg_drawdown']:.6f}")
    print(f"Winrate: {report['winrate']:.2f}%")
    print(f"% Profitable Sims: {report['profitable_simulations_pct']:.2f}%")
    print(f"Expectancy: {report['expectancy_per_trade']:.6f}")
    print(f"Avg trades: {report['avg_trades_per_simulation']:.2f}")

    for warning in report.get("warnings", []):
        print(warning)


def _save_monte_carlo_report_json(report: Dict[str, Any], output_path: str = "montecarlo_report.json") -> None:
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
    print(f"Saved Monte Carlo report JSON: {output_path}")


def _print_monte_carlo_summary(title: str, results: List[Dict[str, Any]], simulations: int) -> Dict[str, float]:
    summary = _build_monte_carlo_summary(results=results, simulations=simulations)

    print(f"--- {title} ---")
    print(f"Simulations: {simulations}")
    print(f"Avg net profit: {summary['avg_net_profit']:.6f}")
    print(f"Median net profit: {summary['median_net_profit']:.6f}")
    print(f"Net profit std dev: {summary['net_profit_std_dev']:.6f}")
    print(f"5th percentile net profit: {summary['p5_net_profit']:.6f}")
    print(f"95th percentile net profit: {summary['p95_net_profit']:.6f}")
    print(f"Best run net profit: {summary['best_run_net_profit']:.6f}")
    print(f"Worst run net profit: {summary['worst_run_net_profit']:.6f}")
    print(f"Profitable runs: {int(summary['profitable_runs'])}/{int(summary['total_runs'])}")
    print(f"% runs rentables: {summary['profitable_run_pct']:.2f}%")
    print(f"Avg gross profit: {summary['avg_gross_profit']:.6f}")
    print(f"Avg gross loss: {summary['avg_gross_loss']:.6f}")
    print(f"Avg win rate: {summary['avg_win_rate']:.2f}%")
    print(f"Weighted win rate: {summary['weighted_win_rate']:.2f}%")
    print(f"Avg trades per run: {summary['avg_trades_per_run']:.2f}")
    print(f"Total trades: {int(summary['total_trades'])}")
    print(f"Total wins: {int(summary['total_wins'])}")
    print(f"Total losses: {int(summary['total_losses'])}")
    print(f"Avg profit factor: {_format_profit_factor(summary['avg_profit_factor'])}")
    print(f"Aggregate profit factor: {_format_profit_factor(summary['aggregate_profit_factor'])}")
    print(f"Avg expectancy: {summary['avg_expectancy']:.6f}")
    print(f"Expectancy per trade: {summary['expectancy_per_trade']:.6f}")
    print(f"Avg trade PnL: {summary['avg_trade_pnl']:.6f}")
    print(f"Median trade PnL: {summary['median_trade_pnl']:.6f}")
    print(f"Max drawdown promedio: {summary['avg_max_drawdown']:.6f}")
    print(f"Sharpe ratio aprox: {summary['sharpe_ratio_approx']:.6f}")
    print(f"Coefficient of variation: {summary['coefficient_of_variation']:.6f}")
    _print_strategy_breakdown(results)
    return summary


def _run_market_monte_carlo(
    title: str,
    market_generator_factory: Callable[[float], MarketGenerator],
    simulations: int,
    strategy=None,
) -> Dict[str, float]:
    run_start = time.time()
    results: List[Dict[str, Any]] = []
    for simulation_index in range(1, simulations + 1):
        if hasattr(strategy, "regime_detector") and hasattr(strategy.regime_detector, "reset_state"):
            strategy.regime_detector.reset_state()

        result = run_single_backtest(
            market_generator_factory=market_generator_factory,
            strategy=strategy,
        )
        results.append(result)

    total_elapsed = time.time() - run_start
    summary = _print_monte_carlo_summary(title=title, results=results, simulations=simulations)
    summary["elapsed_seconds"] = total_elapsed
    return summary


def _save_research_csv(results: Dict[str, Dict[str, float]], output_path: str = "research_results_300sims.csv") -> None:
    if not results:
        return

    fieldnames = ["regime"]
    first_regime = next(iter(results.values()))
    fieldnames.extend(first_regime.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for regime, metrics in results.items():
            row = {"regime": regime}
            row.update(metrics)
            writer.writerow(row)

    print(f"Saved research CSV: {output_path}", flush=True)


def _print_research_summary(results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        return

    print("\n=== RESEARCH SUMMARY ===")

    by_avg_net = max(results.items(), key=lambda item: item[1].get("avg_net_profit", float("-inf")))
    print(f"Regime con edge (mayor avg net profit): {by_avg_net[0]} ({by_avg_net[1]['avg_net_profit']:.6f})")

    pf_gt_1 = [regime for regime, metrics in results.items() if metrics.get("avg_profit_factor", 0.0) > 1.0]
    expectancy_gt_0 = [regime for regime, metrics in results.items() if metrics.get("avg_expectancy", 0.0) > 0.0]
    excessive_dd = [
        regime
        for regime, metrics in results.items()
        if metrics.get("avg_max_drawdown", 0.0) > abs(metrics.get("avg_net_profit", 0.0))
    ]

    print(f"Regímenes con PF > 1: {', '.join(pf_gt_1) if pf_gt_1 else 'Ninguno'}")
    print(f"Regímenes con expectancy > 0: {', '.join(expectancy_gt_0) if expectancy_gt_0 else 'Ninguno'}")
    print(f"Regímenes con drawdown excesivo: {', '.join(excessive_dd) if excessive_dd else 'Ninguno'}")




def _aggregate_final_report_from_scenarios(research_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    all_net_profits: List[float] = []
    all_drawdowns: List[float] = []
    all_trades_per_run: List[int] = []
    total_trades = 0
    total_wins = 0

    for summary in research_results.values():
        all_net_profits.extend(float(v) for v in summary.get("_net_profits", []))
        all_drawdowns.extend(float(v) for v in summary.get("_max_drawdowns", []))
        all_trades_per_run.extend(int(v) for v in summary.get("_trades_per_run", []))
        total_trades += int(summary.get("total_trades", 0))
        total_wins += int(summary.get("total_wins", 0))

    combined = {
        "total_runs": len(all_net_profits),
        "avg_net_profit": mean(all_net_profits) if all_net_profits else 0.0,
        "median_net_profit": median(all_net_profits) if all_net_profits else 0.0,
        "net_profit_std_dev": pstdev(all_net_profits) if len(all_net_profits) > 1 else 0.0,
        "sharpe_ratio_approx": (mean(all_net_profits) / pstdev(all_net_profits)) if len(all_net_profits) > 1 and pstdev(all_net_profits) > 0 else 0.0,
        "worst_max_drawdown": max(all_drawdowns) if all_drawdowns else 0.0,
        "avg_max_drawdown": mean(all_drawdowns) if all_drawdowns else 0.0,
        "weighted_win_rate": ((total_wins / total_trades) * 100.0) if total_trades > 0 else 0.0,
        "profitable_run_pct": ((sum(1 for v in all_net_profits if v > 0) / len(all_net_profits)) * 100.0) if all_net_profits else 0.0,
        "expectancy_per_trade": (sum(all_net_profits) / total_trades) if total_trades > 0 else 0.0,
        "avg_trades_per_run": mean(all_trades_per_run) if all_trades_per_run else 0.0,
    }
    return _build_final_monte_carlo_report(combined)


def run_simulation(strategy, simulations: int = 300) -> Dict[str, Dict[str, float]]:
    scenarios = [
        ("TRENDING", _trending_market_generator),
        ("SIDEWAYS", _sideways_market_generator),
        ("HIGH_VOL", _high_volatility_market_generator),
    ]

    research_results: Dict[str, Dict[str, float]] = {}
    for index, (title, generator_factory) in enumerate(scenarios):
        if index > 0:
            print()
        summary = _run_market_monte_carlo(
            title=title,
            market_generator_factory=generator_factory,
            simulations=simulations,
            strategy=strategy,
        )
        research_results[title] = summary

    _save_research_csv(research_results)
    _print_research_summary(research_results)

    final_report = _aggregate_final_report_from_scenarios(research_results)
    _print_final_monte_carlo_report(final_report)
    _save_monte_carlo_report_json(final_report)
    return research_results


if __name__ == "__main__":
    configure_logging()
    logger = get_logger(__name__)
    start_time = time.time()
    logger.info("Starting app.main")
    BreakoutStrategyConfig = _load_breakout_strategy_config_class()
    RSIStrategyConfig = _load_rsi_strategy_config_class()

    breakout_config = BreakoutStrategyConfig(
        lookback_period=20,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )
    rsi_config = RSIStrategyConfig(
        rsi_oversold=30,
        rsi_overbought=70,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=2.0,
    )

    regime_detector = RegimeDetector()
    rsi_strategy = RSIMeanReversionStrategy(config=rsi_config)
    breakout_strategy = BreakoutTrendStrategy(config=breakout_config)
    strategy_engine = MultiStrategyEngine(
        regime_detector=regime_detector,
        rsi_strategy=rsi_strategy,
        breakout_strategy=breakout_strategy,
    )

    simulations = int(os.getenv("MONTE_CARLO_SIMULATIONS", "300"))
    logger.info("Using multi strategy engine simulations=%s", simulations)
    print(f"USING MULTI STRATEGY ENGINE (simulations={simulations})", flush=True)

    results = run_simulation(strategy=strategy_engine, simulations=simulations)
    logger.info("Research regimes evaluated=%s", list(results.keys()))
    logger.info("Total execution time: %.2fs", time.time() - start_time)
    print(f"Total execution time: {time.time() - start_time:.2f}s", flush=True)
