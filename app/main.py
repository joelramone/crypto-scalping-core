import random
from statistics import mean, median, pstdev
from typing import Any, Callable, Dict, List
import os
import time

from importlib import util as importlib_util
from pathlib import Path

from app.agents.regime_detector import RegimeDetector
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.multi_strategy_engine import MultiStrategyEngine
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from app.trading.paper_wallet import PaperWallet, Trade
from app.utils.strategy_performance_tracker import StrategyPerformanceTracker
from app.utils.logger import configure_logging, get_logger

TRADE_COMMISSION_RATE = 0.001
BACKTEST_TICK_HARD_LIMIT = 2_000_000
HEARTBEAT_SECONDS = 5.0


MarketGenerator = Callable[[float], float]


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

    logger.info(
        "Starting backtest strategy=%s ticks=%s initial_price=%.2f",
        getattr(strategy, "__class__", type(strategy)).__name__,
        ticks,
        initial_price,
    )
    print(f"Backtest started: strategy={strategy.__class__.__name__}, ticks={ticks}", flush=True)

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
    last_output_ts = time.time()

    for tick_index in range(1, ticks + 1):
        price = market_generator(price)
        close_history.append(price)

        atr = _compute_atr(close_history)
        rsi = _compute_rsi(close_history)

        if atr is None or rsi is None:
            equity_curve.append(wallet.total_pnl({"BTC": price}))
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

        equity_curve.append(wallet.total_pnl({"BTC": price}))

        if tick_index == 1 or tick_index % 200 == 0:
            elapsed = time.time() - run_start
            logger.info("Backtest progress tick=%s/%s elapsed=%.2fs", tick_index, ticks, elapsed)
            print(f"Backtest tick {tick_index}/{ticks} | elapsed={elapsed:.2f}s", flush=True)
            last_output_ts = time.time()

        if time.time() - last_output_ts > HEARTBEAT_SECONDS:
            elapsed = time.time() - run_start
            logger.info("Backtest heartbeat tick=%s/%s elapsed=%.2fs", tick_index, ticks, elapsed)
            print(f"Backtest heartbeat... tick {tick_index}/{ticks} | elapsed={elapsed:.2f}s", flush=True)
            last_output_ts = time.time()

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

    trade_pnls = _closed_trade_pnls(wallet.trades)
    total_trades = len(trade_pnls)
    total_wins = sum(1 for pnl in trade_pnls if pnl > 0)
    total_losses = sum(1 for pnl in trade_pnls if pnl < 0)
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = sum(abs(pnl) for pnl in trade_pnls if pnl < 0)
    net_profit = sum(trade_pnls)
    average_trade_pnl = mean(trade_pnls) if trade_pnls else 0.0
    median_trade_pnl = median(trade_pnls) if trade_pnls else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    running_peak = float("-inf")
    max_drawdown = 0.0
    for equity in equity_curve:
        running_peak = max(running_peak, equity)
        drawdown = running_peak - equity
        max_drawdown = max(max_drawdown, drawdown)

    elapsed = time.time() - run_start
    logger.info("Backtest finished ticks=%s trades=%s net_profit=%.4f elapsed=%.2fs", ticks, total_trades, net_profit, elapsed)

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
        print(f"    Profit factor: {profit_factor:.6f}")
        print(f"    Expectancy: {expectancy:.6f}")

def _print_monte_carlo_summary(title: str, results: List[Dict[str, Any]], simulations: int) -> None:
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
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
    average_trade_pnl = mean(result["average_trade_pnl"] for result in results) if total_runs else 0.0
    median_trade_pnl = median(result["median_trade_pnl"] for result in results) if total_runs else 0.0
    average_max_drawdown = mean(result["max_drawdown"] for result in results) if total_runs else 0.0
    expectancy_per_trade = (
        total_net_profit / total_trades_all_runs if total_trades_all_runs > 0 else 0.0
    )
    net_profit_std = pstdev(net_profits) if len(net_profits) > 1 else 0.0
    median_net_profit = median(net_profits) if net_profits else 0.0
    best_run = max(net_profits) if net_profits else 0.0
    worst_run = min(net_profits) if net_profits else 0.0
    profitable_runs = sum(1 for value in net_profits if value > 0)

    print(f"--- {title} ---")
    print(f"Simulations: {simulations}")
    print(f"Avg net profit: {average_net_profit:.6f}")
    print(f"Median net profit: {median_net_profit:.6f}")
    print(f"Net profit std dev: {net_profit_std:.6f}")
    print(f"Best run net profit: {best_run:.6f}")
    print(f"Worst run net profit: {worst_run:.6f}")
    print(f"Profitable runs: {profitable_runs}/{total_runs}")
    print(f"Avg gross profit: {average_gross_profit:.6f}")
    print(f"Avg gross loss: {average_gross_loss:.6f}")
    print(f"Avg win rate: {average_win_rate:.2f}%")
    print(f"Weighted win rate: {weighted_win_rate:.2f}%")
    print(f"Avg trades per run: {average_trades_per_run:.2f}")
    print(f"Total trades: {total_trades_all_runs}")
    print(f"Total wins: {total_wins_all_runs}")
    print(f"Total losses: {total_losses_all_runs}")
    print(f"Profit factor: {profit_factor:.6f}")
    print(f"Expectancy per trade: {expectancy_per_trade:.6f}")
    print(f"Avg trade PnL: {average_trade_pnl:.6f}")
    print(f"Median trade PnL: {median_trade_pnl:.6f}")
    print(f"Avg max drawdown: {average_max_drawdown:.6f}")
    _print_strategy_breakdown(results)


def _run_market_monte_carlo(
    title: str,
    market_generator_factory: Callable[[float], MarketGenerator],
    simulations: int,
    strategy=None,
) -> None:
    logger = get_logger(__name__)
    run_start = time.time()
    logger.info("Scenario started scenario=%s simulations=%s", title, simulations)
    print(f"Scenario {title}: starting {simulations} simulations", flush=True)
    results: List[Dict[str, Any]] = []
    for simulation_index in range(1, simulations + 1):
        logger.info("Evaluating combination %s/%s scenario=%s strategy=%s", simulation_index, simulations, title, getattr(strategy, "__class__", type(strategy)).__name__)
        print(f"Evaluating combination {simulation_index}/{simulations} ({title})", flush=True)
        result = run_single_backtest(
            market_generator_factory=market_generator_factory,
            strategy=strategy,
        )
        results.append(result)

    logger.info("Scenario finished scenario=%s elapsed=%.2fs", title, time.time() - run_start)
    _print_monte_carlo_summary(title=title, results=results, simulations=simulations)


def run_simulation(strategy, simulations: int = 1) -> None:
    scenarios = [
        ("TRENDING", _trending_market_generator),
        ("SIDEWAYS", _sideways_market_generator),
        ("HIGH VOLATILITY", _high_volatility_market_generator),
    ]

    for index, (title, generator_factory) in enumerate(scenarios):
        if index > 0:
            print()
        _run_market_monte_carlo(
            title=title,
            market_generator_factory=generator_factory,
            simulations=simulations,
            strategy=strategy,
        )


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

    simulations = int(os.getenv("MONTE_CARLO_SIMULATIONS", "1"))
    logger.info("Using multi strategy engine simulations=%s", simulations)
    print(f"USING MULTI STRATEGY ENGINE (simulations={simulations})", flush=True)

    run_simulation(strategy=strategy_engine, simulations=simulations)
    logger.info("Total execution time: %.2fs", time.time() - start_time)
    print(f"Total execution time: {time.time() - start_time:.2f}s", flush=True)
