import random
from statistics import mean, median, pstdev
from typing import Callable, Dict, List

from importlib import util as importlib_util
from pathlib import Path

from app.agents.regime_detector import RegimeDetector
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.multi_strategy_engine import MultiStrategyEngine
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from app.trading.paper_wallet import PaperWallet, Trade

TRADE_COMMISSION_RATE = 0.001


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

            buy_price = float(getattr(pending_buy, "price", 0.0) or 0.0)
            sell_price = float(getattr(trade, "price", 0.0) or 0.0)
            quantity = float(getattr(pending_buy, "quantity", 0.0) or 0.0)
            gross_profit = (sell_price - buy_price) * quantity

            buy_fee = float(
                getattr(pending_buy, "fee", buy_price * quantity * TRADE_COMMISSION_RATE) or 0.0
            )
            sell_fee = float(
                getattr(trade, "fee", sell_price * quantity * TRADE_COMMISSION_RATE) or 0.0
            )
            pnl = gross_profit - (buy_fee + sell_fee)
            pnls.append(pnl)
            pending_buy = None

    return pnls


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
    wallet = PaperWallet()

    price = initial_price
    market_generator = market_generator_factory(initial_price)
    equity_curve: List[float] = []
    close_history: List[float] = [price]
    atr_history: List[float] = []
    rsi_history: List[float] = []

    in_position = False
    active_sl: float | None = None
    active_tp: float | None = None

    for _ in range(ticks):
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
            if (active_sl is not None and price <= active_sl) or (active_tp is not None and price >= active_tp):
                qty = wallet.get_balance("BTC")
                if qty > 0:
                    wallet.sell("BTC/USDT", price, qty)
                in_position = False
                active_sl = None
                active_tp = None
            elif signal and signal["side"] == "SHORT":
                qty = wallet.get_balance("BTC")
                if qty > 0:
                    wallet.sell("BTC/USDT", price, qty)
                in_position = False
                active_sl = None
                active_tp = None
        else:
            if signal and signal["side"] == "LONG":
                trade_usdt = min(wallet.get_balance("USDT") * 0.95, 50.0)
                if trade_usdt > 0:
                    qty = trade_usdt / price
                    wallet.buy("BTC/USDT", price, qty)
                    in_position = True
                    active_sl = float(signal["sl"])
                    active_tp = float(signal["tp"])

        equity_curve.append(wallet.total_pnl({"BTC": price}))

    btc_balance = wallet.get_balance("BTC")
    if btc_balance > 0:
        wallet.sell("BTC/USDT", price, btc_balance)

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
    }


def _print_monte_carlo_summary(title: str, results: List[Dict[str, float]], simulations: int) -> None:
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


def _run_market_monte_carlo(
    title: str,
    market_generator_factory: Callable[[float], MarketGenerator],
    simulations: int,
    strategy=None,
) -> None:
    results: List[Dict[str, float]] = []
    for simulation_index in range(1, simulations + 1):
        result = run_single_backtest(
            market_generator_factory=market_generator_factory,
            strategy=strategy,
        )
        results.append(result)

    _print_monte_carlo_summary(title=title, results=results, simulations=simulations)


def run_simulation(strategy, simulations: int = 100) -> None:
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

    print("USING MULTI STRATEGY ENGINE")

    run_simulation(strategy=strategy_engine)
