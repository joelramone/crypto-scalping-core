import random
from typing import Callable, Dict, List

from app.agents.strategy_agent import StrategyAgent
from app.trading.paper_wallet import PaperWallet, Trade

TRADE_COMMISSION_RATE = 0.001


MarketGenerator = Callable[[float], float]


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
    ticks: int = 1000,
    initial_price: float = 50000.0,
) -> Dict[str, float]:
    wallet = PaperWallet()
    strategy = StrategyAgent(wallet=wallet)

    price = initial_price
    market_generator = market_generator_factory(initial_price)
    for _ in range(ticks):
        price = market_generator(price)
        strategy.on_price(price)

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

    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
    }


def _print_monte_carlo_summary(title: str, results: List[Dict[str, float]], simulations: int) -> None:
    total_runs = len(results)
    total_net_profit = sum(result["net_profit"] for result in results)
    total_gross_profit = sum(result["gross_profit"] for result in results)
    total_gross_loss = sum(result["gross_loss"] for result in results)
    total_trades_all_runs = sum(result["total_trades"] for result in results)

    average_net_profit = total_net_profit / total_runs if total_runs else 0.0
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
    average_trades_per_run = total_trades_all_runs / total_runs if total_runs else 0.0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
    expectancy_per_trade = (
        total_net_profit / total_trades_all_runs if total_trades_all_runs > 0 else 0.0
    )

    print(f"--- {title} ---")
    print(f"Simulations: {simulations}")
    print(f"Avg net profit: {average_net_profit:.6f}")
    print(f"Avg win rate: {average_win_rate:.2f}%")
    print(f"Avg trades per run: {average_trades_per_run:.2f}")
    print(f"Profit factor: {profit_factor:.6f}")
    print(f"Expectancy per trade: {expectancy_per_trade:.6f}")


def _run_market_monte_carlo(
    title: str,
    market_generator_factory: Callable[[float], MarketGenerator],
    simulations: int,
) -> None:
    results = [run_single_backtest(market_generator_factory=market_generator_factory) for _ in range(simulations)]
    _print_monte_carlo_summary(title=title, results=results, simulations=simulations)


def run(simulations: int = 100) -> None:
    scenarios = [
        ("TRENDING MARKET", _trending_market_generator),
        ("SIDEWAYS MARKET", _sideways_market_generator),
        ("HIGH VOLATILITY MARKET", _high_volatility_market_generator),
    ]

    for index, (title, generator_factory) in enumerate(scenarios):
        if index > 0:
            print()
        _run_market_monte_carlo(title=title, market_generator_factory=generator_factory, simulations=simulations)


if __name__ == "__main__":
    run()
