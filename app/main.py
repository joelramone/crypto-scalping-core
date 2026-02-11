import random

from app.agents.strategy_agent import StrategyAgent
from app.trading.paper_wallet import PaperWallet
from app.utils import metrics


def run_single_backtest(ticks: int = 1000, initial_price: float = 50000.0) -> float:
    wallet = PaperWallet()
    strategy = StrategyAgent(wallet=wallet)

    price = initial_price
    for _ in range(ticks):
        price *= 1 + random.uniform(-0.005, 0.005)
        strategy.on_price(price)

    btc_balance = wallet.get_balance("BTC")
    if btc_balance > 0:
        wallet.sell("BTC/USDT", price, btc_balance)

    return metrics.profit_net(wallet.trades)


def run(simulations: int = 100):
    net_profits = [run_single_backtest() for _ in range(simulations)]

    profitable_runs = sum(1 for profit in net_profits if profit > 0)
    losing_runs = sum(1 for profit in net_profits if profit < 0)
    average_net_profit = sum(net_profits) / simulations if simulations else 0.0
    best_run = max(net_profits) if net_profits else 0.0
    worst_run = min(net_profits) if net_profits else 0.0

    print("--- MONTE CARLO RESULTS ---")
    print(f"Simulations: {simulations}")
    print(f"Profitable runs: {profitable_runs}")
    print(f"Losing runs: {losing_runs}")
    print(f"Average net profit: {average_net_profit:.6f}")
    print(f"Best run: {best_run:.6f}")
    print(f"Worst run: {worst_run:.6f}")


if __name__ == "__main__":
    run()
