import random

from app.agents.strategy_agent import StrategyAgent
from app.trading.paper_wallet import PaperWallet
from app.utils import metrics


def run():
    wallet = PaperWallet()
    strategy = StrategyAgent(wallet=wallet)

    initial_price = 50000
    ticks = 1000

    prices = []
    price = initial_price
    for _ in range(ticks):
        prices.append(price)
        price *= (1 + random.uniform(-0.005, 0.005))

    for price in prices:
        strategy.on_price(price)

    btc_balance = wallet.get_balance("BTC")
    if btc_balance > 0:
        last_price = prices[-1]
        wallet.sell("BTC/USDT", last_price, btc_balance)

    gross_profit = metrics.profit_gross(wallet.trades)
    net_profit = metrics.profit_net(wallet.trades)
    total_fees = metrics.total_fees(wallet.trades)
    trades_executed = metrics.total_trades(wallet.trades)
    avg_profit_per_trade = metrics.average_profit_per_trade(wallet.trades)
    final_balance = wallet.total_pnl({"BTC": prices[-1]})

    print("--- FINAL STATE ---")
    print("Trades executed:", trades_executed)
    print("Gross profit:", gross_profit)
    print("Net profit:", net_profit)
    print("Total fees:", total_fees)
    print("Avg profit per trade:", avg_profit_per_trade)
    print("Final balances:", final_balance)


if __name__ == "__main__":
    run()
