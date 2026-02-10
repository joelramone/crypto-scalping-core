from app.trading.paper_wallet import PaperWallet
from app.agents.strategy_agent import StrategyAgent
from app.utils import metrics


def run():
    wallet = PaperWallet()

    strategy = StrategyAgent(wallet=wallet)

    prices = [
        50000, 49950, 49850, 49790,
        49820, 49900, 50050, 50120,
        50210, 50300, 50100, 49900
    ]

    print("Starting paper trading...\n")

    for price in prices:
        print(f"Price tick: {price}")
        strategy.on_price(price)

    print("\n--- FINAL STATE ---")
    print("Balances:", wallet.balances)
    print("Total PnL:", wallet.total_pnl({"BTC": prices[-1]}))
    print("Trades executed:", metrics.total_trades(wallet.trades))
    print("Total fees:", metrics.total_fees(wallet.trades))
    print("Net profit:", metrics.profit_net(wallet.trades))
    print("Avg profit per trade:", metrics.average_profit_per_trade(wallet.trades))


if __name__ == "__main__":
    run()
