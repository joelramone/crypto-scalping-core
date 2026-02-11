from collections import deque

from app.config import MODE_500_USD
from app.trading.paper_wallet import PaperWallet


class StrategyAgent:
    def __init__(self, wallet: PaperWallet):
        self.wallet = wallet
        self.config = MODE_500_USD
        self.entry_price = None
        self.in_position = False
        self.last_20_prices = deque(maxlen=20)

    def on_price(self, price: float):
        if not self.in_position:
            if len(self.last_20_prices) == 20 and price > max(self.last_20_prices):
                self.buy(price)
        else:
            take_profit = self.entry_price * 1.008
            stop_loss = self.entry_price * 0.997

            if price >= take_profit or price <= stop_loss:
                self.sell(price)

        self.last_20_prices.append(price)

    def buy(self, price: float):
        trade_usdt = self.config["trade_size_usdt"]
        qty = trade_usdt / price
        self.wallet.buy("BTC/USDT", price, qty)
        self.entry_price = price
        self.in_position = True

    def sell(self, price: float):
        qty = self.wallet.get_balance("BTC")
        self.wallet.sell("BTC/USDT", price, qty)
        self.entry_price = None
        self.in_position = False
