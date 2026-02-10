from app.trading.paper_wallet import PaperWallet
from app.config import MODE_500_USD


class StrategyAgent:
    def __init__(self, wallet: PaperWallet):
        self.wallet = wallet
        self.config = MODE_500_USD
        self.last_price = None
        self.entry_price = None
        self.in_position = False

    def on_price(self, price: float):
        if self.last_price is None:
            self.last_price = price
            return

        change_pct = ((price - self.last_price) / self.last_price) * 100

        if not self.in_position and change_pct <= self.config["buy_threshold_pct"]:
            self.buy(price)

        elif self.in_position:
            gain_pct = ((price - self.entry_price) / self.entry_price) * 100
            if gain_pct >= self.config["sell_threshold_pct"]:
                self.sell(price)

        self.last_price = price

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
