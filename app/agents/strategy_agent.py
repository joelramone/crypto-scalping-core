from app.trading.paper_wallet import PaperWallet


class StrategyAgent:
    def __init__(self, wallet: PaperWallet, buy_below: float, sell_above: float):
        self.wallet = wallet
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.in_position = False

    def on_price(self, price: float):
        if not self.in_position and price <= self.buy_below:
            self.buy(price)

        elif self.in_position and price >= self.sell_above:
            self.sell(price)

    def buy(self, price: float):
        qty = 0.001  # fixed size for now
        self.wallet.buy("BTC/USDT", price, qty)
        self.in_position = True

    def sell(self, price: float):
        qty = 0.001
        self.wallet.sell("BTC/USDT", price, qty)
        self.in_position = False
