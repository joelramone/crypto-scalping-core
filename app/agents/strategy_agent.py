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

        if not self.in_position:
            # Variación porcentual real respecto al último precio (ej: -0.3 = -0.3%)
            change_pct = ((price - self.last_price) / self.last_price) * 100
            if change_pct <= self.config["buy_threshold_pct"]:
                self.buy(price)
        else:
            # Variación porcentual real respecto al precio de entrada
            diff_pct = ((price - self.entry_price) / self.entry_price) * 100
            if (
                diff_pct >= self.config["take_profit_pct"]
                or diff_pct <= -self.config["stop_loss_pct"]
            ):
                self.sell(price)

        # Actualizar siempre al final de cada tick
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
