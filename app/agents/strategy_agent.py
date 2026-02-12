from statistics import mean, pstdev

from app.config import MODE_500_USD
from app.trading.paper_wallet import PaperWallet


class StrategyAgent:
    def __init__(self, wallet: PaperWallet):
        self.wallet = wallet
        self.config = MODE_500_USD
        self.entry_price = None
        self.in_position = False
        self.price_history: list[float] = []

    def on_price(self, price: float):
        self.price_history.append(price)

        bands = self._compute_mean_reversion_bands()
        if bands is None:
            return

        sma20, lower_band = bands

        if not self.in_position:
            if price < lower_band:
                self.buy(price)
            return

        if self.entry_price is None:
            return

        stop_loss = self.entry_price * 0.995
        if price >= sma20 or price <= stop_loss:
            self.sell(price)

    def _compute_mean_reversion_bands(self) -> tuple[float, float] | None:
        if len(self.price_history) < 20:
            return None

        last_20_prices = self.price_history[-20:]
        sma20 = mean(last_20_prices)
        std20 = pstdev(last_20_prices)
        lower_band = sma20 - 2 * std20
        return sma20, lower_band

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
