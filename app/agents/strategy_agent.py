from collections import deque

from app.trading.paper_wallet import PaperWallet
from app.config import MODE_500_USD


class StrategyAgent:
    def __init__(self, wallet: PaperWallet):
        self.wallet = wallet
        self.config = MODE_500_USD
        self.reference_price = None
        self.entry_price = None
        self.in_position = False
        self.recent_prices = deque(maxlen=5)

    def on_price(self, price: float):
        self.recent_prices.append(price)

        if self.reference_price is None:
            self.reference_price = price
            return

        if not self.in_position:
            change_pct = ((price - self.reference_price) / self.reference_price) * 100
            if change_pct <= self.config["buy_threshold_pct"] and self._has_momentum():
                self.buy(price)
        else:
            diff_pct = ((price - self.entry_price) / self.entry_price) * 100
            if diff_pct >= self.config["take_profit_pct"]:
                self.sell(price)
                self.reference_price = price
            elif diff_pct <= -self.config["stop_loss_pct"]:
                self.sell(price)
                self.reference_price = price

    def _has_momentum(self) -> bool:
        if len(self.recent_prices) < 5:
            return False

        first_price = self.recent_prices[0]
        last_price = self.recent_prices[-1]
        momentum_pct = ((last_price - first_price) / first_price) * 100
        return abs(momentum_pct) > 0.3

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
