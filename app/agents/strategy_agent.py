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
        self.volatility_history: list[float] = []

    def on_price(self, price: float):
        self.price_history.append(price)

        current_volatility = self._compute_current_volatility()
        if current_volatility is not None:
            self.volatility_history.append(current_volatility)

        if not self.in_position:
            if self._can_enter(price, current_volatility):
                self.buy(price)
        else:
            take_profit = self.entry_price * 1.008
            stop_loss = self.entry_price * 0.997

            if price >= take_profit or price <= stop_loss:
                self.sell(price)

    def _can_enter(self, price: float, current_volatility: float | None) -> bool:
        if current_volatility is None:
            return False

        if len(self.price_history) < 101:
            return False

        if len(self.volatility_history) < 30:
            return False

        sma20 = mean(self.price_history[-20:])
        sma100 = mean(self.price_history[-100:])

        volatility_percentile_60 = self._percentile(self.volatility_history[-30:], 0.60)
        trending_regime = sma20 > sma100 and current_volatility > volatility_percentile_60

        last_20_prices = self.price_history[-21:-1]
        breakout = len(last_20_prices) == 20 and price > max(last_20_prices)

        return trending_regime and breakout

    def _compute_current_volatility(self) -> float | None:
        if len(self.price_history) < 30:
            return None
        return pstdev(self.price_history[-30:])

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        sorted_values = sorted(values)
        if not sorted_values:
            return 0.0

        position = (len(sorted_values) - 1) * percentile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        interpolation = position - lower_index
        return sorted_values[lower_index] + interpolation * (
            sorted_values[upper_index] - sorted_values[lower_index]
        )

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
