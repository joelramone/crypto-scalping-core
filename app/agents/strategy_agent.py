from statistics import mean, pstdev

from app.agents.regime_detector import RegimeDetector
from app.config import MODE_500_USD
from app.trading.paper_wallet import PaperWallet


class StrategyAgent:
    def __init__(self, wallet: PaperWallet):
        self.wallet = wallet
        self.config = MODE_500_USD
        self.entry_price = None
        self.stop_loss = None
        self.highest_price = None
        self.in_position = False
        self.price_history: list[float] = []
        self.regime_detector = RegimeDetector()
        self.regime_active_ticks = 0
        self.regime_evaluated_ticks = 0

    def on_price(self, price: float):
        self.price_history.append(price)
        regime = self.regime_detector.evaluate(self.price_history)
        indicators = self._compute_trend_breakout_indicators()

        if indicators is None or regime is None:
            return

        self.regime_evaluated_ticks += 1
        if not regime.sideways:
            self.regime_active_ticks += 1

        sma20, sma100, max_last_20, std_short, std_long, slope = indicators

        if not self.in_position:
            if regime.sideways:
                return
            if (
                price > max_last_20
                and sma20 > sma100
                and std_short > std_long
                and slope > 0
            ):
                self.buy(price)
            return

        if self.entry_price is None or self.stop_loss is None:
            return

        if self.highest_price is None:
            self.highest_price = self.entry_price

        if price > self.highest_price:
            self.highest_price = price

        trailing_stop = self.highest_price * 0.995

        if price <= self.stop_loss or price <= trailing_stop:
            self.sell(price)

    def _compute_trend_breakout_indicators(
        self,
    ) -> tuple[float, float, float, float, float, float] | None:
        if len(self.price_history) < 110:
            return None

        last_20_prices = self.price_history[-20:]
        last_100_prices = self.price_history[-100:]
        sma20 = mean(last_20_prices)
        sma100 = mean(last_100_prices)
        sma100_10_ticks_ago = mean(self.price_history[-110:-10])
        slope = sma100 - sma100_10_ticks_ago
        max_last_20 = max(last_20_prices)
        std_short = pstdev(last_20_prices)
        std_long = pstdev(last_100_prices)
        return sma20, sma100, max_last_20, std_short, std_long, slope

    def buy(self, price: float):
        trade_usdt = self.config["trade_size_usdt"]
        qty = trade_usdt / price
        self.wallet.buy("BTC/USDT", price, qty)
        self.entry_price = price
        self.stop_loss = price * 0.996
        self.highest_price = price
        self.in_position = True

    def sell(self, price: float):
        qty = self.wallet.get_balance("BTC")
        self.wallet.sell("BTC/USDT", price, qty)
        self.entry_price = None
        self.stop_loss = None
        self.highest_price = None
        self.in_position = False
