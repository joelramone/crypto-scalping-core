import unittest

from app.agents.regime_detector import RegimeDetector
from app.agents.strategy_agent import StrategyAgent
from app.trading.paper_wallet import PaperWallet


class RegimeDetectorTests(unittest.TestCase):
    def test_detects_high_vol_expansion_when_all_conditions_are_met(self):
        detector = RegimeDetector(momentum_threshold=0.01)

        historical_prices: list[float] = []
        for i in range(130):
            historical_prices.append(100.0 + (0.01 * i))
            detector.evaluate(historical_prices)

        expansion_prices = historical_prices + [140.0, 145.0, 150.0, 155.0, 160.0]
        regime = detector.evaluate(expansion_prices)

        self.assertIsNotNone(regime)
        assert regime is not None
        self.assertTrue(regime.high_vol_expansion)
        self.assertGreater(regime.std_short, regime.std_long)
        self.assertGreater(abs(regime.roc_20), regime.momentum_threshold)

    def test_strategy_only_opens_position_when_regime_is_active(self):
        wallet = PaperWallet()
        strategy = StrategyAgent(wallet=wallet)

        calm_prices = [100.0 + (0.001 * i) for i in range(140)]
        for price in calm_prices:
            strategy.on_price(price)

        self.assertEqual(len(wallet.trades), 0)


if __name__ == "__main__":
    unittest.main()
