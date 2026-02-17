import unittest

from app.agents.volatility_regime_detector import (
    RegimeClassifier,
    RegimeScoreCalculator,
    VolatilityMetrics,
)


class VolatilityRegimeDetectorTests(unittest.TestCase):
    def test_detects_structural_volatility_expansion(self):
        metrics = VolatilityMetrics()
        score_calculator = RegimeScoreCalculator()
        classifier = RegimeClassifier(threshold=0.55, confirmation_bars=1)

        close = [100 + (0.01 * i) for i in range(220)]
        high = [c + 0.2 for c in close]
        low = [c - 0.2 for c in close]
        volume = [1000 + (i % 5) * 10 for i in range(220)]

        for i in range(180, 220):
            close[i] = close[i - 1] + (1.4 if i % 2 == 0 else -1.1)
            high[i] = close[i] + 1.8
            low[i] = close[i] - 1.8
            volume[i] = 2400

        snapshot = metrics.compute(close=close, high=high, low=low, volume=volume)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None

        score = score_calculator.compute(snapshot)
        regime = classifier.classify(score)

        self.assertEqual(regime, RegimeClassifier.HIGH_VOL)
        self.assertGreaterEqual(score.value, 0.55)

    def test_rejects_fake_breakout_without_volume_confirmation(self):
        metrics = VolatilityMetrics()
        score_calculator = RegimeScoreCalculator()
        classifier = RegimeClassifier(threshold=0.60, confirmation_bars=1)

        close = [100 + (0.01 * i) for i in range(220)]
        high = [c + 0.2 for c in close]
        low = [c - 0.2 for c in close]
        volume = [1500 for _ in close]

        for i in range(200, 220):
            close[i] = close[i - 1] + (1.2 if i % 2 == 0 else -1.0)
            high[i] = close[i] + 1.4
            low[i] = close[i] - 1.4
            volume[i] = 900

        snapshot = metrics.compute(close=close, high=high, low=low, volume=volume)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None

        score = score_calculator.compute(snapshot)
        regime = classifier.classify(score)

        self.assertEqual(regime, RegimeClassifier.NORMAL)
        self.assertLess(snapshot.volume_expansion_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
