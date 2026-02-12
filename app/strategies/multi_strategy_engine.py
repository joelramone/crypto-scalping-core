from __future__ import annotations


class MultiStrategyEngine:
    """Engine that routes signal generation to a strategy based on market regime."""

    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRENDING = "TRENDING"

    def __init__(self, regime_detector, rsi_strategy, breakout_strategy) -> None:
        self.regime_detector = regime_detector
        self.rsi_strategy = rsi_strategy
        self.breakout_strategy = breakout_strategy

    def generate_signal(self, market_data):
        close_prices = market_data.get("close")
        if not close_prices:
            return None

        regime = self._detect_regime(close_prices)

        if regime == self.SIDEWAYS:
            return self.rsi_strategy.generate_signal(market_data)

        if regime == self.HIGH_VOLATILITY:
            return self.breakout_strategy.generate_signal(market_data)

        return None

    def _detect_regime(self, close_prices: list[float]) -> str | None:
        regime_state = self.regime_detector.evaluate(close_prices)
        if regime_state is None:
            return None

        if isinstance(regime_state, str):
            return regime_state

        regime_name = getattr(regime_state, "regime", None)
        if isinstance(regime_name, str):
            return regime_name

        if getattr(regime_state, "sideways", False):
            return self.SIDEWAYS

        if getattr(regime_state, "high_vol_expansion", False):
            return self.HIGH_VOLATILITY

        return None
