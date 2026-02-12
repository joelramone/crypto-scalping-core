from __future__ import annotations

from app.utils.logger import get_logger


class MultiStrategyEngine:
    """Engine that routes signal generation to a strategy based on market regime."""

    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRENDING = "TRENDING"

    def __init__(self, regime_detector, rsi_strategy, breakout_strategy) -> None:
        self.regime_detector = regime_detector
        self.rsi_strategy = rsi_strategy
        self.breakout_strategy = breakout_strategy
        self._last_signal_context: dict[str, str] | None = None
        self._trade_log: list[dict[str, str | float]] = []
        self._logger = get_logger(__name__)
        self._calls = 0

    def generate_signal(self, market_data):
        self._calls += 1
        regime = self.regime_detector.detect(market_data)

        if self._calls == 1 or self._calls % 100 == 0:
            self._logger.debug("Signal routing checkpoint call=%s regime=%s", self._calls, regime)

        if regime == self.SIDEWAYS:
            return self._delegate(self.rsi_strategy, regime, market_data)

        if regime == self.HIGH_VOLATILITY:
            return self._delegate(self.breakout_strategy, regime, market_data)

        return None

    def consume_last_signal_context(self) -> dict[str, str] | None:
        context = self._last_signal_context
        self._last_signal_context = None
        return context

    def record_trade_outcome(self, trade_result: str, pnl: float, context: dict[str, str] | None = None) -> None:
        trade_context = context or {}
        self._trade_log.append(
            {
                "strategy_name": trade_context.get("strategy_name", "unknown"),
                "regime_detected": trade_context.get("regime_detected", "unknown"),
                "trade_result": trade_result,
                "pnl": pnl,
            }
        )

    @property
    def trade_log(self) -> list[dict[str, str | float]]:
        return list(self._trade_log)

    def _delegate(self, strategy, regime: str, market_data):
        signal = strategy.generate_signal(market_data)
        if signal is None:
            return None

        self._last_signal_context = {
            "strategy_name": self._strategy_name(strategy),
            "regime_detected": regime,
        }
        return signal

    @staticmethod
    def _strategy_name(strategy) -> str:
        strategy_name = getattr(strategy, "name", None)
        if isinstance(strategy_name, str) and strategy_name:
            return strategy_name
        return strategy.__class__.__name__
