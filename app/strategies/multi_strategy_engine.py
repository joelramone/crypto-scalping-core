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
        self._entry_log_printed = False
        self._last_signal_context: dict[str, str] | None = None
        self._trade_log: list[dict[str, str | float]] = []

    def generate_signal(self, market_data):
        if not self._entry_log_printed:
            print("Entered MultiStrategyEngine.generate_signal()", flush=True)
            self._entry_log_printed = True
        close_prices = market_data.get("close")
        if not close_prices:
            return None

        regime = self._detect_regime(close_prices)

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
