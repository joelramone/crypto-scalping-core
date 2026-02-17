from __future__ import annotations

from app.agents.volatility_regime_detector import (
    RegimeClassifier,
    RegimeScoreCalculator,
    VolatilityMetrics,
)


class HighVolEngine:
    """Single-regime engine: trade only during HIGH_VOLATILITY conditions."""

    def __init__(
        self,
        breakout_strategy,
        volatility_metrics: VolatilityMetrics | None = None,
        score_calculator: RegimeScoreCalculator | None = None,
        classifier: RegimeClassifier | None = None,
    ) -> None:
        self.breakout_strategy = breakout_strategy
        self.volatility_metrics = volatility_metrics or VolatilityMetrics()
        self.score_calculator = score_calculator or RegimeScoreCalculator()
        self.classifier = classifier or RegimeClassifier(threshold=0.50, min_active_features=2, confirmation_bars=1, min_volume_expansion_ratio=1.0)
        self._last_signal_context: dict[str, float | str] | None = None
        self._trade_log: list[dict[str, float | str]] = []

    def generate_signal(self, market_data: dict[str, list[float]]):
        close = market_data.get("close") or []
        high = market_data.get("high") or close
        low = market_data.get("low") or close
        volume = market_data.get("volume") or [1.0 for _ in close]

        snapshot = self.volatility_metrics.compute(close=close, high=high, low=low, volume=volume)
        if snapshot is None:
            return None

        score = self.score_calculator.compute(snapshot)
        regime = self.classifier.classify(score)

        if regime != RegimeClassifier.HIGH_VOL:
            return None

        signal = self.breakout_strategy.generate_signal(market_data)
        if signal is None:
            return None

        self._last_signal_context = {
            "strategy_name": getattr(self.breakout_strategy, "name", self.breakout_strategy.__class__.__name__),
            "regime_detected": regime,
            "regime_score": score.value,
            "atr": snapshot.atr,
        }
        return signal

    def consume_last_signal_context(self) -> dict[str, float | str] | None:
        context = self._last_signal_context
        self._last_signal_context = None
        return context

    def record_trade_outcome(self, payload: dict[str, float | str]) -> None:
        self._trade_log.append(payload)

    def reset(self) -> None:
        self._last_signal_context = None
        self._trade_log.clear()
        self.classifier.reset()

    @property
    def trade_log(self) -> list[dict[str, float | str]]:
        return list(self._trade_log)
