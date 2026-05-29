from __future__ import annotations

from app.agents.regime_agent import RegimeAgent
from app.utils.logger import get_logger


class MultiStrategyEngine:
    HIGH_VOLATILITY = RegimeAgent.HIGH_VOLATILITY

    def __init__(self, breakout_strategy, mean_reversion_strategy, regime_agent: RegimeAgent | None = None, enable_shorts: bool = False) -> None:
        self.breakout_strategy = breakout_strategy
        self.mean_reversion_strategy = mean_reversion_strategy
        self.regime_agent = regime_agent or RegimeAgent()
        self.enable_shorts = enable_shorts
        self._last_signal_context: dict[str, float | str] | None = None
        self._trade_log: list[dict[str, float | str]] = []
        self._logger = get_logger(__name__)

    def generate_signal(self, market_data):
        regime = self.regime_agent.classify(market_data)
        if regime is None:
            self._last_signal_context = None
            return None

        atr_values = market_data.get("atr") or []
        if len(atr_values) >= 20:
            current_atr = float(atr_values[-1])
            avg_atr = sum(float(v) for v in atr_values[-20:]) / 20
            if avg_atr > 0 and (current_atr / avg_atr) <= 0.55:
                self._last_signal_context = {
                    "strategy_name": "NO_TRADE",
                    "regime_detected": regime,
                    "no_trade": "EXTREMELY_LOW_VOLATILITY",
                }
                return None

        selected_strategy = None
        if regime == RegimeAgent.TRENDING:
            selected_strategy = self.breakout_strategy
        elif regime in {RegimeAgent.RANGING, RegimeAgent.LOW_ACTIVITY, RegimeAgent.HIGH_VOLATILITY}:
            selected_strategy = self.mean_reversion_strategy

        if selected_strategy is None:
            self._last_signal_context = None
            return None

        signal = selected_strategy.generate_signal(market_data, regime=regime)
        if signal is None:
            self._last_signal_context = {
                "strategy_name": getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
                "regime_detected": regime,
            }
            return None

        if signal.get("side") == "SHORT" and not self.enable_shorts:
            self._last_signal_context = {
                "strategy_name": getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
                "regime_detected": regime,
                "no_trade": "SHORTS_DISABLED",
            }
            return None

        self._last_signal_context = {
            "strategy_name": getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
            "regime_detected": regime,
            "signal_quality_score": float(signal.get("signal_quality_score", 0.0)),
            "score_components": str(signal.get("score_components", {})),
        }
        self._logger.info(
            "signal_quality_evaluated",
            extra={
                "event_name": "signal_quality_evaluated",
                "parameters": {
                    "strategy_name": self._last_signal_context["strategy_name"],
                    "regime": regime,
                    "total_score": signal.get("signal_quality_score", 0),
                    "score_components": signal.get("score_components", {}),
                },
            },
        )
        return signal

    def consume_last_signal_context(self):
        context = self._last_signal_context
        self._last_signal_context = None
        return context

    def record_trade_outcome(self, payload):
        self._trade_log.append(payload)

    @property
    def trade_log(self):
        return list(self._trade_log)
