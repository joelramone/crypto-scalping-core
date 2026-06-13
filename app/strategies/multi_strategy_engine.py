from __future__ import annotations

from app.agents.regime_agent import RegimeAgent
from app.utils.logger import get_logger
from app.utils.signal_rejections import SignalRejectionEvent, SignalRejectionTracker, build_rejection_event, safe_ratio


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
        self._rejection_tracker = SignalRejectionTracker()

    def generate_signal(self, market_data):
        self._emit_hourly_rejection_summary()
        regime = self.regime_agent.classify(market_data)
        if regime is None:
            self._last_signal_context = None
            self._log_rejection(
                build_rejection_event(
                    strategy="NO_STRATEGY",
                    regime=None,
                    reasons=["regime_unclassified"],
                    **self._market_observability(market_data),
                )
            )
            return None

        atr_values = market_data.get("atr") or []
        if len(atr_values) >= 20:
            current_atr = float(atr_values[-1])
            avg_atr = sum(float(v) for v in atr_values[-20:]) / 20
            atr_ratio = safe_ratio(current_atr, avg_atr)
            if avg_atr > 0 and atr_ratio is not None and atr_ratio <= 0.55:
                self._last_signal_context = {
                    "strategy_name": "NO_TRADE",
                    "regime_detected": regime,
                    "no_trade": "EXTREMELY_LOW_VOLATILITY",
                }
                self._log_rejection(
                    build_rejection_event(
                        strategy="NO_TRADE",
                        regime=regime,
                        reasons=["atr_too_low"],
                        atr=current_atr,
                        atr_status="low",
                        volume_ratio=self._volume_ratio(market_data),
                        ema_distance=self._ema_distance(market_data),
                    )
                )
                return None

        selected_strategy = None
        if regime == RegimeAgent.TRENDING:
            selected_strategy = self.breakout_strategy
        elif regime in {RegimeAgent.RANGING, RegimeAgent.LOW_ACTIVITY, RegimeAgent.HIGH_VOLATILITY}:
            selected_strategy = self.mean_reversion_strategy

        if selected_strategy is None:
            self._last_signal_context = None
            self._log_rejection(
                build_rejection_event(
                    strategy="NO_STRATEGY",
                    regime=regime,
                    reasons=["regime_not_supported"],
                    **self._market_observability(market_data),
                )
            )
            return None

        signal = selected_strategy.generate_signal(market_data, regime=regime)
        if signal is None:
            self._last_signal_context = {
                "strategy_name": getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
                "regime_detected": regime,
            }
            strategy_rejection = self._consume_strategy_rejection(selected_strategy)
            if strategy_rejection is not None:
                self._log_rejection(strategy_rejection)
            return None

        if signal.get("side") == "SHORT" and not self.enable_shorts:
            self._last_signal_context = {
                "strategy_name": getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
                "regime_detected": regime,
                "no_trade": "SHORTS_DISABLED",
            }
            self._log_rejection(
                build_rejection_event(
                    strategy=getattr(selected_strategy, "name", selected_strategy.__class__.__name__),
                    regime=regime,
                    reasons=["shorts_disabled"],
                    score=float(signal.get("signal_quality_score", 0.0)),
                    required_score=float(getattr(getattr(selected_strategy, "config", object()), "signal_score_threshold", 0.0)) or None,
                    **self._market_observability(market_data),
                )
            )
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

    @property
    def rejection_counters(self):
        return self._rejection_tracker.counters.model_copy()

    def _consume_strategy_rejection(self, selected_strategy) -> SignalRejectionEvent | None:
        consumer = getattr(selected_strategy, "consume_last_rejection_event", None)
        if callable(consumer):
            return consumer()
        return None

    def _log_rejection(self, event: SignalRejectionEvent) -> None:
        self._rejection_tracker.record(event)
        for line in SignalRejectionTracker.format_rejection_lines(event):
            self._logger.info(
                line,
                extra={
                    "event_name": "signal_rejected",
                    "parameters": event.model_dump(),
                },
            )

    def _emit_hourly_rejection_summary(self) -> None:
        for line in self._rejection_tracker.maybe_hourly_summary_lines():
            self._logger.info(
                line,
                extra={
                    "event_name": "signal_rejection_summary",
                    "parameters": self._rejection_tracker.counters.model_dump(),
                },
            )

    def _market_observability(self, market_data) -> dict[str, float | str | None]:
        return {
            "atr": self._current_atr(market_data),
            "atr_status": None,
            "volume_ratio": self._volume_ratio(market_data),
            "ema_distance": self._ema_distance(market_data),
        }

    @staticmethod
    def _current_atr(market_data) -> float | None:
        atr_values = market_data.get("atr") or []
        return float(atr_values[-1]) if atr_values else None

    @staticmethod
    def _volume_ratio(market_data) -> float | None:
        volume = market_data.get("volume") or []
        if len(volume) < 2:
            return None
        avg_volume = sum(float(value) for value in volume[:-1]) / len(volume[:-1])
        return safe_ratio(volume[-1], avg_volume)

    @staticmethod
    def _ema_distance(market_data) -> float | None:
        close = market_data.get("close") or []
        if len(close) < 20:
            return None
        ema_value = MultiStrategyEngine._ema(close, 20)
        if ema_value is None:
            return None
        return float(close[-1]) - ema_value

    @staticmethod
    def _ema(values, period: int) -> float | None:
        if len(values) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = sum(float(value) for value in values[:period]) / period
        for value in values[period:]:
            ema = ((float(value) - ema) * multiplier) + ema
        return ema
