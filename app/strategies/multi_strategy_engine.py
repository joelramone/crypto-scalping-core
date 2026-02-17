from __future__ import annotations


class MultiStrategyEngine:
    """Legacy compatibility wrapper now running in single-strategy mode."""

    HIGH_VOLATILITY = "HIGH_VOLATILITY"

    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def generate_signal(self, market_data):
        return self.strategy.generate_signal(market_data)

    def consume_last_signal_context(self):
        consume = getattr(self.strategy, "consume_last_signal_context", None)
        return consume() if callable(consume) else None

    def record_trade_outcome(self, payload):
        record = getattr(self.strategy, "record_trade_outcome", None)
        if callable(record):
            record(payload)

    @property
    def trade_log(self):
        return list(getattr(self.strategy, "trade_log", []))
