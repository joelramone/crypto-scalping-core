from app.strategies.multi_strategy_engine import MultiStrategyEngine


class StubStrategy:
    def __init__(self):
        self.calls = 0
        self.trade_log = []

    def generate_signal(self, _market_data):
        self.calls += 1
        return {"side": "LONG"}

    def consume_last_signal_context(self):
        return {"strategy_name": "BreakoutTrendStrategy", "regime_detected": "HIGH_VOLATILITY"}

    def record_trade_outcome(self, payload):
        self.trade_log.append(payload)


def test_runs_in_single_strategy_mode():
    strategy = StubStrategy()
    engine = MultiStrategyEngine(strategy=strategy)

    signal = engine.generate_signal({"close": [100.0, 101.0], "atr": [1.0]})

    assert signal == {"side": "LONG"}
    assert strategy.calls == 1


def test_exposes_context_and_trade_log_from_wrapped_strategy():
    strategy = StubStrategy()
    engine = MultiStrategyEngine(strategy=strategy)

    context = engine.consume_last_signal_context()
    assert context == {"strategy_name": "BreakoutTrendStrategy", "regime_detected": "HIGH_VOLATILITY"}

    engine.record_trade_outcome({"r_multiple": 1.8})
    assert engine.trade_log == [{"r_multiple": 1.8}]
