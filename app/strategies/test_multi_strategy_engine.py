from app.strategies.multi_strategy_engine import MultiStrategyEngine


class StubRegimeDetector:
    def __init__(self, regime):
        self.regime = regime

    def evaluate(self, _prices):
        return self.regime


class StubStrategy:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    def generate_signal(self, _market_data):
        self.calls += 1
        return {"strategy": self.name}


class RegimeStateStub:
    def __init__(self, sideways=False, high_vol_expansion=False):
        self.sideways = sideways
        self.high_vol_expansion = high_vol_expansion


def test_routes_to_rsi_strategy_in_sideways_regime():
    rsi = StubStrategy("rsi")
    breakout = StubStrategy("breakout")
    engine = MultiStrategyEngine(
        regime_detector=StubRegimeDetector(RegimeStateStub(sideways=True)),
        rsi_strategy=rsi,
        breakout_strategy=breakout,
    )

    signal = engine.generate_signal({"close": [100.0, 101.0], "atr": [1.0], "rsi": [45.0]})

    assert signal == {"strategy": "rsi"}
    assert rsi.calls == 1
    assert breakout.calls == 0


def test_routes_to_breakout_strategy_in_high_volatility_regime():
    rsi = StubStrategy("rsi")
    breakout = StubStrategy("breakout")
    engine = MultiStrategyEngine(
        regime_detector=StubRegimeDetector(RegimeStateStub(high_vol_expansion=True)),
        rsi_strategy=rsi,
        breakout_strategy=breakout,
    )

    signal = engine.generate_signal({"close": [100.0, 101.0], "atr": [1.0], "rsi": [45.0]})

    assert signal == {"strategy": "breakout"}
    assert rsi.calls == 0
    assert breakout.calls == 1


def test_returns_none_when_regime_is_not_tradeable():
    rsi = StubStrategy("rsi")
    breakout = StubStrategy("breakout")
    engine = MultiStrategyEngine(
        regime_detector=StubRegimeDetector("TRENDING"),
        rsi_strategy=rsi,
        breakout_strategy=breakout,
    )

    signal = engine.generate_signal({"close": [100.0, 101.0], "atr": [1.0], "rsi": [45.0]})

    assert signal is None
    assert rsi.calls == 0
    assert breakout.calls == 0


def test_exposes_signal_context_and_records_trade_outcome():
    rsi = StubStrategy("rsi")
    breakout = StubStrategy("breakout")
    engine = MultiStrategyEngine(
        regime_detector=StubRegimeDetector(RegimeStateStub(sideways=True)),
        rsi_strategy=rsi,
        breakout_strategy=breakout,
    )

    engine.generate_signal({"close": [100.0, 101.0], "atr": [1.0], "rsi": [45.0]})
    context = engine.consume_last_signal_context()

    assert context == {"strategy_name": "rsi", "regime_detected": MultiStrategyEngine.SIDEWAYS}
    assert engine.consume_last_signal_context() is None

    engine.record_trade_outcome(trade_result="WIN", pnl=12.5, context=context)
    assert engine.trade_log == [
        {
            "strategy_name": "rsi",
            "regime_detected": MultiStrategyEngine.SIDEWAYS,
            "trade_result": "WIN",
            "pnl": 12.5,
        }
    ]
