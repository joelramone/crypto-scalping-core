from app.agents.regime_agent import RegimeAgent
from app.strategies.multi_strategy_engine import MultiStrategyEngine


class StubStrategy:
    def __init__(self, side: str):
        self.side = side
        self.calls = 0

    def generate_signal(self, _market_data, regime=None):
        _ = regime
        self.calls += 1
        return {"side": self.side}


class StubRegimeAgent:
    def __init__(self, regime: str):
        self.regime = regime

    def classify(self, _market_data):
        return self.regime


def _market_data():
    return {
        "close": [100.0 + i for i in range(30)],
        "high": [101.0 + i for i in range(30)],
        "low": [99.0 + i for i in range(30)],
        "atr": [1.0 for _ in range(30)],
    }


def test_routes_to_breakout_on_trending_regime():
    breakout = StubStrategy(side="LONG")
    mean_rev = StubStrategy(side="SHORT")
    engine = MultiStrategyEngine(breakout_strategy=breakout, mean_reversion_strategy=mean_rev, regime_agent=StubRegimeAgent(RegimeAgent.TRENDING))

    signal = engine.generate_signal(_market_data())

    assert signal == {"side": "LONG"}
    assert breakout.calls == 1
    assert mean_rev.calls == 0


def test_routes_to_mean_reversion_on_low_activity():
    breakout = StubStrategy(side="LONG")
    mean_rev = StubStrategy(side="SHORT")
    engine = MultiStrategyEngine(breakout_strategy=breakout, mean_reversion_strategy=mean_rev, regime_agent=StubRegimeAgent(RegimeAgent.LOW_ACTIVITY))

    signal = engine.generate_signal(_market_data())

    assert signal == {"side": "SHORT"}
    assert breakout.calls == 0
    assert mean_rev.calls == 1
