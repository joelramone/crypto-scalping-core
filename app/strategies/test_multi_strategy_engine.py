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


def test_blocks_short_from_mean_reversion_when_shorts_disabled():
    breakout = StubStrategy(side="LONG")
    mean_rev = StubStrategy(side="SHORT")
    engine = MultiStrategyEngine(breakout_strategy=breakout, mean_reversion_strategy=mean_rev, regime_agent=StubRegimeAgent(RegimeAgent.LOW_ACTIVITY))

    signal = engine.generate_signal(_market_data())

    assert signal is None
    assert breakout.calls == 0
    assert mean_rev.calls == 1


def test_allows_short_when_shorts_enabled():
    breakout = StubStrategy(side="LONG")
    mean_rev = StubStrategy(side="SHORT")
    engine = MultiStrategyEngine(
        breakout_strategy=breakout,
        mean_reversion_strategy=mean_rev,
        regime_agent=StubRegimeAgent(RegimeAgent.LOW_ACTIVITY),
        enable_shorts=True,
    )

    signal = engine.generate_signal(_market_data())

    assert signal == {"side": "SHORT"}

class RejectingStubStrategy:
    name = "RejectingStubStrategy"

    def __init__(self):
        self.calls = 0
        self._event = None

    def generate_signal(self, market_data, regime=None):
        from app.utils.signal_rejections import build_rejection_event

        self.calls += 1
        self._event = build_rejection_event(
            strategy=self.name,
            regime=regime,
            score=3.0,
            required_score=5.0,
            atr=float(market_data["atr"][-1]),
            atr_status="ok",
            volume_ratio=0.82,
            ema_distance=1.5,
            reasons=["score_too_low"],
        )
        return None

    def consume_last_rejection_event(self):
        event = self._event
        self._event = None
        return event


def test_tracks_rejection_counter_from_strategy_context():
    breakout = StubStrategy(side="LONG")
    mean_rev = RejectingStubStrategy()
    engine = MultiStrategyEngine(breakout_strategy=breakout, mean_reversion_strategy=mean_rev, regime_agent=StubRegimeAgent(RegimeAgent.LOW_ACTIVITY))

    signal = engine.generate_signal(_market_data())

    assert signal is None
    assert mean_rev.calls == 1
    assert engine.rejection_counters.rejected_by_score == 1
    assert engine.rejection_counters.rejected_by_volume == 0
