from dataclasses import dataclass

from app.strategies.breakout_trend import BreakoutTrendStrategy


@dataclass
class StubBreakoutConfig:
    lookback_period: int = 20
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    breakout_buffer_atr: float = 0.05
    min_take_profit_r: float = 1.5
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = 1.0
    trend_ema_period: int = 200
    breakout_body_ratio_min: float = 0.70
    atr_avg_period: int = 20


def _build_strategy() -> BreakoutTrendStrategy:
    return BreakoutTrendStrategy(config=StubBreakoutConfig())


def _base_market_data():
    close = [100.0 + (i * 0.1) for i in range(220)]
    close[-2] = 121.8
    close[-1] = 123.0
    return {
        "open": [c - 0.1 for c in close[:-1]] + [121.9],
        "high": [c + 0.2 for c in close[:-1]] + [123.1],
        "low": [c - 0.2 for c in close[:-1]] + [121.8],
        "close": close,
        "atr": [1.0 for _ in range(200)] + [1.1 for _ in range(19)] + [1.6],
    }


def test_generates_long_signal_with_all_filters_aligned():
    strategy = _build_strategy()
    signal = strategy.generate_signal(_base_market_data())

    assert signal is not None
    assert signal["side"] == "LONG"


def test_rejects_long_when_price_below_ema200():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["close"][-1] = 105.0
    market_data["high"][-1] = 105.2
    market_data["low"][-1] = 104.9
    market_data["open"][-1] = 104.95

    signal = strategy.generate_signal(market_data)

    assert signal is None


def test_rejects_weak_breakout_candle_with_large_wicks():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["open"][-1] = 122.6
    market_data["close"][-1] = 123.0
    market_data["high"][-1] = 124.5
    market_data["low"][-1] = 121.0

    signal = strategy.generate_signal(market_data)

    assert signal is None


def test_rejects_when_atr_not_above_rolling_average():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["atr"][-1] = 1.0

    signal = strategy.generate_signal(market_data)

    assert signal is None
