from dataclasses import dataclass

from app.strategies.breakout_trend import BreakoutTrendStrategy


@dataclass
class StubBreakoutConfig:
    lookback_period: int = 20
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    breakout_buffer_atr: float = 0.05
    pullback_tolerance_atr: float = 0.20
    confirmation_sequence_bars: int = 3
    min_take_profit_r: float = 1.5
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = 1.0
    trend_ema_period: int = 200
    breakout_body_ratio_min: float = 0.70
    atr_avg_period: int = 20
    volume_avg_period: int = 20
    volume_multiplier: float = 1.0


def _build_strategy() -> BreakoutTrendStrategy:
    return BreakoutTrendStrategy(config=StubBreakoutConfig())


def _base_market_data():
    close = [100.0 + (i * 0.1) for i in range(220)]
    close[-3] = 123.2
    close[-2] = 122.4
    close[-1] = 123.0

    return {
        "open": [c - 0.1 for c in close[:-3]] + [121.9, 122.7, 122.4],
        "high": [c + 0.2 for c in close[:-3]] + [123.4, 122.9, 123.05],
        "low": [c - 0.2 for c in close[:-3]] + [121.8, 122.2, 122.35],
        "close": close,
        "atr": [1.0 for _ in range(200)] + [1.1 for _ in range(19)] + [1.6],
        "volume": [100.0 for _ in range(200)] + [110.0 for _ in range(19)] + [180.0],
    }


def test_generates_long_signal_after_pullback_confirmation():
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
    market_data["open"][-1] = 105.1

    signal = strategy.generate_signal(market_data)

    assert signal is None


def test_rejects_when_pullback_breaks_below_breakout_zone():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["close"][-2] = 120.0
    market_data["open"][-2] = 121.0
    market_data["high"][-2] = 121.2
    market_data["low"][-2] = 119.8

    signal = strategy.generate_signal(market_data)

    assert signal is None


def test_rejects_when_volume_is_not_elevated():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["volume"][-1] = 90.0

    signal = strategy.generate_signal(market_data)

    assert signal is None
