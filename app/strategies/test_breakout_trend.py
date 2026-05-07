from dataclasses import dataclass

from app.strategies.breakout_trend import BreakoutTrendStrategy


@dataclass
class StubBreakoutConfig:
    rsi_period: int = 7
    rsi_oversold: float = 20.0
    rsi_overbought: float = 80.0
    ema_period: int = 20
    extension_atr_multiplier: float = 0.2
    volume_avg_period: int = 20
    volume_multiplier: float = 1.2
    exhaustion_wick_ratio_min: float = 0.45
    min_body_ratio: float = 0.2
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    min_take_profit_r: float = 1.5
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = 1.0
    atr_avg_period: int = 20
    min_atr_ratio: float = 1.0


def _build_strategy() -> BreakoutTrendStrategy:
    return BreakoutTrendStrategy(config=StubBreakoutConfig())


def _base_market_data():
    close = [100.0 + (i * 0.1) for i in range(60)]
    open_prices = [c - 0.05 for c in close]
    high = [c + 0.15 for c in close]
    low = [c - 0.15 for c in close]

    close[-2] = 103.8
    open_prices[-2] = 104.2
    high[-2] = 104.3
    low[-2] = 103.2

    close[-1] = 104.1
    open_prices[-1] = 103.9
    high[-1] = 104.2
    low[-1] = 103.8

    return {
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "atr": [1.0 for _ in range(39)] + [1.2 for _ in range(21)],
        "volume": [100.0 for _ in range(40)] + [115.0 for _ in range(19)] + [160.0],
        "rsi_7": [50.0 for _ in range(58)] + [18.0, 25.0],
    }


def test_generates_long_signal_on_mean_reversion_setup():
    strategy = _build_strategy()
    signal = strategy.generate_signal(_base_market_data())

    assert signal is not None
    assert signal["side"] == "LONG"


def test_generates_short_signal_on_mean_reversion_setup():
    strategy = _build_strategy()
    market_data = _base_market_data()

    market_data["close"][-2] = 108.8
    market_data["open"][-2] = 108.2
    market_data["high"][-2] = 109.4
    market_data["low"][-2] = 108.1
    market_data["close"][-1] = 108.3
    market_data["open"][-1] = 108.6
    market_data["high"][-1] = 108.7
    market_data["low"][-1] = 108.2
    market_data["rsi_7"][-2] = 84.0

    signal = strategy.generate_signal(market_data)

    assert signal is not None
    assert signal["side"] == "SHORT"


def test_rejects_when_volume_is_not_elevated():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["volume"][-1] = 100.0

    signal = strategy.generate_signal(market_data)

    assert signal is None


def test_rejects_when_confirmation_candle_missing():
    strategy = _build_strategy()
    market_data = _base_market_data()
    market_data["open"][-1] = 104.2

    signal = strategy.generate_signal(market_data)

    assert signal is None
