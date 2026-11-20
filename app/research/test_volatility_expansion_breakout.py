import numpy as np
import pandas as pd
import pytest

from app.research import volatility_expansion_report as report
from app.research.simulation import simulate_strategy
from app.research.strategies import volatility_expansion_breakout as subject


def flat_candles(n, price=100.0, half_range=0.1):
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-04-15", periods=n, freq="15min", tz="UTC"),
        "open": price, "high": price + half_range, "low": price - half_range, "close": price, "volume": 1.0,
    })


def with_bar(c, t, o, h, l, cl):
    c.loc[t, ["open", "high", "low", "close"]] = [o, h, l, cl]
    return c


def test_resample_15m_drops_partial_candles():
    ts = pd.date_range("2023-04-15", periods=40, freq="1min", tz="UTC")
    one = pd.DataFrame({"timestamp": ts, "open": range(40), "high": range(40), "low": range(40),
                        "close": range(40), "volume": 1.0})
    out = subject.resample_to_15m(one)
    assert len(out) == 2 and out["open"].tolist() == [0, 15] and out["close"].tolist() == [14, 29]


def test_atr_excludes_current_candle_and_needs_full_lookback():
    warmup = subject.ATR_LOOKBACK + 1  # TR[0] is undefined (no previous close): 97 warm-up candles
    c = flat_candles(warmup + 2)
    atr = subject.atr_prev(c)
    assert atr.iloc[:warmup].isna().all()
    assert atr.iat[warmup] == pytest.approx(0.2)
    with_bar(c, warmup, 100.0, 105.0, 99.9, 104.9)
    assert subject.atr_prev(c).iat[warmup] == pytest.approx(0.2)


def test_long_and_short_signal_rules():
    t = subject.ATR_LOOKBACK + 1
    c = with_bar(flat_candles(t + 5), t, 100.0, 100.5, 99.95, 100.45)  # TR 0.55 >= 2*0.2, loc 0.91
    assert subject.signals(c, "long")[subject.signals(c, "long")].index.tolist() == [t]
    assert not subject.signals(c, "short").any()
    c = with_bar(flat_candles(t + 5), t, 100.0, 100.05, 99.5, 99.55)
    assert subject.signals(c, "short")[subject.signals(c, "short")].index.tolist() == [t]


def test_no_signal_when_close_not_at_extreme_or_range_too_small():
    t = subject.ATR_LOOKBACK + 1
    mid = with_bar(flat_candles(t + 2), t, 100.0, 100.5, 99.5, 100.1)  # loc 0.6
    small = with_bar(flat_candles(t + 2), t, 100.0, 100.15, 99.95, 100.14)  # TR 0.2 < 0.4
    for c in (mid, small):
        assert not subject.signals(c, "long").any() and not subject.signals(c, "short").any()


def test_signals_do_not_use_future_candles():
    rng = np.random.default_rng(0)
    n = subject.ATR_LOOKBACK + 60
    close = 100 + rng.normal(0, 0.3, n).cumsum()
    open_ = np.r_[100.0, close[:-1]]
    spread = rng.uniform(0.05, 1.5, n)
    c = pd.DataFrame({"timestamp": pd.date_range("2023-04-15", periods=n, freq="15min", tz="UTC"),
                      "open": open_, "close": close, "high": np.maximum(open_, close) + spread * 0.1,
                      "low": np.minimum(open_, close) - spread * 0.1, "volume": 1.0})
    for side in ("long", "short"):
        full = subject.signals(c, side)
        for cut in range(subject.ATR_LOOKBACK, n, 7):
            assert subject.signals(c.iloc[:cut], side).tolist() == full.iloc[:cut].tolist()


def test_next_open_entry_long_take_profit():
    t = subject.ATR_LOOKBACK + 1
    c = with_bar(flat_candles(t + 5), t, 100.0, 100.5, 99.95, 100.45)
    c = with_bar(c, t + 1, 100.5, 101.6, 100.4, 101.5)
    result = simulate_strategy(c, subject.VolatilityExpansionBreakoutStrategy("long"), execution_mode="next_bar_open")
    trade = result.trades[0]
    assert trade.entry_index == t + 1 and trade.entry_price == 100.5 and trade.exit_reason == "take_profit"


def test_runner_refuses_wrong_hash(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("timestamp\n")
    with pytest.raises(SystemExit):
        report.run(bad)
