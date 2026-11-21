import numpy as np
import pandas as pd
import pytest

from app.research import daily_trend as subject


def frame_from_close(close):
    close = np.asarray(close, dtype=float)
    n = len(close)
    t0 = pd.Timestamp("2018-01-01", tz="UTC").value // 1_000_000
    return pd.DataFrame({"open_time_ms": t0 + np.arange(n) * 86_400_000,
                         "timestamp": pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC"),
                         "open": close, "high": close, "low": close, "close": close})


def test_positions_take_vote_values_and_need_warmup():
    pos = subject.positions(pd.Series(np.linspace(100, 300, 400)))
    assert pos.iloc[: subject.WARMUP].isna().all()
    assert set(pos.dropna().unique()) <= {0.0, 1 / 3, 2 / 3, 1.0}
    assert pos.iloc[-1] == 1.0


def test_positions_do_not_use_future_closes():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(rng.normal(0, 0.03, 500).cumsum()))
    full = subject.positions(close)
    for cut in range(subject.WARMUP + 1, 500, 37):
        assert subject.positions(close.iloc[:cut]).equals(full.iloc[:cut])


def test_uptrend_strategy_is_fully_invested_and_pays_one_entry_fee():
    close = 100 * 1.001 ** np.arange(400)
    strategy, buy_hold, held = subject.daily_returns(frame_from_close(close))
    assert (held == 1.0).all()
    assert strategy.iloc[0] == pytest.approx(0.001 - subject.FEE_RATE)
    assert np.allclose(strategy.iloc[1:], 0.001) and np.allclose(buy_hold.iloc[1:], 0.001)


def test_crash_is_suffered_before_the_signal_can_react():
    k = 300
    close = np.r_[100 * 1.001 ** np.arange(k), np.full(50, 50.0)]  # open == close; price halves at day k
    frame = frame_from_close(close)
    strategy, _, held = subject.daily_returns(frame)
    day = lambda i: frame["timestamp"].iat[i]
    # Holding open[k-1] -> open[k] was decided at close[k-2] (fully long): the drop is taken.
    assert held.loc[day(k - 1)] == 1.0
    assert strategy.loc[day(k - 1)] == pytest.approx(50.0 / close[k - 1] - 1.0)
    # The exit decided at close[k] only applies from open[k+1].
    assert held.loc[day(k)] == 1.0
    assert held.loc[day(k + 1)] == subject.positions(frame["close"]).iat[k] < 1.0


def test_metrics_drawdown_and_verdict_rules():
    m = subject.metrics(pd.Series([0.1, -0.5, 0.2], index=pd.date_range("2020-01-01", periods=3)))
    assert m["max_drawdown"] == pytest.approx(-0.5)
    good = {"cagr": 0.3, "sharpe": 1.2, "max_drawdown": -0.3}
    bh = {"cagr": 0.5, "sharpe": 0.9, "max_drawdown": -0.8}
    block = {"strategy": good, "buy_and_hold": bh}
    res = {"full": block, "first_half": block, "second_half": block}
    assert subject.verdict(res, res)[0] == "TREND_CANDIDATE"
    worse_dd = {"full": {"strategy": good | {"max_drawdown": -0.5}, "buy_and_hold": bh},
                "first_half": block, "second_half": block}
    assert subject.verdict(worse_dd, res)[0] == "TREND_REJECT"


def test_load_refuses_wrong_hash(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("open_time_ms\n")
    with pytest.raises(SystemExit):
        subject.load(bad, "0" * 64)
