import json

import numpy as np
import pandas as pd
import pytest

from app.research import daily_trend_forward as subject
from app.research.download_daily import DAY_MS
from app.research.download_funding import to_ms


def make_fetch(price_fn):
    start = to_ms(subject.WARMUP_START)

    def fetch(session, symbol, s, e):
        rows = []
        t = max(s, start)
        while t <= e and len(rows) < 1000:
            i = (t - start) // DAY_MS
            p = price_fn(symbol, i)
            rows.append([t, str(p), str(p * 1.01), str(p * 0.99), str(p), "10", t + DAY_MS - 1, "1", 1, "0", "0", "0"])
            t += DAY_MS
        return rows
    return fetch


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.research.download_daily.REQUEST_SLEEP_SECONDS", 0)


def uptrend(symbol, i):
    return 100 * 1.002 ** i


def test_only_forward_days_are_evaluated_and_first_day_pays_entry(tmp_path):
    as_of = pd.Timestamp("2026-12-01", tz="UTC")
    status = subject.run(as_of, make_fetch(uptrend), tmp_path)
    days = status["daily"]["BTCUSDT"]
    assert days[0]["date"] == "2026-11-21" and days[-1]["date"] == "2026-11-29"
    assert status["forward_days"] == 9 and status["verdict"] == "FORWARD_IN_PROGRESS"
    assert days[0]["position"] == 1.0
    assert days[0]["strategy"] == pytest.approx(0.002 - subject.FEE_RATE)
    assert days[1]["strategy"] == pytest.approx(0.002)
    assert (tmp_path / "report.md").read_text().startswith("# Daily Trend BTC")


def test_revision_guard_blocks_rewriting_history(tmp_path):
    subject.run(pd.Timestamp("2026-12-01", tz="UTC"), make_fetch(uptrend), tmp_path)
    revised = make_fetch(lambda s, i: uptrend(s, i) * (1.05 if i == 270 else 1.0))
    with pytest.raises(ValueError, match="revised"):
        subject.run(pd.Timestamp("2026-12-10", tz="UTC"), revised, tmp_path)
    subject.run(pd.Timestamp("2026-12-10", tz="UTC"), make_fetch(uptrend), tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["forward_days"] == 18


def test_rule_matches_frozen_in_sample_implementation():
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(rng.normal(0, 0.03, n).cumsum())
    ts = pd.date_range("2026-03-01", periods=n, freq="D", tz="UTC")
    frame = pd.DataFrame({"timestamp": ts, "open": close, "close": close})
    strategy, _, held = subject.forward_returns(frame)
    expected = subject.positions(pd.Series(close)).shift(1)
    expected.index = ts
    assert held.equals(expected.loc[held.index])
    assert held.index[0] == subject.FORWARD_START


def test_verdict_rules():
    early = pd.Timestamp("2027-06-01", tz="UTC")
    late = pd.Timestamp("2028-11-22", tz="UTC")
    block = lambda s, d: {"sharpe": s, "max_drawdown": d}
    good = {"BTCUSDT": {"strategy": block(1.0, -0.3), "buy_and_hold": block(0.5, -0.5)},
            "ETHUSDT": {"strategy": block(0.8, -0.4), "buy_and_hold": block(0.4, -0.6)}}
    assert subject.verdict(early, good) == "FORWARD_IN_PROGRESS"
    assert subject.verdict(late, good) == "FORWARD_CONFIRMED"
    calm = {**good, "BTCUSDT": {"strategy": block(1.0, -0.05), "buy_and_hold": block(0.5, -0.1)}}
    assert subject.verdict(late, calm) == "FORWARD_INCONCLUSIVE"
    bad = {**good, "ETHUSDT": {"strategy": block(0.3, -0.4), "buy_and_hold": block(0.4, -0.6)}}
    assert subject.verdict(late, bad) == "FORWARD_REJECTED"
