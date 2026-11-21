import numpy as np
import pandas as pd
import pytest

from app.research import download_yahoo as subject


def fake_chart(dates, price=100.0):
    ts = [int(pd.Timestamp(f"{d} 14:30", tz="UTC").timestamp()) for d in dates]
    n = len(dates)
    p = [price] * n
    return {"meta": {"exchangeTimezoneName": "America/New_York"}, "timestamp": ts,
            "indicators": {"quote": [{"open": p, "high": [price + 1] * n, "low": [price - 1] * n, "close": p,
                                      "volume": [1] * n}],
                           "adjclose": [{"adjclose": [price / 2] * n}]}}


def business_days(start, end):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end, inclusive="left")]


@pytest.fixture
def scope(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subject.time, "sleep", lambda s: None)
    monkeypatch.setitem(subject.AUTHORIZED_SCOPES, "TEST", [("SPY", subject.Path("data/spy.csv"))])
    return "2020-01-01T00:00:00Z", "2022-01-01T00:00:00Z"


def test_publishes_adjusted_frame(tmp_path, scope):
    days = business_days("2020-01-02", "2022-01-01")
    manifests = subject.execute("TEST", lambda *a: fake_chart(days), *scope)
    frame = pd.read_csv(tmp_path / "data/spy.csv")
    assert manifests[0]["integrity_checks"]["row_count"] == len(days)
    assert np.allclose(frame["adj_open"], 50.0)


def test_gap_or_missing_prices_block_publication(tmp_path, scope):
    days = business_days("2020-01-02", "2022-01-01")
    gappy = days[:100] + days[110:]
    with pytest.raises(ValueError):
        subject.execute("TEST", lambda *a: fake_chart(gappy), *scope)
    chart = fake_chart(days)
    chart["indicators"]["quote"][0]["close"][5] = None
    with pytest.raises(ValueError):
        subject.execute("TEST", lambda *a: chart, *scope)
    assert not (tmp_path / "data/spy.csv").exists()


def test_refuses_overwrite_and_unknown_scope(tmp_path, scope):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/spy.csv").write_text("x")
    with pytest.raises(FileExistsError):
        subject.execute("TEST", lambda *a: {}, *scope)
    with pytest.raises(KeyError):
        subject.execute("TASK-999X")
    assert [t["ticker"] for t in subject.plan()["targets"]] == ["SPY", "EFA", "TLT", "GLD"]
