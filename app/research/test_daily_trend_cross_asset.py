import numpy as np
import pandas as pd
import pytest

from app.research import daily_trend as base
from app.research import daily_trend_cross_asset as subject


def test_load_uses_adjusted_prices_and_checks_hash(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "open": [10, 11], "close": [10, 11],
                  "adjclose": [5, 6], "adj_open": [5, 5.5]}).to_csv(path, index=False)
    import hashlib
    frame = subject.load("X", hashlib.sha256(path.read_bytes()).hexdigest(), path)
    assert frame["open"].tolist() == [5, 5.5] and frame["close"].tolist() == [5, 6]
    with pytest.raises(SystemExit):
        subject.load("X", "0" * 64, path)


def test_annualization_uses_trading_days():
    returns = pd.Series([0.001] * 252, index=pd.date_range("2020-01-01", periods=252))
    assert base.metrics(returns, 252)["cagr"] == pytest.approx(1.001 ** 252 - 1)
    assert base.metrics(returns)["cagr"] == pytest.approx(1.001 ** 365 - 1)


def test_pass_rules_and_verdict():
    block = lambda s, d: {"sharpe": s, "max_drawdown": d}
    good = {"full": {"strategy": block(0.8, -0.2), "buy_and_hold": block(0.5, -0.5)}}
    bad_dd = {"full": {"strategy": block(0.8, -0.4), "buy_and_hold": block(0.5, -0.5)}}
    assert subject.asset_passes(good)["passes"] and not subject.asset_passes(bad_dd)["passes"]
    p, f = {"passes": True}, {"passes": False}
    assert subject.verdict({"A": p, "B": p, "C": p, "D": f}) == "XASSET_SUPPORTS_TREND"
    assert subject.verdict({"A": p, "B": p, "C": f, "D": f}) == "XASSET_DOES_NOT_SUPPORT"
