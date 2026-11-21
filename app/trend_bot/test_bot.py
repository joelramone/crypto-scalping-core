import json

import numpy as np
import pandas as pd
import pytest

from app.trend_bot import bot

DAY = bot.DAY_MS
T0 = int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)


def fake_get(closes, price=None):
    def get(path, params):
        if path == "/api/v3/klines":
            return [[T0 + i * DAY, "1", "1", "1", str(c), "1", T0 + (i + 1) * DAY - 1] for i, c in enumerate(closes)]
        if path == "/api/v3/ticker/price":
            return {"price": str(price or closes[-1])}
        raise AssertionError(path)
    return get


def now_after(closes, extra_ms=60_000):
    return T0 + len(closes) * DAY + extra_ms


def config(tmp_path, **kw):
    return bot.Config(mode="paper", capital_usdt=1000.0, state_dir=tmp_path, kill_switch=tmp_path / "STOP", **kw)


UP = list(100 * 1.01 ** np.arange(260))
DOWN = list(100 * 0.99 ** np.arange(260))


def test_uptrend_buys_full_sleeve_with_fee(tmp_path):
    result = bot.run_once(config(tmp_path), now_ms=now_after(UP), get=fake_get(UP))
    assert result["action"] == "BUY" and result["target_fraction"] == 1.0
    assert result["sleeve"]["usdt"] == pytest.approx(0.0)
    assert result["sleeve"]["btc"] == pytest.approx(1000 * (1 - bot.FEE_RATE) / UP[-1])
    assert (tmp_path / "ledger.jsonl").read_text().count("\n") == 1


def test_idempotent_per_daily_candle_and_kill_switch(tmp_path):
    c = config(tmp_path)
    bot.run_once(c, now_ms=now_after(UP), get=fake_get(UP))
    assert bot.run_once(c, now_ms=now_after(UP, 3_600_000), get=fake_get(UP))["status"] == "ALREADY_RAN"
    (tmp_path / "STOP").write_text("")
    assert bot.run_once(c, now_ms=now_after(UP + [UP[-1]]), get=fake_get(UP + [UP[-1]]))["status"] == "STOPPED"


def test_unclosed_candle_is_ignored(tmp_path):
    closes = UP + [1.0]  # last candle still open at 'now'
    frame = bot.closed_daily_candles(T0 + (len(closes) - 1) * DAY + 3_600_000, fake_get(closes))
    assert frame["close"].iloc[-1] == UP[-1]


def test_downtrend_sells_everything_after_holding(tmp_path):
    c = config(tmp_path)
    bot.run_once(c, now_ms=now_after(UP), get=fake_get(UP))
    path = UP + list(UP[-1] * 0.97 ** np.arange(1, 120))
    result = None
    for n in range(len(UP) + 1, len(path) + 1):
        result = bot.run_once(c, now_ms=now_after(path[:n]), get=fake_get(path[:n]))
    state = json.loads((tmp_path / "state.json").read_text())
    assert result["target_fraction"] == 0.0 and state["sleeve"]["btc"] == 0.0
    assert state["sleeve"]["usdt"] > 0 and state["trades"] >= 2


def test_plan_trade_hold_drift_and_minimum():
    sleeve = {"usdt": 500.0, "btc": 5.0}
    assert bot.plan_trade(sleeve, 100.0, 0.5, 0.5) == ("HOLD", 0.0)
    side, amount = bot.plan_trade(sleeve, 100.0, 2 / 3, 1 / 3)
    assert side == "BUY" and amount == pytest.approx((2 / 3 - 0.5) * 1000)
    assert bot.plan_trade({"usdt": 5.0, "btc": 0.0}, 100.0, 1.0, 0.0) == ("HOLD", 0.0)
    assert bot.plan_trade(sleeve, 100.0, 0.0, 1 / 3) == ("SELL", 5.0)


def test_floor_step():
    assert bot.floor_step(0.123456789, "0.00001") == pytest.approx(0.12345)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, withdrawals=False, usdt=1000.0, btc=0.0):
        self.withdrawals, self.usdt, self.btc, self.orders = withdrawals, usdt, btc, []

    def request(self, method, url, headers=None, timeout=None):
        assert "signature=" in url and headers["X-MBX-APIKEY"] == "k"
        if "/sapi/v1/account/apiRestrictions" in url:
            return FakeResponse({"enableWithdrawals": self.withdrawals, "enableSpotAndMarginTrading": True})
        if "/api/v3/account" in url:
            return FakeResponse({"balances": [{"asset": "USDT", "free": str(self.usdt)}, {"asset": "BTC", "free": str(self.btc)}]})
        if "/api/v3/order" in url:
            self.orders.append(url)
            return FakeResponse({"orderId": 1, "executedQty": "0.00999000", "cummulativeQuoteQty": "999.00",
                                 "fills": [{"price": "100000", "commission": "0.999", "commissionAsset": "USDT"}]})
        raise AssertionError(url)

    def get(self, url, params=None, timeout=None):
        return FakeResponse({"symbols": [{"filters": [{"filterType": "LOT_SIZE", "stepSize": "0.00001000"}]}]})


def test_live_refuses_withdraw_enabled_key():
    broker = bot.LiveBroker("k", "s", FakeSession(withdrawals=True))
    with pytest.raises(SystemExit, match="withdraw"):
        broker.preflight({"usdt": 100.0, "btc": 0.0})


def test_live_refuses_when_account_does_not_cover_sleeve():
    broker = bot.LiveBroker("k", "s", FakeSession(usdt=50.0))
    with pytest.raises(SystemExit, match="do not cover"):
        broker.preflight({"usdt": 100.0, "btc": 0.0})


def test_live_buy_uses_quote_qty_and_idempotent_client_id(tmp_path):
    session = FakeSession()
    broker = bot.LiveBroker("k", "s", session)
    c = bot.Config(mode="live", capital_usdt=1000.0, state_dir=tmp_path, kill_switch=tmp_path / "STOP")
    result = bot.run_once(c, broker=broker, now_ms=now_after(UP), get=fake_get(UP, price=100000))
    assert result["action"] == "BUY" and len(session.orders) == 1
    assert "quoteOrderQty=1000.00" in session.orders[0] and "newClientOrderId=trend-" in session.orders[0]
    assert result["sleeve"]["btc"] == pytest.approx(0.00999) and result["sleeve"]["usdt"] == pytest.approx(1.0)


def test_rule_matches_research_implementation():
    closes = list(100 * np.exp(np.random.default_rng(5).normal(0, 0.03, 260).cumsum()))
    frame = bot.closed_daily_candles(now_after(closes), fake_get(closes))
    from app.research.daily_trend import positions
    assert bot.target_fraction(frame) == positions(pd.Series(closes)).iloc[-1]
