from app.trend_bot import notify


def event(action="HOLD", target=0.0, fill=None):
    return {"mode": "paper", "candle_date": "2026-12-01", "target_fraction": target, "action": action, "fill": fill,
            "price": 60000.0, "equity_usdt": 1000.0, "return_since_start": 0.0}


def test_no_new_event_means_no_notification():
    assert not notify.should_notify([event()], since_line=1)


def test_hold_with_same_target_is_silent_but_first_event_and_changes_notify():
    assert notify.should_notify([event()], since_line=0)
    assert not notify.should_notify([event(), event()], since_line=1)
    assert notify.should_notify([event(), event(target=1 / 3)], since_line=1)
    assert notify.should_notify([event(), event("BUY", 1 / 3)], since_line=1)


def test_render_includes_fill_and_target():
    fill = {"btc": 0.005, "price": 60000.0, "fee_usdt": 0.3}
    text = notify.render(event("BUY", 1 / 3, fill))
    assert "33% en BTC" in text and "Compra" in text and "0.005000 BTC" in text
