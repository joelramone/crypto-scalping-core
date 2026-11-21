import pandas as pd
import pytest

from app.research import funding_carry_economics as subject


def frame(rates, start="2022-01-01"):
    times = pd.date_range(start, periods=len(rates), freq="8h", tz="UTC")
    return pd.DataFrame({"funding_time": times, "funding_rate": rates})


def test_round_trip_cost_counts_four_fills():
    assert subject.round_trip_cost({"spot": 0.001, "perp": 0.0005}) == pytest.approx(0.003)


def test_capital_includes_spot_and_margin():
    assert subject.capital_per_notional(1) == 2.0
    assert subject.capital_per_notional(3) == pytest.approx(4 / 3)


def test_constant_rate_annualizes_and_nets_fees():
    rate = 0.0001  # 0.01% per 8h -> 10.95% per year
    result = subject.analyze(frame([rate] * (3 * 365 * 2)))
    year = result["by_year"]["2022"]
    assert year["annualized_funding"] == pytest.approx(0.1095)
    taker = year["net_if_held_whole_year"]["taker_all_fills"]
    assert taker["net_on_notional"] == pytest.approx(0.1095 - 0.003)
    assert taker["annualized_on_capital"]["1x"] == pytest.approx((0.1095 - 0.003) / 2)
    assert result["rolling_365d_funding_sum"]["fraction_below_zero"] == 0.0


def test_negative_streak_and_breakeven():
    assert subject.longest_negative_streak(pd.Series([1, -1, -1, 0, -1, -1, -1, 2])) == 3
    days = subject.breakeven_days(0.1095)
    assert days["taker_all_fills"] == pytest.approx(0.003 / (0.1095 / 365))
    assert subject.breakeven_days(-0.01)["taker_all_fills"] is None


def test_load_refuses_wrong_hash(tmp_path):
    bad = tmp_path / "f.csv"
    bad.write_text("funding_time,funding_rate\n")
    with pytest.raises(SystemExit):
        subject.load(bad)
