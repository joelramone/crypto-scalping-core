import numpy as np
import pandas as pd
import pytest

from app.research import aggressor_flow_report as report
from app.research.simulation import simulate_strategy
from app.research.strategies import aggressor_flow_persistence as subject


def candles_from_imbalance(values, price=100.0):
    n = len(values)
    volume = np.full(n, 10.0)
    taker = (np.asarray(values, dtype=float) + 1.0) / 2.0 * volume
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC"),
        "open": price, "high": price, "low": price, "close": price, "volume": volume, "taker_buy": taker,
    })


def test_resample_is_causal_and_drops_partial_candles():
    ts = pd.date_range("2023-01-01", periods=12, freq="1min", tz="UTC")
    one = pd.DataFrame({"timestamp": ts, "open": range(12), "high": range(12), "low": range(12),
                        "close": range(12), "volume": 1.0, "taker_buy_base_volume": 0.5})
    out = subject.resample_to_5m(one)
    assert len(out) == 2  # last group has only 2 minutes
    assert out["open"].tolist() == [0, 5] and out["close"].tolist() == [4, 9]
    assert out["volume"].tolist() == [5.0, 5.0] and out["taker_buy"].tolist() == [2.5, 2.5]


def test_imbalance_formula_and_zero_volume():
    c = candles_from_imbalance([0.2, -0.5])
    c.loc[1, "volume"] = 0.0
    imb = subject.imbalance(c)
    assert imb.iat[0] == pytest.approx(0.2) and np.isnan(imb.iat[1])


def test_onset_only_and_threshold_inclusive():
    vals = [0.0, 0.1, 0.1, 0.1, 0.3, 0.3, 0.0, 0.2, 0.2, 0.2]
    c = candles_from_imbalance(vals)
    long_sig = subject.onset_signals(c, "long")
    assert long_sig[long_sig].index.tolist() == [3, 9]
    assert not subject.onset_signals(c, "short").any()


def test_short_mirror_and_warmup():
    c = candles_from_imbalance([-0.5, -0.5, -0.5, -0.5])
    sig = subject.onset_signals(c, "short")
    assert sig[sig].index.tolist() == [2]


def test_signal_does_not_use_future_candles():
    vals = [0.5, 0.5, 0.5, -0.9, 0.9, -0.9]
    base = subject.onset_signals(candles_from_imbalance(vals), "long")
    for cut in range(1, len(vals)):
        truncated = subject.onset_signals(candles_from_imbalance(vals[:cut]), "long")
        assert truncated.tolist() == base.iloc[:cut].tolist()


def test_next_open_entry_and_short_take_profit():
    c = candles_from_imbalance([-0.5, -0.5, -0.5, 0.0, 0.0])
    c.loc[3, "open"] = 100.0
    c.loc[4, "low"] = 99.0
    result = simulate_strategy(c, subject.AggressorFlowPersistenceStrategy("short"), execution_mode="next_bar_open")
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_index == 3 and trade.side == "short" and trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(100.0 * (1 - subject.TAKE_PROFIT_PCT))


def leg(verdict, trades=150, gross=0.01):
    return {"verdict": verdict, "diagnostics": {"completed_trades": trades, "gross_expectancy": gross}}


def test_hypothesis_verdict_requires_mirror_gross_edge():
    assert report.hypothesis_verdict({"long": leg("BASELINE_CANDIDATE"), "short": leg("BASELINE_REJECT", gross=-0.01)}) == "BASELINE_REJECT"
    assert report.hypothesis_verdict({"long": leg("BASELINE_CANDIDATE"), "short": leg("BASELINE_REJECT", gross=0.01)}) == "BASELINE_CANDIDATE"
    assert report.hypothesis_verdict({"long": leg("INSUFFICIENT_SAMPLE", 10), "short": leg("INSUFFICIENT_SAMPLE", 50)}) == "INSUFFICIENT_SAMPLE"


def test_leg_verdict_rules():
    good = {"completed_trades": 120, "net_profit_factor": 1.2, "net_expectancy": 0.01, "net_pnl": 1.0,
            "gross_expectancy": 0.05, "positive_pnl_concentration_top_2_months": 0.6}
    assert report.leg_verdict(good) == "BASELINE_CANDIDATE"
    assert report.leg_verdict(good | {"positive_pnl_concentration_top_2_months": 0.81}) == "BASELINE_REJECT"
    assert report.leg_verdict(good | {"completed_trades": 99}) == "INSUFFICIENT_SAMPLE"


def test_runner_refuses_wrong_hash(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("timestamp\n")
    with pytest.raises(SystemExit):
        report.run(bad)
