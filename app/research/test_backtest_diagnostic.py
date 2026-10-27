"""Synthetic timing and JSON diagnostics; never reads historical datasets."""
import argparse
import hashlib
import json
import math
from unittest.mock import Mock

import pandas as pd
import pytest

from app.research import backtester
from app.research.features import compute_features
from app.research.simulation import simulate_strategy
from app.research.strategies.baseline_trend import BaselineTrendStrategy
from app.research.test_simulation import SyntheticStrategy, candles


def frame(*rows):
    df = candles(*rows)
    df["open"] = df["close"]
    return df


@pytest.mark.parametrize("side", ["long", "short"])
def test_next_open_ignores_entry_bar_information_for_decision(side):
    df = frame((100,100,100),(120,130,110),(120,121,119))
    strategy = SyntheticStrategy(side=side)
    first = simulate_strategy(df,strategy,execution_mode="next_bar_open").trades[0]
    assert first.entry_index==1 and first.entry_price==120
    # Signal candle extremes cannot trigger an exit for the following entry.
    df.loc[0,["high","low"]]=[10000,1]
    # Following close/high/low affect exits, never the decision/open-price entry.
    df.loc[1,["close","high","low"]]=[5,10000,1]
    second = simulate_strategy(df,strategy,execution_mode="next_bar_open").trades[0]
    assert second.entry_index==first.entry_index and second.entry_price==first.entry_price
    assert second.exit_reason=="stop_loss"
    assert second.exit_index==1


def test_legacy_default_preserved():
    df=frame((100,100,100),(120,130,110),(120,121,119))
    assert simulate_strategy(df,SyntheticStrategy()).model_dump()==simulate_strategy(df,SyntheticStrategy(),execution_mode="signal_close").model_dump()
    assert simulate_strategy(df,SyntheticStrategy()).trades[0].entry_index==0
    assert simulate_strategy(df,SyntheticStrategy()).trades[0].entry_price==100


def test_no_following_candle_no_entry():
    assert not simulate_strategy(frame((100,100,100)),SyntheticStrategy(),execution_mode="next_bar_open").trades
    class LastOnly(SyntheticStrategy):
        def generate_entries(self,df):return pd.Series([False,True],index=df.index)
    assert not simulate_strategy(frame((100,100,100),(100,100,100)),LastOnly(),execution_mode="next_bar_open").trades


def test_holding_limit_includes_entry_bar():
    df=frame(*[(100,101,99)]*6)
    result=simulate_strategy(df,SyntheticStrategy(holding=2),execution_mode="next_bar_open")
    assert (result.trades[0].entry_index,result.trades[0].exit_index)==(1,2)
    assert result.trades[0].exit_reason=="max_holding"


def test_invalid_mode():
    with pytest.raises(ValueError):simulate_strategy(frame((100,100,100)),SyntheticStrategy(),execution_mode="wrong")


def test_json_metrics_reconcile_with_simulator():
    df=frame((100,100,100),(100,111,94),(100,100,100))
    result=simulate_strategy(df,SyntheticStrategy(),execution_mode="next_bar_open")
    report=backtester.diagnostic_report(result,BaselineTrendStrategy(),"hash",{"revision":"revision"},0.1,"next_bar_open",df,df)
    metrics=report["metrics"]
    assert metrics["trades"]==len(result.trades)==1
    assert metrics["gross_pnl"]==sum(t.gross_pnl for t in result.trades)
    assert metrics["total_fees"]==sum(t.fees for t in result.trades)>0
    assert metrics["net_pnl"]==pytest.approx(metrics["gross_pnl"]-metrics["total_fees"])
    assert metrics["profit_factor"]==result.metrics.profit_factor
    assert metrics["maximum_drawdown_usdt"]==result.metrics.max_drawdown
    json.dumps(report,allow_nan=False)


def test_unbounded_profit_factor_is_valid_json():
    df=frame((100,100,100),(100,111,99))
    result=simulate_strategy(df,SyntheticStrategy(),execution_mode="next_bar_open")
    assert math.isinf(result.metrics.profit_factor)
    report=backtester.diagnostic_report(result,BaselineTrendStrategy(),"hash",{},0,"next_bar_open",df,df)
    assert report["metrics"]["profit_factor"] is None
    assert report["metrics"]["profit_factor_status"]=="positive_profit_without_losses"
    json.dumps(report,allow_nan=False)


def synthetic_source(tmp_path):
    n=450
    close=pd.Series([100+math.sin(i/5) for i in range(n)])
    df=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=n,freq="1min",tz="UTC"),
                     "open":close,"high":close+1,"low":close-1,"close":close,"volume":1.})
    path=tmp_path/"synthetic.csv";df.to_csv(path,index=False)
    return path


def args(path,**changes):
    return argparse.Namespace(**(dict(data=str(path),strategy="baseline_trend",timeframe="1m",
        execution_mode="next_bar_open",json=True,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())|changes))


def test_hash_rejected_before_strategy(tmp_path,monkeypatch):
    path=synthetic_source(tmp_path)
    strategy=Mock(side_effect=AssertionError("Strategy must not load"))
    monkeypatch.setattr(backtester,"load_strategy",strategy)
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        backtester.run_backtest(args(path,expected_sha256="wrong"))
    strategy.assert_not_called()


def test_zero_trades_cli_and_no_side_effects(tmp_path,monkeypatch,capsys):
    path=synthetic_source(tmp_path);before=path.read_bytes()
    # No optimizer/memory import is needed to run this diagnostic.
    import builtins
    original=builtins.__import__
    def guard(name,*a,**kw):
        if name.startswith(("app.research.optimizer","app.research.memory","app.execution")):raise AssertionError(name)
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,"__import__",guard)
    backtester.run_backtest(args(path))
    report=json.loads(capsys.readouterr().out)
    assert report["metrics"]["trades"]==0
    assert report["metrics"]["net_pnl"]==report["metrics"]["total_fees"]==0
    assert report["metrics"]["profit_factor"]==0
    assert report["dataset"]["sha256"]==hashlib.sha256(before).hexdigest()
    assert report["code"]["revision"]
    assert report["execution_mode"]=="next_bar_open"
    assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]


def test_baseline_features_and_signals_prefix_causal(tmp_path):
    path=synthetic_source(tmp_path)
    raw=backtester.load_ohlcv_csv(path)
    full=compute_features(raw)
    prefix=compute_features(raw.iloc[:350])
    pd.testing.assert_frame_equal(full.iloc[:350],prefix)
    strategy=BaselineTrendStrategy()
    pd.testing.assert_series_equal(strategy.generate_entries(full).iloc[:350],strategy.generate_entries(prefix))


def test_json_configuration_guard(tmp_path):
    path=synthetic_source(tmp_path)
    for changes in (dict(strategy="mean_reversion"),dict(timeframe="5m"),dict(expected_sha256=None)):
        with pytest.raises(ValueError):backtester.run_backtest(args(path,**changes))


def test_penultimate_signal_can_enter_last_candle():
    df=frame((100,100,100),(100,101,99))
    result=simulate_strategy(df,SyntheticStrategy(),execution_mode="next_bar_open")
    assert len(result.trades)==1
    assert result.trades[0].entry_index==result.trades[0].exit_index==1


def test_main_json_error_and_cli_mode(tmp_path,monkeypatch,capsys):
    path=synthetic_source(tmp_path)
    monkeypatch.setattr("sys.argv",["backtester","--data",str(path),"--strategy","baseline_trend",
        "--timeframe","1m","--execution-mode","next_bar_open","--json","--expected-sha256","wrong"])
    assert backtester.parse_args().execution_mode=="next_bar_open"
    with pytest.raises(SystemExit) as error:backtester.main()
    assert error.value.code==2
    assert json.loads(capsys.readouterr().out)["status"]=="DIAGNOSTIC_ERROR"
