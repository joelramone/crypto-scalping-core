"""Synthetic-only attribution reconciliation and one-run integration."""
import json
from unittest.mock import Mock

import pytest

from app.research import backtester
from app.research.analysis.trade_diagnostics import calculate_loss_attribution
from app.research.simulation import BacktestTrade, BacktestResult, calculate_metrics
from app.research.test_backtest_diagnostic import args, synthetic_source


def records():
    # Binary-exact amounts make grouped/aggregate equality exact in this fixture.
    return [BacktestTrade(entry_index=i*5, exit_index=i*5+holding,
        entry_timestamp=f"2024-01-01T00:{i*5:02}:00Z", exit_timestamp=f"2024-01-01T00:{i*5+holding:02}:00Z",
        entry_price=100,exit_price=100,notional=100,gross_pnl=gross,fees=fee,
        net_pnl=gross-fee,exit_reason=reason)
        for i,(reason,gross,fee,holding) in enumerate([
            ("take_profit",4.,.5,1),("stop_loss",-2.,.25,0),
            ("max_holding",.5,.5,3),("take_profit",.25,.5,2)])]


@pytest.mark.parametrize("trades", [[], records(), records()[:1], records()[1:2]])
def test_reconciles_with_existing_metrics(trades):
    attribution=calculate_loss_attribution(trades);metrics=calculate_metrics(trades)
    assert attribution["totals"] == dict(trades=metrics.total_trades,gross_pnl=metrics.gross_pnl,
                                        net_pnl=metrics.net_pnl,total_fees=metrics.estimated_fees)
    for key in ("trades","gross_pnl","net_pnl","total_fees"):
        assert sum(group[key] for group in attribution["by_exit_reason"].values())==attribution["totals"][key]
    assert attribution["wins"]+attribution["losses"]+attribution["flats"]==len(trades)
    assert attribution["total_fee_burden_usdt"]==metrics.estimated_fees
    assert attribution["net_pnl_per_trade"]==metrics.expectancy
    json.dumps(attribution,allow_nan=False)


def test_outcomes_holdings_and_fee_dominated_tp():
    a=calculate_loss_attribution(records())
    assert (a["wins"],a["losses"],a["flats"])==(1,2,1)
    assert a["average_net_winner"]==3.5
    assert a["average_net_loser"]==(-2.25-.25)/2
    assert a["mean_holding_candles"]==a["median_holding_candles"]==1.5
    assert a["by_exit_reason"]["take_profit"]["trades"]==2
    assert a["by_exit_reason"]["strategy_exit"]["trades"]==0


def test_single_simulation_and_no_writes(tmp_path,monkeypatch,capsys):
    path=synthetic_source(tmp_path);before=path.read_bytes()
    trades=records();result=BacktestResult(trades=trades,metrics=calculate_metrics(trades))
    simulate=Mock(return_value=result)
    monkeypatch.setattr(backtester,"simulate_strategy",simulate)
    a=args(path);a.loss_attribution=True
    backtester.run_backtest(a)
    output=json.loads(capsys.readouterr().out)
    simulate.assert_called_once()
    assert output["loss_attribution"]["totals"]["net_pnl"]==output["metrics"]["net_pnl"]
    assert output["loss_attribution"]["totals"]["total_fees"]==output["metrics"]["total_fees"]
    assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize("changes", [dict(json=False),dict(execution_mode="signal_close")])
def test_requires_diagnostic_next_open(tmp_path,monkeypatch,changes):
    path=synthetic_source(tmp_path);a=args(path,**changes);a.loss_attribution=True
    simulate=Mock(side_effect=AssertionError("No simulation"))
    monkeypatch.setattr(backtester,"simulate_strategy",simulate)
    with pytest.raises(ValueError):backtester.run_backtest(a)
    simulate.assert_not_called()
