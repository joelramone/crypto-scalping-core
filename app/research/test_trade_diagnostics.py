"""Focused tests for permanent official-trade diagnostics."""

from pathlib import Path

from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.discovery.leaderboard_loader import load_leaderboard_csv
from app.research.simulation import BacktestTrade


def _trade(entry: int, gross: float, fee: float, reason: str, month: int = 1) -> BacktestTrade:
    return BacktestTrade(
        entry_index=entry,
        exit_index=entry + 2,
        entry_timestamp=f"2026-{month:02d}-01",
        exit_timestamp=f"2026-{month:02d}-02",
        entry_price=100.0,
        exit_price=101.0,
        notional=100.0,
        gross_pnl=gross,
        fees=fee,
        net_pnl=gross - fee,
        exit_reason=reason,
    )


def test_trade_reconciliation_payoff_exits_and_overlap() -> None:
    trades = [
        _trade(1, 4.5, 0.5, "take_profit"),
        _trade(5, -1.5, 0.5, "stop_loss"),
        _trade(9, 0.5, 0.5, "max_holding", 2),
        _trade(13, 2.5, 0.5, "strategy_exit", 2),
    ]
    diagnostics = calculate_trade_diagnostics(trades, [1, 2, 5, 6, 9, 13])

    assert diagnostics.gross_pnl_before_fees == sum(trade.gross_pnl for trade in trades)
    assert diagnostics.total_fees == sum(trade.fees for trade in trades)
    assert diagnostics.net_pnl == sum(trade.net_pnl for trade in trades)
    assert diagnostics.payoff_ratio == 1.5
    assert diagnostics.break_even_win_rate == 0.4
    assert sum(item.count for item in diagnostics.exits.values()) == len(trades)
    assert diagnostics.raw_entry_signals == 6
    assert diagnostics.suppressed_signals == 2
    assert diagnostics.suppression_rate == 2 / 6
    assert diagnostics.raw_signals_per_opened_trade == 1.5
    assert diagnostics.positive_months == 2
    assert diagnostics.positive_pnl_concentration_top_2_months == 1.0


def test_legacy_leaderboard_loading_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "strategy,rank,total_trades,win_rate,profit_factor,expectancy,max_drawdown,gross_pnl,net_pnl\n"
        "legacy,1,2,0.5,1.2,0.1,1.0,0.4,0.2\n",
        encoding="utf-8",
    )
    loaded = load_leaderboard_csv(path)
    assert loaded.loc[0, "timeframe"] == "1m"
    assert "gross_expectancy" not in loaded.columns
