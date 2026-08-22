"""Tests for the frozen volatility-exhaustion baseline."""

from pathlib import Path

import pandas as pd

from app.research.backtester import STRATEGIES
from app.research.simulation import (
    BacktestResult,
    BacktestTrade,
    calculate_metrics,
    simulate_strategy,
)
from app.research.strategies.volatility_exhaustion import (
    VolatilityExhaustionStrategy,
)
from app.research.volatility_exhaustion_report import (
    build_volatility_exhaustion_report,
    determine_baseline_verdict,
    write_volatility_exhaustion_report,
)


def _candles(closes: list[float], lower: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="15min"),
            "open": closes,
            "high": [value * 1.004 for value in closes],
            "low": [value * 0.996 for value in closes],
            "close": closes,
            "volume": 1.0,
            "bb_lower": lower,
        }
    )


def test_entry_requires_prior_excursion_and_current_reentry() -> None:
    strategy = VolatilityExhaustionStrategy()

    no_excursion = strategy.generate_entries(_candles([101.0, 102.0], [100.0, 100.0]))
    still_below = strategy.generate_entries(_candles([99.0, 98.0], [100.0, 100.0]))
    reentry = strategy.generate_entries(_candles([99.0, 100.0], [100.0, 100.0]))

    assert no_excursion.tolist() == [False, False]
    assert still_below.tolist() == [False, False]
    assert reentry.tolist() == [False, True]


def test_entry_uses_no_future_candle() -> None:
    strategy = VolatilityExhaustionStrategy()
    original = _candles([99.0, 100.0, 500.0], [100.0, 100.0, 1.0])
    changed_future = original.copy()
    changed_future.loc[2, ["close", "bb_lower"]] = [0.5, 1000.0]

    assert strategy.generate_entries(original).iloc[:2].equals(
        strategy.generate_entries(changed_future).iloc[:2]
    )


def test_nan_warmup_is_false_and_index_is_preserved() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0, 101.0], [float("nan"), 100.0, 100.0])
    frame.index = pd.Index([10, 20, 40], name="candle")

    entries = strategy.generate_entries(frame)

    assert entries.index.equals(frame.index)
    assert entries.dtype == bool
    assert not entries.any()


def test_frozen_long_only_parameters_and_exits() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0], [100.0, 100.0])

    assert strategy.direction == "long_only"
    assert strategy.take_profit_pct() == 0.003
    assert strategy.stop_loss_pct() == 0.002
    assert strategy.max_holding_candles() == 25
    assert strategy.generate_exits(frame).eq(False).all()
    assert STRATEGIES["volatility_exhaustion"] is VolatilityExhaustionStrategy


def test_official_simulator_executes_reentry_at_close() -> None:
    strategy = VolatilityExhaustionStrategy()
    frame = _candles([99.0, 100.0, 100.4], [100.0, 100.0, 100.0])

    result = simulate_strategy(frame, strategy, fee_rate=0.0)

    assert result.metrics.total_trades == 1
    assert result.trades[0].entry_index == 1
    assert result.trades[0].entry_price == 100.0
    assert result.trades[0].exit_reason == "take_profit"


def _trade(
    entry: str,
    exit_: str,
    net_pnl: float,
    reason: str,
    holding: int,
) -> BacktestTrade:
    return BacktestTrade(
        entry_index=0,
        exit_index=holding,
        entry_timestamp=pd.Timestamp(entry),
        exit_timestamp=pd.Timestamp(exit_),
        entry_price=100.0,
        exit_price=101.0,
        notional=100.0,
        gross_pnl=net_pnl + 0.1,
        fees=0.1,
        net_pnl=net_pnl,
        exit_reason=reason,
    )


def test_baseline_report_uses_official_metrics_and_trade_records(tmp_path: Path) -> None:
    trades = [
        _trade("2025-01-01", "2025-01-02", 0.9, "take_profit", 3),
        _trade("2025-02-01", "2025-02-02", -0.6, "stop_loss", 5),
    ]
    result = BacktestResult(trades=trades, metrics=calculate_metrics(trades))

    report = build_volatility_exhaustion_report(
        result=result,
        total_candles=1_000,
        feature_rows=980,
        data_path="data/BTCUSDT_1m.csv",
        timeframe="15m",
        parameters={
            "take_profit_pct": 0.003,
            "stop_loss_pct": 0.002,
            "max_holding_candles": 25,
        },
    )
    output = tmp_path / "reports" / "baseline.md"
    write_volatility_exhaustion_report(report, output)

    persisted = output.read_text(encoding="utf-8")
    assert "| Total candles | 1000 |" in persisted
    assert "| Feature rows after warm-up | 980 |" in persisted
    assert "| Total trades | 2 |" in persisted
    assert "| Wins | 1 |" in persisted
    assert "| Losses | 1 |" in persisted
    assert "| Fees | 0.2000 USDT |" in persisted
    assert "| Average holding candles | 4.00 |" in persisted
    assert "| stop_loss | 1 |" in persisted
    assert "| take_profit | 1 |" in persisted
    assert "| 2025-01 | 1 |" in persisted
    assert "| 2025-02 | 1 |" in persisted
    assert "**INSUFFICIENT_SAMPLE**" in persisted


def test_baseline_verdict_rules_are_deterministic() -> None:
    losing_trades = [
        _trade("2025-01-01", "2025-01-02", -0.1, "stop_loss", 1)
        for _ in range(100)
    ]
    winning_trades = [
        _trade("2025-01-01", "2025-01-02", 0.1, "take_profit", 1)
        for _ in range(100)
    ]

    assert determine_baseline_verdict(
        BacktestResult(trades=losing_trades, metrics=calculate_metrics(losing_trades))
    ) == "BASELINE_REJECT"
    assert determine_baseline_verdict(
        BacktestResult(trades=winning_trades, metrics=calculate_metrics(winning_trades))
    ) == "BASELINE_CANDIDATE"
