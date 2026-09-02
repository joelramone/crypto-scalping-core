"""Synthetic tests for symmetric long and short simulation."""

from typing import ClassVar, Literal

import pandas as pd
import pytest
from pydantic import ValidationError

from app.research.analysis.trade_diagnostics import calculate_trade_diagnostics
from app.research.simulation import BacktestTrade, simulate_strategy
from app.research.strategies.base import BaseStrategy
from app.research.strategies.regime_transition import RegimeTransitionStrategy


class SyntheticStrategy(BaseStrategy):
    def __init__(
        self,
        side: Literal["long", "short"] = "long",
        exit_at: int | None = None,
        holding: int = 2,
    ) -> None:
        self.side = side
        self.exit_at = exit_at
        self.holding = holding

    def direction(self) -> Literal["long", "short"]:
        return self.side

    def take_profit_pct(self) -> float:
        return 0.10

    def stop_loss_pct(self) -> float:
        return 0.05

    def max_holding_candles(self) -> int:
        return self.holding

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([True] + [False] * (len(df) - 1), index=df.index)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(False, index=df.index)
        if self.exit_at is not None:
            result.iloc[self.exit_at] = True
        return result

    def name(self) -> str:
        return "synthetic"


class LegacyLongStrategy(SyntheticStrategy):
    direction: ClassVar[Literal["long_only"]] = "long_only"


def candles(*rows: tuple[float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=i),
                "close": close,
                "high": high,
                "low": low,
            }
            for i, (close, high, low) in enumerate(rows)
        ]
    )


def trade(df: pd.DataFrame, **kwargs: object) -> BacktestTrade:
    result = simulate_strategy(df, SyntheticStrategy(**kwargs), fee_rate=0.001)
    assert len(result.trades) == 1
    return result.trades[0]


@pytest.mark.parametrize(
    ("side", "row", "reason", "price", "gross"),
    [
        ("long", (109.0, 111.0, 99.0), "take_profit", 110.0, 10.0),
        ("long", (96.0, 101.0, 94.0), "stop_loss", 95.0, -5.0),
        ("short", (91.0, 101.0, 89.0), "take_profit", 90.0, 10.0),
        ("short", (104.0, 106.0, 99.0), "stop_loss", 105.0, -5.0),
    ],
)
def test_protective_exits(side, row, reason, price, gross):
    completed = trade(candles((100, 100, 100), row, (100, 100, 100)), side=side)
    assert completed.side == side
    assert completed.exit_reason == reason
    assert completed.exit_price == pytest.approx(price)
    assert completed.gross_pnl == pytest.approx(gross)


@pytest.mark.parametrize(
    ("side", "last_close", "gross"),
    [("long", 102.0, 2.0), ("short", 98.0, 2.0), ("short", 102.0, -2.0)],
)
def test_max_holding(side, last_close, gross):
    completed = trade(
        candles((100, 100, 100), (100, 101, 99), (last_close, 104, 96)),
        side=side,
    )
    assert completed.exit_reason == "max_holding"
    assert completed.holding_candles == 2
    assert completed.gross_pnl == pytest.approx(gross)


@pytest.mark.parametrize("side", ["long", "short"])
def test_fees_include_entry_and_exit_notional(side):
    completed = trade(
        candles((100, 100, 100), (100, 101, 99), (100, 101, 99)), side=side
    )
    assert completed.fees == pytest.approx(0.2)
    assert completed.net_pnl == pytest.approx(-0.2)


@pytest.mark.parametrize("side", ["long", "short"])
def test_same_candle_stop_precedes_take_profit(side):
    completed = trade(
        candles((100, 100, 100), (100, 111, 89), (100, 100, 100)), side=side
    )
    assert completed.exit_reason == "stop_loss"


@pytest.mark.parametrize(
    ("side", "close", "gross"),
    [("long", 103.0, 3.0), ("short", 97.0, 3.0), ("short", 103.0, -3.0)],
)
def test_strategy_exit(side, close, gross):
    completed = trade(
        candles((100, 100, 100), (close, 104, 96), (100, 100, 100)),
        side=side,
        exit_at=1,
    )
    assert completed.exit_reason == "strategy_exit"
    assert completed.gross_pnl == pytest.approx(gross)


@pytest.mark.parametrize("side", ["long", "short"])
def test_protective_level_precedes_strategy_exit(side):
    completed = trade(
        candles((100, 100, 100), (100, 111, 89), (100, 100, 100)),
        side=side,
        exit_at=1,
    )
    assert completed.exit_reason == "stop_loss"


def historical_trade_dict() -> dict[str, object]:
    return {
        "entry_index": 0,
        "exit_index": 1,
        "entry_timestamp": "2025-01-01",
        "exit_timestamp": "2025-01-02",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "notional": 100.0,
        "gross_pnl": 1.0,
        "fees": 0.2,
        "net_pnl": 0.8,
        "exit_reason": "max_holding",
    }


def test_missing_side_and_historical_dict_default_to_long():
    completed = BacktestTrade(**historical_trade_dict())
    assert completed.side == "long"
    assert "side" not in completed.model_fields_set


def test_invalid_side_is_rejected():
    with pytest.raises(ValidationError):
        BacktestTrade(**historical_trade_dict(), side="flat")


def test_existing_default_and_legacy_strategies_execute_long():
    df = candles((100, 100, 100), (101, 102, 99), (102, 103, 100))
    assert SyntheticStrategy().direction() == "long"
    assert RegimeTransitionStrategy.direction == "long_only"
    assert simulate_strategy(df, LegacyLongStrategy()).trades[0].side == "long"


def test_existing_long_execution_semantics_are_unchanged():
    completed = simulate_strategy(
        candles((100, 100, 100), (109, 111, 99), (100, 100, 100)),
        LegacyLongStrategy(),
        fee_rate=0.001,
    ).trades[0]
    assert completed.exit_reason == "take_profit"
    assert completed.exit_price == pytest.approx(110.0)
    assert completed.gross_pnl == pytest.approx(10.0)


def test_trade_diagnostics_are_side_label_independent():
    long_trade = BacktestTrade(**historical_trade_dict(), side="long")
    short_trade = BacktestTrade(**historical_trade_dict(), side="short")
    assert calculate_trade_diagnostics([long_trade]) == calculate_trade_diagnostics(
        [short_trade]
    )
