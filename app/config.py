from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mode_name: str
    exchange_name: str
    symbol: str
    capital_usd: float
    max_daily_loss: float
    daily_profit_target: float
    monthly_profit_target: float
    trading_days_per_month: int
    risk_per_trade_pct: float
    max_position_size_usd: float
    max_trades_per_day: int
    strategy_enabled: bool
    initial_balance: float
    paper_mode: bool


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def load_settings(paper_mode: bool) -> Settings:
    monthly_profit_target = _get_float("MONTHLY_PROFIT_TARGET", 500.0)
    trading_days_per_month = _get_int("TRADING_DAYS_PER_MONTH", 20)
    daily_profit_target = _get_float(
        "DAILY_PROFIT_TARGET",
        monthly_profit_target / trading_days_per_month,
    )

    return Settings(
        mode_name=os.getenv("OPERATING_MODE", "500_usd_per_month"),
        exchange_name=os.getenv("EXCHANGE_NAME", "binance"),
        symbol=os.getenv("SYMBOL", "BTC/USDT"),
        capital_usd=_get_float("CAPITAL_USD", 5000.0),
        max_daily_loss=_get_float("MAX_DAILY_LOSS", 25.0),
        daily_profit_target=daily_profit_target,
        monthly_profit_target=monthly_profit_target,
        trading_days_per_month=trading_days_per_month,
        risk_per_trade_pct=_get_float("RISK_PER_TRADE_PCT", 0.005),
        max_position_size_usd=_get_float("MAX_POSITION_SIZE_USD", 250.0),
        max_trades_per_day=_get_int("MAX_TRADES_PER_DAY", 10),
        strategy_enabled=os.getenv("STRATEGY_ENABLED", "false").lower() == "true",
        initial_balance=_get_float("INITIAL_BALANCE", _get_float("CAPITAL_USD", 5000.0)),
        paper_mode=True,
    )
