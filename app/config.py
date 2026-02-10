from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    exchange_name: str
    symbol: str
    max_daily_loss: float
    daily_profit_target: float
    risk_per_trade_pct: float
    max_position_size: float
    initial_balance: float
    paper_mode: bool


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def load_settings(paper_mode: bool) -> Settings:
    return Settings(
        exchange_name=os.getenv("EXCHANGE_NAME", "binance"),
        symbol=os.getenv("SYMBOL", "BTC/USDT"),
        max_daily_loss=_get_float("MAX_DAILY_LOSS", 100.0),
        daily_profit_target=_get_float("DAILY_PROFIT_TARGET", 200.0),
        risk_per_trade_pct=_get_float("RISK_PER_TRADE_PCT", 0.01),
        max_position_size=_get_float("MAX_POSITION_SIZE", 0.01),
        initial_balance=_get_float("INITIAL_BALANCE", 1000.0),
        paper_mode=paper_mode,
    )
