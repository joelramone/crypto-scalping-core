from __future__ import annotations

import os
from dataclasses import dataclass


MODE_500_USD = {
    "initial_balance": 250.0,
    "trade_size_usdt": 50.0,
    "max_trades_per_day": 10,
    "target_daily_profit_usdt": 2.0,
    "max_daily_loss_usdt": 1.0,
    "buy_threshold_pct": -0.3,
    "sell_threshold_pct": 0.3,
    "stop_loss_pct": 0.4,
    "take_profit_pct": 0.6,
    "fee_rate": 0.001,
}


@dataclass(frozen=True)
class TradingCoreConfig:
    MONTE_CARLO_MAX_ITER: int = int(os.getenv("MONTE_CARLO_MAX_ITER", "2000"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    DAILY_MAX_LOSS: float = float(os.getenv("DAILY_MAX_LOSS", "0.03"))
    FEE_RATE: float = float(os.getenv("FEE_RATE", "0.0004"))
    SLIPPAGE_ESTIMATE: float = float(os.getenv("SLIPPAGE_ESTIMATE", "0.0002"))


TRADING_CORE_CONFIG = TradingCoreConfig()
