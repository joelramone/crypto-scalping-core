from __future__ import annotations

from dataclasses import dataclass

from app.agents.strategy_agent import StrategyDecision
from app.config import Settings


@dataclass(frozen=True)
class RiskApproval:
    approved: bool
    reason: str
    position_size: float


class RiskAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, decision: StrategyDecision, pnl_today: float, balance: float, trades_today: int) -> RiskApproval:
        if decision.signal == "hold":
            return RiskApproval(approved=False, reason="No trade signal", position_size=0.0)

        if trades_today >= self.settings.max_trades_per_day:
            return RiskApproval(approved=False, reason="Max daily trades reached", position_size=0.0)

        if pnl_today <= -self.settings.max_daily_loss:
            return RiskApproval(approved=False, reason="Daily loss limit reached", position_size=0.0)

        if pnl_today >= self.settings.daily_profit_target:
            return RiskApproval(approved=False, reason="Daily profit target reached", position_size=0.0)

        nominal_size = balance * self.settings.risk_per_trade_pct
        bounded_size = min(nominal_size, self.settings.max_position_size_usd)

        if bounded_size <= 0:
            return RiskApproval(approved=False, reason="Invalid position size", position_size=0.0)

        return RiskApproval(approved=True, reason="Approved", position_size=bounded_size)
