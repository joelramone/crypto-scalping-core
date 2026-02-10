from __future__ import annotations

from dataclasses import dataclass

from app.agents.risk_agent import RiskApproval
from app.agents.strategy_agent import StrategyDecision
from app.trading.paper_wallet import PaperWallet


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    pnl: float
    reason: str


class Executor:
    def __init__(self, wallet: PaperWallet) -> None:
        self.wallet = wallet

    def execute(
        self,
        decision: StrategyDecision,
        risk: RiskApproval,
        entry_price: float,
        exit_price: float,
    ) -> ExecutionResult:
        if not risk.approved:
            return ExecutionResult(executed=False, pnl=0.0, reason=risk.reason)

        if decision.signal == "hold":
            return ExecutionResult(executed=False, pnl=0.0, reason="Hold signal")

        pnl = self.wallet.apply_trade(
            direction=decision.signal,
            size=risk.position_size,
            entry_price=entry_price,
            exit_price=exit_price,
        )
        return ExecutionResult(executed=True, pnl=pnl, reason="Executed")
