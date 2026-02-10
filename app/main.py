from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.agents.risk_agent import RiskAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.config import load_settings
from app.data.features import FeatureBuilder
from app.data.market_stream import MarketStream
from app.storage.trades_repo import TradeRecord, TradesRepository
from app.trading.executor import Executor
from app.trading.paper_wallet import PaperWallet
from app.utils.logger import get_logger


def run(paper_mode: bool, steps: int) -> None:
    logger = get_logger("main")
    settings = load_settings(paper_mode=paper_mode)

    stream = MarketStream(symbol=settings.symbol)
    feature_builder = FeatureBuilder()
    strategy_agent = StrategyAgent(settings=settings)
    risk_agent = RiskAgent(settings=settings)
    supervisor_agent = SupervisorAgent(settings=settings)
    wallet = PaperWallet(balance=settings.initial_balance)
    executor = Executor(wallet=wallet)
    trades_repo = TradesRepository()

    pnl_today = 0.0
    trades_today = 0
    ticks = stream.stream()

    for _ in range(steps):
        tick = next(ticks)
        features = feature_builder.build(tick)
        decision = strategy_agent.decide(features)
        approval = risk_agent.evaluate(
            decision=decision,
            pnl_today=pnl_today,
            balance=wallet.balance,
            trades_today=trades_today,
        )

        simulated_exit_price = tick.price * (1 + features.momentum)
        result = executor.execute(
            decision=decision,
            risk=approval,
            entry_price=tick.price,
            exit_price=simulated_exit_price,
        )

        if result.executed:
            trades_today += 1
            pnl_today += result.pnl
            trades_repo.add(
                TradeRecord(
                    timestamp=datetime.now(timezone.utc),
                    symbol=settings.symbol,
                    signal=decision.signal,
                    size=approval.position_size,
                    entry_price=tick.price,
                    exit_price=simulated_exit_price,
                    pnl=result.pnl,
                )
            )
            logger.info("Trade executed | signal=%s pnl=%.4f", decision.signal, result.pnl)
        else:
            logger.info("Trade skipped | reason=%s", result.reason)

        if supervisor_agent.should_stop(pnl_today=pnl_today, trades_today=trades_today):
            logger.warning("Supervisor halted operations | pnl_today=%.4f", pnl_today)
            break

    logger.info("Session finished | balance=%.2f pnl_today=%.2f trades=%s", wallet.balance, pnl_today, len(trades_repo.all()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto scalping core")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulation ticks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(paper_mode=args.paper, steps=args.steps)
