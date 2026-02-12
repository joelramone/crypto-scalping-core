from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyStats:
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0


@dataclass
class StrategyPerformanceTracker:
    _stats_by_strategy: dict[str, StrategyStats] = field(default_factory=dict)

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        stats = self._stats_by_strategy.setdefault(strategy_name, StrategyStats())
        stats.trades_count += 1
        stats.net_profit += pnl

        if pnl > 0:
            stats.wins += 1
            stats.gross_profit += pnl
        elif pnl < 0:
            stats.losses += 1
            stats.gross_loss += abs(pnl)

    def export(self) -> dict[str, dict[str, float]]:
        return {
            strategy_name: {
                "trades_count": stats.trades_count,
                "wins": stats.wins,
                "losses": stats.losses,
                "gross_profit": stats.gross_profit,
                "gross_loss": stats.gross_loss,
                "net_profit": stats.net_profit,
            }
            for strategy_name, stats in self._stats_by_strategy.items()
        }

    @staticmethod
    def compute_win_rate(stats: dict[str, float]) -> float:
        trades_count = int(stats.get("trades_count", 0))
        if trades_count == 0:
            return 0.0
        return (float(stats.get("wins", 0)) / trades_count) * 100

    @staticmethod
    def compute_profit_factor(stats: dict[str, float]) -> float:
        gross_loss = float(stats.get("gross_loss", 0.0))
        if gross_loss == 0:
            return 0.0
        return float(stats.get("gross_profit", 0.0)) / gross_loss

    @staticmethod
    def compute_expectancy(stats: dict[str, float]) -> float:
        trades_count = int(stats.get("trades_count", 0))
        if trades_count == 0:
            return 0.0
        return float(stats.get("net_profit", 0.0)) / trades_count
