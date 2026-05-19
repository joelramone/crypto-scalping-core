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
    _stats_by_regime: dict[str, StrategyStats] = field(default_factory=dict)
    _stats_by_score: dict[str, StrategyStats] = field(default_factory=dict)

    def record_trade(self, strategy_name: str, pnl: float, regime: str | None = None, signal_quality_score: int | None = None) -> None:
        strategy_stats = self._stats_by_strategy.setdefault(strategy_name, StrategyStats())
        self._accumulate_stats(strategy_stats, pnl)

        if regime:
            regime_stats = self._stats_by_regime.setdefault(regime, StrategyStats())
            self._accumulate_stats(regime_stats, pnl)
        if signal_quality_score is not None:
            score_key = str(int(signal_quality_score))
            score_stats = self._stats_by_score.setdefault(score_key, StrategyStats())
            self._accumulate_stats(score_stats, pnl)

    @staticmethod
    def _accumulate_stats(stats: StrategyStats, pnl: float) -> None:
        stats.trades_count += 1
        stats.net_profit += pnl

        if pnl > 0:
            stats.wins += 1
            stats.gross_profit += pnl
        elif pnl < 0:
            stats.losses += 1
            stats.gross_loss += abs(pnl)

    def export(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            "by_strategy": {name: self._serialize_stats(stats) for name, stats in self._stats_by_strategy.items()},
            "by_regime": {name: self._serialize_stats(stats) for name, stats in self._stats_by_regime.items()},
            "by_signal_quality_score": {name: self._serialize_stats(stats) for name, stats in self._stats_by_score.items()},
        }

    @staticmethod
    def _serialize_stats(stats: StrategyStats) -> dict[str, float]:
        return {
            "trades_count": stats.trades_count,
            "wins": stats.wins,
            "losses": stats.losses,
            "gross_profit": stats.gross_profit,
            "gross_loss": stats.gross_loss,
            "net_profit": stats.net_profit,
            "win_rate": StrategyPerformanceTracker.compute_win_rate({"trades_count": stats.trades_count, "wins": stats.wins}),
            "expectancy": StrategyPerformanceTracker.compute_expectancy({"trades_count": stats.trades_count, "net_profit": stats.net_profit}),
        }

    @staticmethod
    def compute_win_rate(stats: dict[str, float]) -> float:
        trades_count = int(stats.get("trades_count", 0))
        if trades_count == 0:
            return 0.0
        return (float(stats.get("wins", 0)) / trades_count) * 100

    @staticmethod
    def compute_profit_factor(stats: dict[str, float]) -> float:
        gross_profit = float(stats.get("gross_profit", 0.0))
        gross_loss = float(stats.get("gross_loss", 0.0))

        normalized_gross_loss = abs(gross_loss)
        if normalized_gross_loss == 0.0:
            if gross_profit > 0.0:
                return float("inf")
            return 0.0

        if gross_profit <= 0.0:
            return 0.0

        return gross_profit / normalized_gross_loss

    @staticmethod
    def compute_expectancy(stats: dict[str, float]) -> float:
        trades_count = int(stats.get("trades_count", 0))
        if trades_count == 0:
            return 0.0
        return float(stats.get("net_profit", 0.0)) / trades_count
