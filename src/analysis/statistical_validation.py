from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MonteCarloResult:
    final_equity_distribution: list[float]
    probability_of_ruin: float
    mean_final_equity: float
    median_final_equity: float
    percentile_5: float
    percentile_95: float


class StatisticalValidator:
    """Statistical validator for trade-level strategy performance."""

    _PNL_COLUMNS = ("pnl", "profit", "net_profit", "returns", "return")
    _PATH_FAILSAFE_PADDING = 5

    def __init__(self, trades_df: pd.DataFrame):
        if trades_df is None or trades_df.empty:
            raise ValueError("trades_df must be a non-empty DataFrame")

        self.trades_df = trades_df.copy()
        self.pnl = self._extract_pnl_series(self.trades_df)
        self.n_trades = int(self.pnl.shape[0])

        if "initial_capital" in self.trades_df.columns:
            initial_capital = float(self.trades_df["initial_capital"].iloc[0])
        else:
            initial_capital = 10_000.0
        self.initial_capital = initial_capital

    @classmethod
    def _extract_pnl_series(cls, trades_df: pd.DataFrame) -> pd.Series:
        for column in cls._PNL_COLUMNS:
            if column in trades_df.columns:
                pnl_series = pd.to_numeric(trades_df[column], errors="coerce").dropna()
                if pnl_series.empty:
                    raise ValueError(f"Column '{column}' contains no numeric values")
                return pnl_series.astype(float).reset_index(drop=True)
        raise ValueError(f"No supported PnL column found. Expected one of: {cls._PNL_COLUMNS}")

    def win_rate(self) -> float:
        return float((self.pnl > 0).mean())

    def payoff_ratio(self) -> float:
        wins = self.pnl[self.pnl > 0]
        losses = self.pnl[self.pnl < 0]
        if losses.empty:
            return float("inf") if not wins.empty else 0.0
        mean_win = float(wins.mean()) if not wins.empty else 0.0
        mean_loss = float(np.abs(losses.mean()))
        if mean_loss == 0:
            return float("inf")
        return float(mean_win / mean_loss)

    def expectancy(self) -> float:
        return float(self.pnl.mean())

    def profit_factor(self) -> float:
        gross_profit = float(self.pnl[self.pnl > 0].sum())
        gross_loss = float(np.abs(self.pnl[self.pnl < 0].sum()))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        excess_returns = self.pnl - risk_free_rate
        std = float(excess_returns.std(ddof=1))
        if std == 0:
            return 0.0
        mean = float(excess_returns.mean())
        return float(np.sqrt(self.n_trades) * (mean / std))

    def max_drawdown(self) -> float:
        equity_curve = self.initial_capital + self.pnl.cumsum()
        rolling_peak = equity_curve.cummax()
        drawdown = (equity_curve - rolling_peak) / rolling_peak
        return float(np.abs(drawdown.min()))

    def monte_carlo_simulation(self, iterations: int = 1000) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        rng = np.random.default_rng(42)
        pnl_values = self.pnl.to_numpy(dtype=float)
        final_equities = np.empty(iterations, dtype=float)
        ruins = 0

        for i in range(iterations):
            shuffled = rng.permutation(pnl_values)
            simulation = self._simulate_equity_path(shuffled)
            final_equities[i] = simulation["final_equity"]
            if simulation["ruined"]:
                ruins += 1

        result = MonteCarloResult(
            final_equity_distribution=final_equities.tolist(),
            probability_of_ruin=float(ruins / iterations),
            mean_final_equity=float(np.mean(final_equities)),
            median_final_equity=float(np.median(final_equities)),
            percentile_5=float(np.percentile(final_equities, 5)),
            percentile_95=float(np.percentile(final_equities, 95)),
        )
        return result.__dict__

    def _simulate_equity_path(self, shuffled_pnl: np.ndarray) -> dict[str, float | bool]:
        """Simulate one shuffled PnL path with explicit loop controls.

        Notes:
        - We advance `idx` on every loop iteration to guarantee forward progress.
        - We terminate when all shuffled trades are consumed (idx >= total_points).
        - We include a failsafe (`max_steps`) as protection against accidental infinite loops
          if this method is modified in the future.
        """

        idx = 0
        steps = 0
        total_points = int(shuffled_pnl.shape[0])
        max_steps = total_points + self._PATH_FAILSAFE_PADDING

        equity = float(self.initial_capital)
        ruined = False

        while idx < total_points:
            # Failsafe to avoid eternal loops if index progress is broken in future changes.
            if steps >= max_steps:
                raise RuntimeError(
                    "Monte Carlo path exceeded failsafe step limit. "
                    "Check index progression and loop termination conditions."
                )

            equity += float(shuffled_pnl[idx])
            if equity <= 0:
                ruined = True

            # Critical for loop safety: always advance index and step counters.
            idx += 1
            steps += 1

        return {"final_equity": equity, "ruined": ruined}

    def bootstrap_confidence_interval(self, n_iterations: int = 1000) -> dict[str, float]:
        if n_iterations <= 0:
            raise ValueError("n_iterations must be > 0")

        rng = np.random.default_rng(42)
        pnl_values = self.pnl.to_numpy(dtype=float)
        n = pnl_values.shape[0]

        expectancy_samples = np.empty(n_iterations, dtype=float)
        for i in range(n_iterations):
            sample = rng.choice(pnl_values, size=n, replace=True)
            expectancy_samples[i] = float(np.mean(sample))

        lower, upper = np.percentile(expectancy_samples, [2.5, 97.5])
        return {
            "expectancy_mean": float(np.mean(expectancy_samples)),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
        }

    def distribution_test(self) -> dict[str, float]:
        pnl_values = self.pnl.to_numpy(dtype=float)

        shapiro_input = pnl_values[:5000] if pnl_values.shape[0] > 5000 else pnl_values
        shapiro_stat, shapiro_p_value = stats.shapiro(shapiro_input)
        t_stat, t_p_value = stats.ttest_1samp(pnl_values, popmean=0.0)

        return {
            "shapiro_statistic": float(shapiro_stat),
            "shapiro_p_value": float(shapiro_p_value),
            "t_statistic": float(t_stat),
            "t_test_p_value": float(t_p_value),
        }
