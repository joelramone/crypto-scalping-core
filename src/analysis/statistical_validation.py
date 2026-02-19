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


class MonteCarloBootstrap:
    """Bootstrap Monte Carlo over realized trade R-multiples.

    This implementation does **not** shuffle OHLC prices or alter market time structure.
    It resamples only the distribution of realized R-multiples from already executed trades.
    """

    _R_KEYS = ("r_multiple", "r_value", "r", "R")

    def __init__(
        self,
        trades: list[Any],
        runs: int = 10_000,
        initial_equity: float = 0.0,
        random_state: int | None = 42,
    ):
        if not trades:
            raise ValueError("trades must be a non-empty list")
        if runs <= 0:
            raise ValueError("runs must be > 0")

        self.trades = trades
        self.runs = int(runs)
        self.initial_equity = float(initial_equity)
        self.r_values = self._extract_r_values(trades)
        self._rng = np.random.default_rng(random_state)

    @classmethod
    def _extract_r_values(cls, trades: list[Any]) -> np.ndarray:
        r_values: list[float] = []

        for trade in trades:
            value: Any | None = None

            if isinstance(trade, dict):
                for key in cls._R_KEYS:
                    if key in trade:
                        value = trade[key]
                        break
            else:
                for key in cls._R_KEYS:
                    if hasattr(trade, key):
                        value = getattr(trade, key)
                        break

            if value is None:
                continue

            try:
                r_values.append(float(value))
            except (TypeError, ValueError):
                continue

        if not r_values:
            raise ValueError("No valid R-multiples found in trades")

        return np.asarray(r_values, dtype=float)

    def run(self) -> dict[str, Any]:
        n = int(self.r_values.shape[0])
        expectancy_dist = np.empty(self.runs, dtype=float)
        profit_factor_dist = np.empty(self.runs, dtype=float)
        max_drawdown_dist = np.empty(self.runs, dtype=float)
        sharpe_dist = np.empty(self.runs, dtype=float)
        equity_curves: list[list[float]] = []

        for i in range(self.runs):
            sample = self._rng.choice(self.r_values, size=n, replace=True)
            equity_curve = self._build_equity_curve(sample)

            expectancy_dist[i] = float(np.mean(sample))
            profit_factor_dist[i] = self._profit_factor(sample)
            max_drawdown_dist[i] = self._max_drawdown(equity_curve)
            sharpe_dist[i] = self._sharpe(sample)
            equity_curves.append(equity_curve.tolist())

        distributions = {
            "expectancy": expectancy_dist.tolist(),
            "profit_factor": profit_factor_dist.tolist(),
            "max_drawdown": max_drawdown_dist.tolist(),
            "sharpe": sharpe_dist.tolist(),
            "equity_curves": equity_curves,
        }

        percentiles = {
            "expectancy": self._percentiles(expectancy_dist),
            "profit_factor": self._percentiles(profit_factor_dist),
            "max_drawdown": self._percentiles(max_drawdown_dist),
            "sharpe": self._percentiles(sharpe_dist),
        }

        return {
            "runs": self.runs,
            "trades_per_run": n,
            "r_values": self.r_values.tolist(),
            "distributions": distributions,
            "percentiles": percentiles,
        }

    def _build_equity_curve(self, sample: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(sample, dtype=float)
        return np.concatenate(([self.initial_equity], self.initial_equity + cumulative))

    @staticmethod
    def _profit_factor(sample: np.ndarray) -> float:
        gross_profit = float(sample[sample > 0].sum())
        gross_loss = float(np.abs(sample[sample < 0].sum()))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = peaks - equity_curve
        return float(np.max(drawdowns))

    @staticmethod
    def _sharpe(sample: np.ndarray) -> float:
        if sample.shape[0] < 2:
            return 0.0

        std = float(np.std(sample, ddof=1))
        if std == 0:
            return 0.0

        return float(np.sqrt(sample.shape[0]) * (float(np.mean(sample)) / std))

    @staticmethod
    def _percentiles(values: np.ndarray) -> dict[str, float]:
        p05, p50, p95 = np.percentile(values, [5, 50, 95])
        return {
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
        }


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

    @staticmethod
    def _compute_metrics(returns: np.ndarray) -> dict[str, float]:
        trades = int(returns.shape[0])
        expectancy = float(np.mean(returns)) if trades else 0.0

        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(np.abs(returns[returns < 0].sum()))
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = float(gross_profit / gross_loss)

        if trades < 2:
            sharpe = 0.0
        else:
            std = float(np.std(returns, ddof=1))
            sharpe = 0.0 if std == 0 else float(np.sqrt(trades) * (expectancy / std))

        return {
            "trades": float(trades),
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
        }

    def shuffle_test_returns(self, iterations: int = 5000, random_state: int = 7) -> dict[str, Any]:
        """Permutation test using sign-shuffling to stress directional edge.

        This preserves absolute return magnitude and randomizes direction.
        """
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        rng = np.random.default_rng(random_state)
        returns = self.pnl.to_numpy(dtype=float)
        observed = self._compute_metrics(returns)

        abs_returns = np.abs(returns)
        shuffled_expectancies = np.empty(iterations, dtype=float)
        shuffled_sharpes = np.empty(iterations, dtype=float)

        for i in range(iterations):
            random_signs = rng.choice(np.array([-1.0, 1.0]), size=abs_returns.shape[0], replace=True)
            shuffled = abs_returns * random_signs
            shuffled_metrics = self._compute_metrics(shuffled)
            shuffled_expectancies[i] = shuffled_metrics["expectancy"]
            shuffled_sharpes[i] = shuffled_metrics["sharpe"]

        p_value_expectancy = float((np.sum(shuffled_expectancies >= observed["expectancy"]) + 1) / (iterations + 1))
        p_value_sharpe = float((np.sum(shuffled_sharpes >= observed["sharpe"]) + 1) / (iterations + 1))

        return {
            "observed": observed,
            "null_distribution": {
                "expectancy_mean": float(np.mean(shuffled_expectancies)),
                "expectancy_std": float(np.std(shuffled_expectancies, ddof=1)),
                "sharpe_mean": float(np.mean(shuffled_sharpes)),
                "sharpe_std": float(np.std(shuffled_sharpes, ddof=1)),
            },
            "p_values": {
                "expectancy": p_value_expectancy,
                "sharpe": p_value_sharpe,
            },
        }

    def bootstrap_resampling_real(self, iterations: int = 5000, random_state: int = 11) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        rng = np.random.default_rng(random_state)
        returns = self.pnl.to_numpy(dtype=float)
        n = returns.shape[0]

        expectancy = np.empty(iterations, dtype=float)
        profit_factor = np.empty(iterations, dtype=float)
        sharpe = np.empty(iterations, dtype=float)

        for i in range(iterations):
            sample = rng.choice(returns, size=n, replace=True)
            metrics = self._compute_metrics(sample)
            expectancy[i] = metrics["expectancy"]
            profit_factor[i] = metrics["profit_factor"]
            sharpe[i] = metrics["sharpe"]

        return {
            "expectancy_ci_95": [float(x) for x in np.percentile(expectancy, [2.5, 97.5])],
            "profit_factor_ci_95": [float(x) for x in np.percentile(profit_factor, [2.5, 97.5])],
            "sharpe_ci_95": [float(x) for x in np.percentile(sharpe, [2.5, 97.5])],
            "means": {
                "expectancy": float(np.mean(expectancy)),
                "profit_factor": float(np.mean(profit_factor[np.isfinite(profit_factor)])) if np.isfinite(profit_factor).any() else float("inf"),
                "sharpe": float(np.mean(sharpe)),
            },
        }

    def walk_forward_split(self, folds: int = 4, min_train_fraction: float = 0.5) -> dict[str, Any]:
        if folds < 2:
            raise ValueError("folds must be >= 2")
        if not 0 < min_train_fraction < 1:
            raise ValueError("min_train_fraction must be in (0,1)")

        returns = self.pnl.to_numpy(dtype=float)
        n = returns.shape[0]
        min_train = max(2, int(n * min_train_fraction))
        test_block = max(1, (n - min_train) // folds)

        segments: list[dict[str, Any]] = []
        start = min_train
        while start < n:
            end = min(n, start + test_block)
            train = returns[:start]
            test = returns[start:end]
            if test.shape[0] == 0:
                break
            segments.append(
                {
                    "train_range": [0, int(start - 1)],
                    "test_range": [int(start), int(end - 1)],
                    "train_metrics": self._compute_metrics(train),
                    "test_metrics": self._compute_metrics(test),
                    "expectancy_decay": float(self._compute_metrics(test)["expectancy"] - self._compute_metrics(train)["expectancy"]),
                }
            )
            start = end

        test_expectancies = np.asarray([s["test_metrics"]["expectancy"] for s in segments], dtype=float)
        test_sharpes = np.asarray([s["test_metrics"]["sharpe"] for s in segments], dtype=float)

        return {
            "folds": segments,
            "aggregate": {
                "oos_expectancy_mean": float(np.mean(test_expectancies)) if segments else 0.0,
                "oos_sharpe_mean": float(np.mean(test_sharpes)) if segments else 0.0,
                "oos_expectancy_std": float(np.std(test_expectancies, ddof=1)) if len(segments) > 1 else 0.0,
            },
        }

    def out_of_sample_test(self, holdout_fraction: float = 0.25) -> dict[str, Any]:
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0, 1)")

        returns = self.pnl.to_numpy(dtype=float)
        split_idx = max(2, int(returns.shape[0] * (1 - holdout_fraction)))
        train = returns[:split_idx]
        test = returns[split_idx:]

        train_metrics = self._compute_metrics(train)
        test_metrics = self._compute_metrics(test)
        return {
            "split_index": int(split_idx),
            "in_sample": train_metrics,
            "out_of_sample": test_metrics,
            "degradation": {
                "expectancy": float(test_metrics["expectancy"] - train_metrics["expectancy"]),
                "profit_factor": float(test_metrics["profit_factor"] - train_metrics["profit_factor"]) if np.isfinite(test_metrics["profit_factor"]) and np.isfinite(train_metrics["profit_factor"]) else float("nan"),
                "sharpe": float(test_metrics["sharpe"] - train_metrics["sharpe"]),
            },
        }

    def seed_variability_test(self, seeds: list[int] | None = None, iterations: int = 2000) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")
        seed_list = seeds or [3, 7, 13, 29, 43, 101]
        if not seed_list:
            raise ValueError("seeds must be non-empty")

        returns = self.pnl.to_numpy(dtype=float)
        n = returns.shape[0]
        summaries: list[dict[str, float]] = []

        for seed in seed_list:
            rng = np.random.default_rng(seed)
            expectancy_values = np.empty(iterations, dtype=float)
            for i in range(iterations):
                sample = rng.choice(returns, size=n, replace=True)
                expectancy_values[i] = float(np.mean(sample))

            summaries.append(
                {
                    "seed": float(seed),
                    "expectancy_mean": float(np.mean(expectancy_values)),
                    "expectancy_std": float(np.std(expectancy_values, ddof=1)),
                }
            )

        seed_means = np.asarray([s["expectancy_mean"] for s in summaries], dtype=float)
        return {
            "per_seed": summaries,
            "stability": {
                "mean_of_means": float(np.mean(seed_means)),
                "std_of_means": float(np.std(seed_means, ddof=1)) if seed_means.shape[0] > 1 else 0.0,
                "coefficient_of_variation": float(np.std(seed_means, ddof=1) / np.abs(np.mean(seed_means))) if seed_means.shape[0] > 1 and np.mean(seed_means) != 0 else 0.0,
            },
        }

    def randomized_entry_timing_test(self, iterations: int = 3000, random_state: int = 17) -> dict[str, Any]:
        """Random circular shifts to evaluate timing sensitivity without changing outcomes."""
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        rng = np.random.default_rng(random_state)
        returns = self.pnl.to_numpy(dtype=float)
        n = returns.shape[0]

        observed = self._compute_metrics(returns)
        shifted_drawdowns = np.empty(iterations, dtype=float)

        for i in range(iterations):
            shift = int(rng.integers(0, n))
            shifted = np.roll(returns, shift)
            equity = self.initial_capital + np.cumsum(shifted)
            peaks = np.maximum.accumulate(equity)
            shifted_drawdowns[i] = float(np.max((peaks - equity) / peaks))

        equity = self.initial_capital + np.cumsum(returns)
        peaks = np.maximum.accumulate(equity)
        observed_dd = float(np.max((peaks - equity) / peaks))

        return {
            "observed": {
                "expectancy": observed["expectancy"],
                "sharpe": observed["sharpe"],
                "max_drawdown": observed_dd,
            },
            "randomized_timing": {
                "max_drawdown_mean": float(np.mean(shifted_drawdowns)),
                "max_drawdown_ci_95": [float(x) for x in np.percentile(shifted_drawdowns, [2.5, 97.5])],
                "p_value_drawdown_better": float((np.sum(shifted_drawdowns <= observed_dd) + 1) / (iterations + 1)),
            },
        }

    def simulation_independence_test(self, iterations: int = 3000, random_state: int = 19) -> dict[str, Any]:
        if iterations <= 1:
            raise ValueError("iterations must be > 1")

        rng = np.random.default_rng(random_state)
        returns = self.pnl.to_numpy(dtype=float)
        n = returns.shape[0]

        samples = np.empty((iterations, n), dtype=float)
        signatures: set[bytes] = set()
        for i in range(iterations):
            sample = rng.choice(returns, size=n, replace=True)
            samples[i] = sample
            signatures.add(sample.tobytes())

        expectancy_series = samples.mean(axis=1)
        lag_corr = np.corrcoef(expectancy_series[:-1], expectancy_series[1:])[0, 1]

        return {
            "iterations": iterations,
            "unique_path_ratio": float(len(signatures) / iterations),
            "lag1_expectancy_correlation": float(lag_corr),
        }

    def full_robustness_report(self) -> dict[str, Any]:
        return {
            "baseline": {
                "expectancy": self.expectancy(),
                "profit_factor": self.profit_factor(),
                "sharpe": self.sharpe_ratio(),
                "trades": float(self.n_trades),
            },
            "shuffle_test": self.shuffle_test_returns(),
            "bootstrap_real": self.bootstrap_resampling_real(),
            "walk_forward": self.walk_forward_split(),
            "out_of_sample": self.out_of_sample_test(),
            "seed_variability": self.seed_variability_test(),
            "randomized_timing": self.randomized_entry_timing_test(),
            "simulation_independence": self.simulation_independence_test(),
        }
