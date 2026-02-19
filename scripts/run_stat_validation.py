#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.statistical_validation import StatisticalValidator


def _save_plots(validator: StatisticalValidator, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_curve = validator.initial_capital + validator.pnl.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index + 1, equity_curve.values, color="tab:blue", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    equity_path = output_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)

    mc = validator.monte_carlo_simulation(iterations=1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mc["final_equity_distribution"], bins=40, color="tab:green", alpha=0.85)
    ax.set_title("Monte Carlo Final Equity Distribution")
    ax.set_xlabel("Final Equity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    mc_path = output_dir / "monte_carlo_final_equity.png"
    fig.tight_layout()
    fig.savefig(mc_path, dpi=150)
    plt.close(fig)

    rng_results = validator.bootstrap_confidence_interval(n_iterations=1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(rng_results["ci_95_lower"], color="tab:red", linestyle="--", label="95% CI Lower")
    ax.axvline(rng_results["ci_95_upper"], color="tab:purple", linestyle="--", label="95% CI Upper")
    ax.axvline(validator.expectancy(), color="tab:orange", linewidth=2, label="Observed Expectancy")
    ax.set_title("Bootstrap 95% CI for Expectancy")
    ax.set_xlabel("Expectancy")
    ax.set_yticks([])
    ax.legend()
    ax.grid(alpha=0.3)
    bs_path = output_dir / "bootstrap_expectancy_ci.png"
    fig.tight_layout()
    fig.savefig(bs_path, dpi=150)
    plt.close(fig)

    return {
        "equity_curve": str(equity_path),
        "monte_carlo_distribution": str(mc_path),
        "bootstrap_ci": str(bs_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical validation for trade history.")
    parser.add_argument("--csv", type=Path, default=Path("data/sample_trades.csv"), help="Path to trades CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/statistical"),
        help="Directory where charts are written",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Monte Carlo and bootstrap iterations")
    args = parser.parse_args()

    trades_df = pd.read_csv(args.csv)
    validator = StatisticalValidator(trades_df)

    monte_carlo = validator.monte_carlo_simulation(iterations=args.iterations)
    bootstrap = validator.bootstrap_confidence_interval(n_iterations=args.iterations)
    distribution = validator.distribution_test()

    plots = _save_plots(validator, args.output_dir)

    robustness = validator.full_robustness_report()

    result = {
        "input": {
            "csv": str(args.csv),
            "trades": int(validator.n_trades),
            "initial_capital": float(validator.initial_capital),
        },
        "metrics": {
            "win_rate": validator.win_rate(),
            "payoff_ratio": validator.payoff_ratio(),
            "expectancy": validator.expectancy(),
            "profit_factor": validator.profit_factor(),
            "sharpe_ratio": validator.sharpe_ratio(),
            "max_drawdown": validator.max_drawdown(),
        },
        "monte_carlo": monte_carlo,
        "bootstrap": bootstrap,
        "distribution_test": distribution,
        "robustness": robustness,
        "artifacts": plots,
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
