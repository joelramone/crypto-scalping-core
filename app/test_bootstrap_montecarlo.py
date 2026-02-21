import numpy as np

from app.main import run_bootstrap_monte_carlo


def test_run_bootstrap_monte_carlo_shapes_and_ranges():
    returns = [1.0, -1.0, 0.5, -0.5]
    runs = 250
    initial_capital = 100.0

    result = run_bootstrap_monte_carlo(
        returns=returns,
        runs=runs,
        initial_capital=initial_capital,
        risk_per_trade=0.01,
        ruin_threshold_pct=0.5,
        random_seed=7,
    )

    assert result["runs"] == runs
    assert result["trades_per_run"] == len(returns)
    assert len(result["final_equity"]["distribution"]) == runs
    assert len(result["max_drawdown"]["distribution"]) == runs
    assert len(result["max_consecutive_losses"]["distribution"]) == runs
    assert 0.0 <= result["risk_of_ruin"] <= 1.0
    assert all(value >= 0.0 for value in result["max_drawdown"]["distribution"])


def test_run_bootstrap_monte_carlo_reproducible_seed():
    returns = [0.2, -0.1, 0.3, -0.4, 0.1]

    first = run_bootstrap_monte_carlo(returns=returns, runs=120, random_seed=42)
    second = run_bootstrap_monte_carlo(returns=returns, runs=120, random_seed=42)

    assert first["final_equity"]["distribution"] == second["final_equity"]["distribution"]
    assert first["max_drawdown"]["distribution"] == second["max_drawdown"]["distribution"]
    assert first["max_consecutive_losses"]["distribution"] == second["max_consecutive_losses"]["distribution"]


def test_run_bootstrap_monte_carlo_invalid_inputs():
    with np.testing.assert_raises(ValueError):
        run_bootstrap_monte_carlo(returns=[], runs=10)
    with np.testing.assert_raises(ValueError):
        run_bootstrap_monte_carlo(returns=[0.1], runs=0)
    with np.testing.assert_raises(ValueError):
        run_bootstrap_monte_carlo(returns=[0.1], initial_capital=0)
    with np.testing.assert_raises(ValueError):
        run_bootstrap_monte_carlo(returns=[0.1], risk_per_trade=0)
