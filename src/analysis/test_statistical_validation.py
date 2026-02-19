import numpy as np
import pandas as pd

from src.analysis.statistical_validation import StatisticalValidator


def _build_df() -> pd.DataFrame:
    pnl = [1.538, -1.0, 1.538, -1.0, 1.538, -1.0, 1.538, -1.0, 1.538, -1.0, 1.538, -1.0]
    return pd.DataFrame({"pnl": pnl, "initial_capital": [10_000.0] * len(pnl)})


def test_full_robustness_report_contains_requested_sections():
    validator = StatisticalValidator(_build_df())

    report = validator.full_robustness_report()

    assert "shuffle_test" in report
    assert "bootstrap_real" in report
    assert "walk_forward" in report
    assert "out_of_sample" in report
    assert "seed_variability" in report
    assert "randomized_timing" in report
    assert "simulation_independence" in report


def test_shuffle_test_returns_valid_pvalues():
    validator = StatisticalValidator(_build_df())

    result = validator.shuffle_test_returns(iterations=300, random_state=5)

    assert 0.0 <= result["p_values"]["expectancy"] <= 1.0
    assert 0.0 <= result["p_values"]["sharpe"] <= 1.0


def test_seed_variability_is_stable_for_same_input():
    validator = StatisticalValidator(_build_df())

    a = validator.seed_variability_test(seeds=[1, 2, 3], iterations=300)
    b = validator.seed_variability_test(seeds=[1, 2, 3], iterations=300)

    assert np.isclose(a["stability"]["mean_of_means"], b["stability"]["mean_of_means"])
    assert np.isclose(a["stability"]["std_of_means"], b["stability"]["std_of_means"])


def test_simulation_independence_ratio_is_nonzero():
    validator = StatisticalValidator(_build_df())

    result = validator.simulation_independence_test(iterations=300, random_state=31)

    assert result["unique_path_ratio"] > 0.0
    assert -1.0 <= result["lag1_expectancy_correlation"] <= 1.0
