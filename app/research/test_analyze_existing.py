from pathlib import Path

import pandas as pd

from app.research.automation.analyze_existing import analyze_existing, render_failure_pattern_report


def _leaderboard(path: Path, strategy: str, gross: float, fees: float, net: float) -> Path:
    pd.DataFrame([{
        "strategy": strategy,
        "timeframe": "15m",
        "total_trades": 100,
        "gross_pnl_before_fees": gross,
        "total_fees": fees,
        "net_pnl": net,
        "gross_profit_factor": 1.1,
        "net_profit_factor": 0.9,
        "gross_expectancy": gross / 100,
        "net_expectancy": net / 100,
        "suppression_rate": 0.2,
        "profitable_month_percentage": 0.5,
    }]).to_csv(path, index=False)
    return path


def test_analyze_existing_classifies_failure_modes(tmp_path: Path) -> None:
    paths = [
        _leaderboard(tmp_path / "no_edge.csv", "no_edge", -1.0, 5.0, -6.0),
        _leaderboard(tmp_path / "fees.csv", "fees", 2.0, 5.0, -3.0),
        _leaderboard(tmp_path / "positive.csv", "positive", 8.0, 2.0, 6.0),
    ]
    result = analyze_existing(paths)
    assert [item["failure_mode"] for item in result] == [
        "NO_GROSS_EDGE",
        "FEE_DOMINATED",
        "POSITIVE_NET_EDGE",
    ]


def test_report_contains_cross_family_summary(tmp_path: Path) -> None:
    paths = [
        _leaderboard(tmp_path / "a.csv", "family_a", -1.0, 4.0, -5.0),
        _leaderboard(tmp_path / "b.csv", "family_b", 1.0, 4.0, -3.0),
    ]
    report = render_failure_pattern_report(analyze_existing(paths))
    assert "Baselines analyzed: 2" in report
    assert "NO_GROSS_EDGE: 1" in report
    assert "FEE_DOMINATED: 1" in report
    assert "family_a" in report
    assert "family_b" in report
    assert "No optimizer execution" in report
