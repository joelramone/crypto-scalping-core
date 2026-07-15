"""Simple parameter importance based on linear correlations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.research.discovery.leaderboard_loader import get_numeric_parameter_columns


@dataclass(slots=True)
class ParameterImportanceRow:
    """Correlation summary for a single numeric parameter."""

    parameter: str
    profit_factor_correlation: float
    expectancy_correlation: float
    importance_score: float


def _safe_correlation(series: pd.Series, target: pd.Series) -> float:
    """Return a correlation value, defaulting to zero for degenerate inputs."""
    valid = pd.concat([series, target], axis=1).dropna()
    if len(valid) < 2:
        return 0.0
    correlation = valid.iloc[:, 0].corr(valid.iloc[:, 1])
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def compute_parameter_importance(df: pd.DataFrame) -> list[ParameterImportanceRow]:
    """Compute parameter correlations against PF and expectancy."""
    rows: list[ParameterImportanceRow] = []
    for column in get_numeric_parameter_columns(df):
        pf_corr = _safe_correlation(df[column], df["profit_factor"])
        expectancy_corr = _safe_correlation(df[column], df["expectancy"])
        rows.append(
            ParameterImportanceRow(
                parameter=column,
                profit_factor_correlation=pf_corr,
                expectancy_correlation=expectancy_corr,
                importance_score=max(abs(pf_corr), abs(expectancy_corr)),
            )
        )

    rows.sort(key=lambda row: row.importance_score, reverse=True)
    return rows
