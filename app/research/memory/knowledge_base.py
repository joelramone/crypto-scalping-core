"""Aggregate helpers for reading and summarizing Research Memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.research.memory.experiment_store import load_memory_index


@dataclass(slots=True)
class MemorySummary:
    """Compact summary of the experiment memory index."""

    total_experiments: int
    completed_experiments: int
    strategies_tested: list[str]
    timeframes_tested: list[str]
    best_experiment_id: str
    best_strategy: str
    best_timeframe: str
    best_profit_factor: float
    best_expectancy: float
    average_best_profit_factor: float
    average_best_expectancy: float


def _safe_mean(series: pd.Series) -> float:
    """Return a numeric mean with empty-series protection."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def _safe_float(value: object) -> float:
    """Return a numeric scalar with NaN protection."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def build_memory_summary(index_path: str | Path | None = None) -> MemorySummary:
    """Build an aggregate summary from the Research Memory index."""
    index_df = load_memory_index(index_path)
    if index_df.empty:
        return MemorySummary(
            total_experiments=0,
            completed_experiments=0,
            strategies_tested=[],
            timeframes_tested=[],
            best_experiment_id="n/a",
            best_strategy="n/a",
            best_timeframe="n/a",
            best_profit_factor=0.0,
            best_expectancy=0.0,
            average_best_profit_factor=0.0,
            average_best_expectancy=0.0,
        )

    pf_values = pd.to_numeric(index_df["best_profit_factor"], errors="coerce")
    best_experiment_id = "n/a"
    best_strategy = "n/a"
    best_timeframe = "n/a"
    best_profit_factor = 0.0
    best_expectancy = 0.0
    if pf_values.notna().any():
        best_row = index_df.loc[pf_values.idxmax()]
        best_experiment_id = str(best_row["experiment_id"])
        best_strategy = str(best_row.get("strategy", "") or "n/a")
        best_timeframe = str(best_row.get("timeframe", "") or "n/a")
        best_profit_factor = _safe_float(best_row.get("best_profit_factor"))
        best_expectancy = _safe_float(best_row.get("best_expectancy"))

    return MemorySummary(
        total_experiments=len(index_df),
        completed_experiments=int(index_df["status"].astype(str).eq("completed").sum()),
        strategies_tested=sorted(
            value
            for value in index_df["strategy"].astype(str).unique().tolist()
            if value
        ),
        timeframes_tested=sorted(
            value
            for value in index_df["timeframe"].astype(str).unique().tolist()
            if value
        ),
        best_experiment_id=best_experiment_id,
        best_strategy=best_strategy,
        best_timeframe=best_timeframe,
        best_profit_factor=best_profit_factor,
        best_expectancy=best_expectancy,
        average_best_profit_factor=_safe_mean(index_df["best_profit_factor"]),
        average_best_expectancy=_safe_mean(index_df["best_expectancy"]),
    )
