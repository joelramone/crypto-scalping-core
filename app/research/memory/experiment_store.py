"""CSV-backed storage helpers for Research Memory."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
RESEARCH_DIR = ROOT_DIR / "research"
JOURNAL_DIR = RESEARCH_DIR / "journal"
MEMORY_DIR = RESEARCH_DIR / "memory"
MEMORY_INDEX_PATH = MEMORY_DIR / "index.csv"

MEMORY_INDEX_COLUMNS = [
    "experiment_id",
    "created_at_utc",
    "strategy",
    "timeframe",
    "dataset",
    "config_file",
    "leaderboard_file",
    "total_configurations",
    "eligible_configurations",
    "best_profit_factor",
    "best_expectancy",
    "best_max_drawdown",
    "best_total_trades",
    "best_configuration",
    "status",
]


def ensure_memory_directories() -> None:
    """Ensure the Research Memory directories exist."""
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_existing_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a legacy or canonical row into the canonical schema."""
    return {
        "experiment_id": row.get("experiment_id", ""),
        "created_at_utc": row.get("created_at_utc", ""),
        "strategy": row.get("strategy", ""),
        "timeframe": row.get("timeframe", ""),
        "dataset": row.get("dataset", ""),
        "config_file": row.get("config_file", ""),
        "leaderboard_file": row.get("leaderboard_file", ""),
        "total_configurations": row.get("total_configurations", ""),
        "eligible_configurations": row.get("eligible_configurations", ""),
        "best_profit_factor": row.get("best_profit_factor", ""),
        "best_expectancy": row.get("best_expectancy", ""),
        "best_max_drawdown": row.get("best_max_drawdown", ""),
        "best_total_trades": row.get("best_total_trades", ""),
        "best_configuration": row.get("best_configuration", ""),
        "status": row.get("status", ""),
    }


def migrate_memory_index_schema(index_path: str | Path) -> Path:
    """Upgrade a legacy Research Memory index CSV to the canonical schema."""
    path = Path(index_path)
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if fieldnames == MEMORY_INDEX_COLUMNS:
            return path
        rows = [_normalize_existing_row(row) for row in reader]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEMORY_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def initialize_memory_index(index_path: str | Path | None = None) -> Path:
    """Create the memory index CSV with headers if it does not exist."""
    ensure_memory_directories()
    path = Path(index_path or MEMORY_INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return migrate_memory_index_schema(path)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEMORY_INDEX_COLUMNS)
        writer.writeheader()
    return path


def load_memory_index(index_path: str | Path | None = None) -> pd.DataFrame:
    """Load the Research Memory index as a DataFrame."""
    path = initialize_memory_index(index_path)
    return pd.read_csv(path, keep_default_na=False)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a CSV-ready memory row with all required columns."""
    normalized = {column: row.get(column, "") for column in MEMORY_INDEX_COLUMNS}
    return normalized


def upsert_memory_index_row(
    row: dict[str, Any],
    index_path: str | Path | None = None,
) -> Path:
    """Append or replace a Research Memory index row by experiment ID."""
    path = initialize_memory_index(index_path)
    normalized_row = _normalize_row(row)
    existing_rows: list[dict[str, Any]] = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for existing in reader:
            if existing.get("experiment_id") == normalized_row["experiment_id"]:
                continue
            existing_rows.append(existing)

    existing_rows.append(normalized_row)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEMORY_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)

    return path


def find_experiment_by_leaderboard_file(
    leaderboard_file: str | Path,
    index_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the existing memory row for a leaderboard file, if any."""
    index_df = load_memory_index(index_path)
    normalized_path = str(leaderboard_file)
    matches = index_df[index_df["leaderboard_file"].astype(str) == normalized_path]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()
