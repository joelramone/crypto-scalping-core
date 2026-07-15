"""Sequential experiment ID generation for Research Memory."""

from __future__ import annotations

from pathlib import Path

from app.research.memory.experiment_store import MEMORY_INDEX_PATH, load_memory_index

EXPERIMENT_ID_PREFIX = "EXP-"
EXPERIMENT_ID_WIDTH = 6


def _parse_experiment_number(experiment_id: str) -> int | None:
    """Parse the numeric portion of an experiment ID."""
    if not experiment_id.startswith(EXPERIMENT_ID_PREFIX):
        return None
    numeric_part = experiment_id.removeprefix(EXPERIMENT_ID_PREFIX)
    if not numeric_part.isdigit():
        return None
    return int(numeric_part)


def generate_next_experiment_id(index_path: str | Path | None = None) -> str:
    """Return the next sequential experiment ID."""
    index_df = load_memory_index(index_path)
    max_number = 0
    if "experiment_id" in index_df.columns:
        for experiment_id in index_df["experiment_id"].dropna().astype(str):
            parsed = _parse_experiment_number(experiment_id)
            if parsed is not None:
                max_number = max(max_number, parsed)

    next_number = max_number + 1
    return f"{EXPERIMENT_ID_PREFIX}{next_number:0{EXPERIMENT_ID_WIDTH}d}"
