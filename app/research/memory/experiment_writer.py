"""Write journals and memory index entries for completed experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.research.memory.experiment_id import generate_next_experiment_id
from app.research.memory.experiment_store import (
    JOURNAL_DIR,
    ensure_memory_directories,
    upsert_memory_index_row,
)
from app.research.memory.journal import build_experiment_journal

if TYPE_CHECKING:
    from app.research.optimizer.grid_search import GridSearchConfig, GridSearchSummary


@dataclass(slots=True)
class ExperimentMemoryArtifacts:
    """Paths and identifiers created for a completed memory write."""

    experiment_id: str
    journal_path: Path
    index_path: Path


def _best_result_fields(summary: GridSearchSummary) -> tuple[str, str, str, str, str, str]:
    """Return best-result fields for the memory index."""
    if not summary.ranked_results:
        return "", "", "", "", "{}", "completed"

    best_result = summary.ranked_results[0]
    return (
        f"{best_result.metrics.profit_factor:.10g}",
        f"{best_result.metrics.expectancy:.10g}",
        f"{best_result.metrics.max_drawdown:.10g}",
        str(best_result.metrics.total_trades),
        json.dumps(best_result.parameters, sort_keys=True),
        "completed",
    )


def write_experiment_memory(
    config: GridSearchConfig,
    summary: GridSearchSummary,
) -> ExperimentMemoryArtifacts:
    """Persist a completed optimizer run into the Research Memory layer."""
    ensure_memory_directories()
    experiment_id = generate_next_experiment_id()
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    journal_markdown = build_experiment_journal(experiment_id, created_at_utc, config, summary)
    journal_path = JOURNAL_DIR / f"{experiment_id}.md"
    journal_path.write_text(journal_markdown, encoding="utf-8")

    (
        best_profit_factor,
        best_expectancy,
        best_max_drawdown,
        best_total_trades,
        best_configuration,
        status,
    ) = _best_result_fields(summary)
    index_path = upsert_memory_index_row(
        {
            "experiment_id": experiment_id,
            "created_at_utc": created_at_utc,
            "strategy": config.strategy,
            "timeframe": config.timeframe,
            "dataset": str(config.data),
            "config_file": str(getattr(config, "config_file", "") or ""),
            "leaderboard_file": str(config.output),
            "total_configurations": str(summary.evaluated_configurations),
            "eligible_configurations": str(len(summary.ranked_results)),
            "best_profit_factor": best_profit_factor,
            "best_expectancy": best_expectancy,
            "best_max_drawdown": best_max_drawdown,
            "best_total_trades": best_total_trades,
            "best_configuration": best_configuration,
            "status": status,
        }
    )
    return ExperimentMemoryArtifacts(
        experiment_id=experiment_id,
        journal_path=journal_path,
        index_path=index_path,
    )


def write_imported_experiment_memory(
    config: GridSearchConfig,
    summary: GridSearchSummary,
) -> ExperimentMemoryArtifacts:
    """Persist a historical leaderboard import into the Research Memory layer."""
    ensure_memory_directories()
    experiment_id = generate_next_experiment_id()
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    journal_markdown = build_experiment_journal(
        experiment_id,
        created_at_utc,
        config,
        summary,
        source="historical leaderboard import",
        optimizer_rerun=False,
    )
    journal_path = JOURNAL_DIR / f"{experiment_id}.md"
    journal_path.write_text(journal_markdown, encoding="utf-8")

    (
        best_profit_factor,
        best_expectancy,
        best_max_drawdown,
        best_total_trades,
        best_configuration,
        _status,
    ) = _best_result_fields(summary)
    index_path = upsert_memory_index_row(
        {
            "experiment_id": experiment_id,
            "created_at_utc": created_at_utc,
            "strategy": config.strategy,
            "timeframe": config.timeframe,
            "dataset": str(config.data),
            "config_file": str(getattr(config, "config_file", "") or ""),
            "leaderboard_file": str(config.output),
            "total_configurations": str(summary.evaluated_configurations),
            "eligible_configurations": str(len(summary.ranked_results)),
            "best_profit_factor": best_profit_factor,
            "best_expectancy": best_expectancy,
            "best_max_drawdown": best_max_drawdown,
            "best_total_trades": best_total_trades,
            "best_configuration": best_configuration,
            "status": "imported",
        }
    )
    return ExperimentMemoryArtifacts(
        experiment_id=experiment_id,
        journal_path=journal_path,
        index_path=index_path,
    )
