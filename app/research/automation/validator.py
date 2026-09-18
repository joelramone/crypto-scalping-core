"""Preflight validation for one-shot research baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.research.optimizer.grid_search import expand_parameter_grid, load_grid_search_config


@dataclass(frozen=True)
class ValidationResult:
    config_path: Path
    strategy: str
    timeframe: str
    output_path: Path
    combinations: int


def validate_baseline(config_path: str | Path) -> ValidationResult:
    """Require an existing config with exactly one frozen parameter combination."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Research config not found: {path}")

    config = load_grid_search_config(path)
    combinations = list(expand_parameter_grid(config))
    if len(combinations) != 1:
        raise ValueError(
            f"Automation only accepts frozen baselines; found {len(combinations)} combinations"
        )

    data_path = Path(config.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Authorized dataset not found: {data_path}")

    return ValidationResult(
        config_path=path,
        strategy=config.strategy,
        timeframe=config.timeframe,
        output_path=Path(config.output),
        combinations=1,
    )
