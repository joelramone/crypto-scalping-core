"""Markdown journal rendering for completed experiments."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.research.optimizer.grid_search import GridSearchConfig, GridSearchSummary


def _format_value(value: Any) -> str:
    """Format scalars for journal output."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_parameter_grid(parameter_grid: dict[str, list[Any]]) -> str:
    """Render the explored parameter grid as markdown bullets."""
    lines: list[str] = []
    for name, values in parameter_grid.items():
        rendered = ", ".join(_format_value(value) for value in values)
        lines.append(f"- {name}: {rendered}")
    return "\n".join(lines) if lines else "- n/a"


def _format_best_configuration(summary: GridSearchSummary) -> str:
    """Render the best configuration section."""
    if not summary.ranked_results:
        return "No configurations passed the experiment filters."

    best_result = summary.ranked_results[0]
    lines = [
        f"- Profit Factor: {best_result.metrics.profit_factor:.4f}",
        f"- Expectancy: {best_result.metrics.expectancy:.4f}",
        f"- Max Drawdown: {best_result.metrics.max_drawdown:.4f}",
        f"- Total Trades: {best_result.metrics.total_trades}",
        f"- Parameters: `{json.dumps(best_result.parameters, sort_keys=True)}`",
    ]
    return "\n".join(lines)


def _best_metrics_lines(summary: GridSearchSummary) -> list[str]:
    """Render best-result metric lines for journals."""
    if not summary.ranked_results:
        return [
            "- Best PF: n/a",
            "- Best Expectancy: n/a",
            "- Best Max Drawdown: n/a",
            "- Best Total Trades: n/a",
            "- Best Configuration: `{}`",
        ]

    best_result = summary.ranked_results[0]
    return [
        f"- Best PF: {best_result.metrics.profit_factor:.4f}",
        f"- Best Expectancy: {best_result.metrics.expectancy:.4f}",
        f"- Best Max Drawdown: {best_result.metrics.max_drawdown:.4f}",
        f"- Best Total Trades: {best_result.metrics.total_trades}",
        f"- Best Configuration: `{json.dumps(best_result.parameters, sort_keys=True)}`",
    ]


def _baseline_metrics_lines(summary: GridSearchSummary) -> list[str]:
    """Render neutral metric labels for a single frozen baseline."""
    return [line.replace("Best ", "Observed ") for line in _best_metrics_lines(summary)]


def _is_frozen_baseline(config: GridSearchConfig) -> bool:
    """Return whether the preregistration contains exactly one combination."""
    return config.preregistered and all(len(values) == 1 for values in config.parameters.values())


def _build_objective(config: GridSearchConfig) -> str:
    """Create a simple experiment objective from config."""
    if _is_frozen_baseline(config):
        return (
            f"Evaluate the one frozen, preregistered baseline configuration for "
            f"{config.hypothesis_id} on {config.timeframe} candles."
        )
    return (
        f"Evaluate {config.strategy} on {config.timeframe} candles using the optimizer "
        f"to identify stronger parameter combinations on {config.data}."
    )


def _build_hypothesis(config: GridSearchConfig) -> str:
    """Create a simple experiment hypothesis from config."""
    if _is_frozen_baseline(config):
        return (
            "This experiment evaluates one frozen baseline configuration rather than "
            "a parameter sweep."
        )
    parameter_count = sum(len(values) for values in config.parameters.values())
    return (
        f"A structured sweep across {parameter_count} parameter values may uncover "
        f"{config.strategy} settings with stronger profit factor and expectancy at "
        f"the {config.timeframe} timeframe."
    )


def _build_results(summary: GridSearchSummary) -> str:
    """Render headline experiment results."""
    passing_results = len(summary.ranked_results)
    lines = [
        f"- Evaluated configurations: {summary.evaluated_configurations}",
        f"- Passing configurations: {passing_results}",
    ]
    if summary.ranked_results:
        best_result = summary.ranked_results[0]
        lines.extend(
            [
                f"- Best Profit Factor: {best_result.metrics.profit_factor:.4f}",
                f"- Best Expectancy: {best_result.metrics.expectancy:.4f}",
            ]
        )
    else:
        lines.append("- Best result: no configuration passed the minimum trade filter")
    return "\n".join(lines)


def _is_no_rescue_closure(
    config: GridSearchConfig,
    verdict: str | None = None,
    final_status: str | None = None,
) -> bool:
    """Return whether the observed outcome closes further tuning."""
    rejected = final_status == "CLOSED_REJECTED" or verdict == "BASELINE_REJECT"
    return config.preregistered and config.anti_tuning and rejected


def _closure_recommendation() -> str:
    """Return the deterministic recommendation for an anti-tuning closure."""
    return (
        "- Experiment closed under preregistered anti-tuning rules. No rescue or "
        "parameter optimization is authorized."
    )


def _build_lessons_learned(
    config: GridSearchConfig,
    summary: GridSearchSummary,
    verdict: str | None,
    final_status: str | None,
) -> str:
    """Render a lightweight lessons-learned section."""
    if _is_no_rescue_closure(config, verdict, final_status):
        return _closure_recommendation()
    if _is_frozen_baseline(config):
        return "- The preregistered baseline configuration produced the observed metrics above."
    if not summary.ranked_results:
        return (
            "- The current search space did not produce any configurations that met the "
            "minimum trade threshold.\n"
            "- The next iteration should either relax the trade filter or widen the grid."
        )

    best_result = summary.ranked_results[0]
    return "\n".join(
        [
            f"- The strongest configuration achieved PF={best_result.metrics.profit_factor:.4f} "
            f"and expectancy={best_result.metrics.expectancy:.4f}.",
            "- The current grid is now documented as reusable research knowledge for follow-up runs.",
        ]
    )


def _build_next_experiments(
    config: GridSearchConfig,
    summary: GridSearchSummary,
    verdict: str | None,
    final_status: str | None,
) -> str:
    """Suggest the next research steps in simple heuristic form."""
    if _is_no_rescue_closure(config, verdict, final_status):
        return _closure_recommendation()
    if _is_frozen_baseline(config):
        return "- Follow-up is governed by the preregistered outcome rules, not by a parameter sweep."
    if not summary.ranked_results:
        return (
            f"- Re-run {config.strategy} on {config.timeframe} with a wider parameter grid.\n"
            "- Compare the same strategy on an adjacent timeframe such as 1m or 15m."
        )

    best_result = summary.ranked_results[0]
    return "\n".join(
        [
            f"- Center the next grid around the current best parameters: "
            f"`{json.dumps(best_result.parameters, sort_keys=True)}`.",
            f"- Re-test {config.strategy} on another timeframe to compare against {config.timeframe}.",
        ]
    )


def build_experiment_journal(
    experiment_id: str,
    created_at_utc: str,
    config: GridSearchConfig,
    summary: GridSearchSummary,
    source: str = "optimizer run",
    optimizer_rerun: bool = True,
    verdict: str | None = None,
    final_status: str | None = None,
    failure_classification: str | None = None,
) -> str:
    """Return the markdown journal for a completed optimizer experiment."""
    # Explicit outcome values are produced after metrics evaluation. Config fallbacks
    # retain compatibility with already-finalized historical records.
    verdict = verdict if verdict is not None else config.verdict
    final_status = final_status if final_status is not None else config.final_status
    failure_classification = (
        failure_classification
        if failure_classification is not None
        else config.failure_classification
    )
    closed = _is_no_rescue_closure(config, verdict, final_status)
    sections = [
        "# Experiment",
        "",
        f"- Experiment ID: {experiment_id}",
        f"- Date: {created_at_utc}",
        f"- Strategy: {config.strategy}",
        f"- Timeframe: {config.timeframe}",
        f"- Dataset: {config.data}",
        f"- Config File: {getattr(config, 'config_file', '') or ''}",
        f"- Leaderboard File: {config.output}",
        f"- Source: {source}",
        f"- Optimizer was not rerun: {'yes' if not optimizer_rerun else 'no'}",
        f"- Hypothesis ID: {config.hypothesis_id or ''}",
        f"- Preregistered: {'yes' if config.preregistered else 'no'}",
        f"- Verdict: {verdict or ''}",
        f"- Final Status: {final_status or ''}",
        f"- Failure Classification: {failure_classification or ''}",
        "",
        "## Objective",
        _build_objective(config),
        "",
        "## Hypothesis",
        _build_hypothesis(config),
        "",
        "## Dataset",
        str(config.data),
        "",
        "## Strategy",
        config.strategy,
        "",
        "## Timeframe",
        config.timeframe,
        "",
        "## Frozen Parameters" if _is_frozen_baseline(config) else "## Parameter Grid",
        _format_parameter_grid(config.parameters),
        "",
        "## Results",
        _build_results(summary),
        "",
        "## Run Metadata",
        f"- Total Configurations: {summary.evaluated_configurations}",
        f"- Eligible Configurations: {len(summary.ranked_results)}",
        "",
        "## Baseline Configuration" if _is_frozen_baseline(config) else "## Best Configuration",
        "\n".join(
            _baseline_metrics_lines(summary)
            if _is_frozen_baseline(config)
            else _best_metrics_lines(summary)
        ),
        "",
        "## Lessons Learned",
        _build_lessons_learned(config, summary, verdict, final_status),
        "",
        "## Deterministic Interpretation",
        (
            "This rejected preregistered baseline is closed and cannot be tuned or rescued."
            if closed
            else "The preregistered baseline configuration produced the observed metrics above."
            if _is_frozen_baseline(config)
            else "The strongest observed configuration is treated as the current boundary of known performance within this exact experiment grid."
            if summary.ranked_results
            else "This experiment established that the current grid did not clear the minimum trade boundary."
        ),
        "",
        "## Deterministic Boundary Recommendations",
        _build_next_experiments(config, summary, verdict, final_status),
        "",
        "## Recommended Next Experiments",
        _build_next_experiments(config, summary, verdict, final_status),
        "",
    ]
    return "\n".join(sections)
