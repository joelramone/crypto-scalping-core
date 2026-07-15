"""CLI summary for the Research Memory index."""

from __future__ import annotations

from app.research.memory.knowledge_base import build_memory_summary


def render_memory_summary() -> list[str]:
    """Render the Research Memory summary as printable lines."""
    summary = build_memory_summary()
    return [
        f"Total experiments: {summary.total_experiments}",
        f"Completed experiments: {summary.completed_experiments}",
        f"Strategies tested: {', '.join(summary.strategies_tested) if summary.strategies_tested else 'n/a'}",
        f"Timeframes tested: {', '.join(summary.timeframes_tested) if summary.timeframes_tested else 'n/a'}",
        f"Best experiment ID: {summary.best_experiment_id}",
        f"Best strategy: {summary.best_strategy}",
        f"Best timeframe: {summary.best_timeframe}",
        f"Best Profit Factor: {summary.best_profit_factor:.4f}",
        f"Best Expectancy: {summary.best_expectancy:.4f}",
        f"Average best Profit Factor: {summary.average_best_profit_factor:.4f}",
        f"Average best Expectancy: {summary.average_best_expectancy:.4f}",
    ]


def main() -> None:
    """Print a compact summary of stored experiment memory."""
    for line in render_memory_summary():
        print(line)


if __name__ == "__main__":
    main()
