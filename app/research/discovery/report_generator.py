"""CLI entry point for generating the Quant Research Lab discovery report."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import pandas as pd

from app.research.discovery.experiment_analyzer import (
    DiscoverySummary,
    analyze_experiments,
    select_best_experiments,
)
from app.research.discovery.leaderboard_loader import ROOT_DIR, load_all_leaderboards
from app.research.discovery.parameter_importance import (
    ParameterImportanceRow,
    compute_parameter_importance,
)
from app.research.discovery.pattern_mining import (
    PatternDiscovery,
    discover_top_profit_factor_patterns,
)
from app.research.discovery.recommendations import generate_recommendations

REPORT_PATH = ROOT_DIR / "research" / "reports" / "discovery_report.md"


def _format_number(value: object, decimals: int = 4) -> str:
    """Format numeric values for markdown output."""
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.{decimals}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a simple markdown table."""
    if not rows:
        rows = [["n/a" for _ in headers]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _build_executive_summary(summary: DiscoverySummary) -> str:
    """Return the Executive Summary section."""
    return "\n".join(
        [
            "## Executive Summary",
            "",
            f"- Total experiments: {summary.total_experiments}",
            f"- Strategies tested: {', '.join(summary.strategies_tested)}",
            f"- Best strategy by average PF: {summary.best_strategy}",
            f"- Best timeframe by average PF: {summary.best_timeframe}",
            f"- Average Profit Factor: {_format_number(summary.average_profit_factor)}",
            f"- Average Expectancy: {_format_number(summary.average_expectancy)}",
            f"- Average Drawdown: {_format_number(summary.average_drawdown)}",
        ]
    )


def _build_best_experiments_section(best_df: pd.DataFrame) -> str:
    """Return the Best Experiments section."""
    columns = [
        "strategy",
        "timeframe",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "total_trades",
        "source_file",
    ]
    rows: list[list[object]] = []
    for _, row in best_df.iterrows():
        rows.append(
            [
                row.get("strategy", "n/a"),
                row.get("timeframe", "n/a"),
                _format_number(row.get("profit_factor")),
                _format_number(row.get("expectancy")),
                _format_number(row.get("max_drawdown")),
                _format_number(row.get("total_trades"), decimals=0),
                row.get("source_file", "n/a"),
            ]
        )

    return "\n".join(
        [
            "## Best Experiments",
            "",
            _markdown_table(columns, rows),
        ]
    )


def _build_parameter_importance_section(rows: list[ParameterImportanceRow]) -> str:
    """Return the Parameter Importance section."""
    table_rows = [
        [
            row.parameter,
            _format_number(row.profit_factor_correlation),
            _format_number(row.expectancy_correlation),
            _format_number(row.importance_score),
        ]
        for row in rows
    ]
    return "\n".join(
        [
            "## Parameter Importance",
            "",
            _markdown_table(
                [
                    "parameter",
                    "corr_profit_factor",
                    "corr_expectancy",
                    "importance_score",
                ],
                table_rows,
            ),
        ]
    )


def _build_pattern_discovery_section(patterns: PatternDiscovery) -> str:
    """Return the Pattern Discovery section."""
    lines = [
        "## Pattern Discovery",
        "",
        f"- Top profit-factor threshold: {_format_number(patterns.profit_factor_threshold)}",
        f"- Experiments in top bucket: {patterns.top_bucket_size}",
    ]
    for key, value in patterns.common_values.items():
        lines.append(f"- Most common {key}: {_format_number(value)}")
    return "\n".join(lines)


def _build_recommendations_section(recommendations: list[str]) -> str:
    """Return the Recommendations section."""
    lines = ["## Recommendations", ""]
    if not recommendations:
        lines.append("- No recommendations available.")
    else:
        lines.extend(f"- {recommendation}" for recommendation in recommendations)
    return "\n".join(lines)


def generate_report_markdown(
    summary: DiscoverySummary,
    best_df: pd.DataFrame,
    parameter_rows: list[ParameterImportanceRow],
    patterns: PatternDiscovery,
    recommendations: list[str],
) -> str:
    """Assemble the final discovery report markdown."""
    sections = [
        "# Quant Research Lab Discovery Report",
        "",
        _build_executive_summary(summary),
        "",
        _build_best_experiments_section(best_df),
        "",
        _build_parameter_importance_section(parameter_rows),
        "",
        _build_pattern_discovery_section(patterns),
        "",
        _build_recommendations_section(recommendations),
        "",
    ]
    return "\n".join(sections)


def write_discovery_report(report_path: str | Path = REPORT_PATH) -> Path:
    """Load experiments, generate the discovery report, and write it to disk."""
    all_experiments = load_all_leaderboards()
    summary = analyze_experiments(all_experiments)
    best_df = select_best_experiments(all_experiments, limit=10)
    parameter_rows = compute_parameter_importance(all_experiments)
    patterns = discover_top_profit_factor_patterns(all_experiments)
    recommendations = generate_recommendations(summary, parameter_rows, patterns)

    markdown = generate_report_markdown(
        summary=summary,
        best_df=best_df,
        parameter_rows=parameter_rows,
        patterns=patterns,
        recommendations=recommendations,
    )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def main() -> None:
    """Generate the discovery report from available leaderboard CSVs."""
    report_path = write_discovery_report()
    print(f"Discovery report written to: {report_path}")


if __name__ == "__main__":
    main()
