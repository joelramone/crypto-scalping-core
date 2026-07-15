"""Generate lightweight heuristic recommendations from discovery outputs."""

from __future__ import annotations

from app.research.discovery.experiment_analyzer import DiscoverySummary
from app.research.discovery.parameter_importance import ParameterImportanceRow
from app.research.discovery.pattern_mining import PatternDiscovery


def generate_recommendations(
    summary: DiscoverySummary,
    parameter_rows: list[ParameterImportanceRow],
    patterns: PatternDiscovery,
) -> list[str]:
    """Create simple next-step recommendations from the discovery analysis."""
    recommendations: list[str] = []

    recommendations.append(
        f"Prioritize {summary.best_timeframe} experiments first because that timeframe "
        "currently has the strongest average profit factor."
    )
    recommendations.append(
        f"Start new sweeps from the strongest strategy family so far: {summary.best_strategy}."
    )

    tp = patterns.common_values.get("TP", "n/a")
    sl = patterns.common_values.get("SL", "n/a")
    holding = patterns.common_values.get("holding", "n/a")
    if tp != "n/a" or sl != "n/a" or holding != "n/a":
        recommendations.append(
            f"Bias new grids toward TP={tp}, SL={sl}, and holding={holding}, since those "
            "values recur most often in the top profit-factor bucket."
        )

    rsi = patterns.common_values.get("RSI", "n/a")
    volume = patterns.common_values.get("volume", "n/a")
    lookback = patterns.common_values.get("lookback", "n/a")
    if rsi != "n/a" or volume != "n/a" or lookback != "n/a":
        recommendations.append(
            f"Keep RSI={rsi}, volume ratio={volume}, and lookback={lookback} near the "
            "current winning cluster before exploring wider ranges."
        )

    if parameter_rows:
        top_parameter = parameter_rows[0]
        recommendations.append(
            f"Focus tuning attention on {top_parameter.parameter}; it shows the strongest "
            "linear relationship with profit factor or expectancy in the current dataset."
        )

    return recommendations
