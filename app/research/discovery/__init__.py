"""Discovery engine for learning from completed research experiments."""

from app.research.discovery.experiment_analyzer import (
    DiscoverySummary,
    analyze_experiments,
)
from app.research.discovery.leaderboard_loader import (
    load_all_leaderboards,
    load_leaderboard_csv,
)
from app.research.discovery.parameter_importance import (
    ParameterImportanceRow,
    compute_parameter_importance,
)
from app.research.discovery.pattern_mining import (
    PatternDiscovery,
    discover_top_profit_factor_patterns,
)
from app.research.discovery.recommendations import generate_recommendations

__all__ = [
    "DiscoverySummary",
    "ParameterImportanceRow",
    "PatternDiscovery",
    "analyze_experiments",
    "compute_parameter_importance",
    "discover_top_profit_factor_patterns",
    "generate_recommendations",
    "load_all_leaderboards",
    "load_leaderboard_csv",
]
