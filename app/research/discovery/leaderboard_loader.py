"""Load and normalize leaderboard CSV data for discovery analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
LEADERBOARD_DIR = ROOT_DIR / "research" / "leaderboards"

CORE_COLUMNS = {
    "strategy",
    "timeframe",
    "rank",
    "total_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "gross_pnl",
    "net_pnl",
    "source_file",
}

NUMERIC_BASE_COLUMNS = {
    "rank",
    "total_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "gross_pnl",
    "net_pnl",
}


def load_leaderboard_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a single leaderboard CSV and normalize legacy columns."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "strategy" not in df.columns:
        raise ValueError(f"Leaderboard is missing required 'strategy' column: {path}")

    if "timeframe" not in df.columns:
        df["timeframe"] = "1m"

    for column in NUMERIC_BASE_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    parameter_columns = [
        column
        for column in df.columns
        if column not in CORE_COLUMNS and column not in {"strategy", "timeframe"}
    ]
    for column in parameter_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["strategy"] = df["strategy"].astype(str)
    df["timeframe"] = df["timeframe"].fillna("1m").astype(str)
    df["source_file"] = path.name
    return df


def load_all_leaderboards(leaderboard_dir: str | Path = LEADERBOARD_DIR) -> pd.DataFrame:
    """Load and merge every leaderboard CSV in the research leaderboard directory."""
    directory = Path(leaderboard_dir)
    csv_paths = sorted(directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No leaderboard CSVs found in: {directory}")

    frames = [load_leaderboard_csv(csv_path) for csv_path in csv_paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def get_numeric_parameter_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric parameter columns, excluding core metric fields."""
    return [
        column
        for column in df.columns
        if column not in CORE_COLUMNS
        and column not in {"strategy", "timeframe"}
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].notna().any()
    ]
