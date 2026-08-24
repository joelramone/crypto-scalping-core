"""Generate a read-only strategy/timeframe view of existing Research Memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.research.memory.experiment_store import MEMORY_INDEX_PATH

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT_DIR / "research" / "reports" / "research_timeframe_summary.md"


def _value(row: pd.Series, name: str, fallback: str = "") -> object:
    value = row.get(name, fallback)
    return fallback if pd.isna(value) else value


def build_timeframe_summary(index_path: str | Path = MEMORY_INDEX_PATH) -> str:
    """Build Markdown without changing memory or historical leaderboard files."""
    memory = pd.read_csv(index_path, keep_default_na=False)
    records: list[dict[str, object]] = []
    for _, experiment in memory.iterrows():
        leaderboard_path = Path(str(experiment.get("leaderboard_file", "")))
        if not leaderboard_path.is_absolute():
            leaderboard_path = ROOT_DIR / leaderboard_path
        if not leaderboard_path.is_file():
            continue
        leaderboard = pd.read_csv(leaderboard_path)
        if leaderboard.empty:
            continue
        stored_best = leaderboard[leaderboard.get("rank", pd.Series(dtype=float)) == 1]
        row = stored_best.iloc[0] if not stored_best.empty else leaderboard.iloc[0]
        trades = float(_value(row, "total_trades", 0))
        total_fees = _value(row, "total_fees", "")
        records.append({
            "strategy": experiment.get("strategy", _value(row, "strategy")),
            "timeframe": experiment.get("timeframe", _value(row, "timeframe", "1m")),
            "trades": int(trades),
            "gross expectancy": _value(row, "gross_expectancy"),
            "fees/trade": float(total_fees) / trades if total_fees != "" and trades else "",
            "net expectancy": _value(row, "net_expectancy", _value(row, "expectancy")),
            "PF": _value(row, "net_profit_factor", _value(row, "profit_factor")),
            "average holding": _value(row, "average_holding_candles"),
            "validation status": experiment.get("status", ""),
        })
    columns = ["trades", "gross expectancy", "fees/trade", "net expectancy", "PF", "average holding", "validation status"]
    lines = ["# Research Timeframe Summary", "", "Existing stored rankings are preserved; this report does not select new winners.", ""]
    if not records:
        return "\n".join(lines + ["No experiment memory with an available leaderboard was found.", ""])
    frame = pd.DataFrame(records).sort_values(["strategy", "timeframe"], kind="stable")
    for (strategy, timeframe), group in frame.groupby(["strategy", "timeframe"], sort=False):
        lines.extend([
            f"## {strategy} — {timeframe}", "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ])
        for _, row in group.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_timeframe_summary(output: str | Path = DEFAULT_OUTPUT, index_path: str | Path = MEMORY_INDEX_PATH) -> Path:
    """Write the derived Markdown report only."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_timeframe_summary(index_path), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--index", default=str(MEMORY_INDEX_PATH))
    args = parser.parse_args()
    print(write_timeframe_summary(args.output, args.index))


if __name__ == "__main__":
    main()
