"""Compact deterministic diagnostics for completed baseline leaderboards."""

from __future__ import annotations

from math import isfinite, isnan
from pathlib import Path
from typing import Any

import pandas as pd


def diagnose_leaderboard(path: str | Path) -> dict[str, Any]:
    """Read the single best row and expose the metrics used for research triage."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Leaderboard is empty: {path}") from exc
    if frame.empty:
        raise ValueError(f"Leaderboard is empty: {path}")

    required = (
        "strategy", "timeframe", "total_trades", "gross_pnl_before_fees",
        "total_fees", "net_pnl", "gross_profit_factor", "net_profit_factor",
        "gross_expectancy", "net_expectancy", "suppression_rate",
        "profitable_month_percentage",
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Leaderboard {path} missing required columns: {', '.join(missing)}")

    # Preserve the existing first-row convention; never optimize or rerank rows.
    row = frame.iloc[0].copy()
    for field in ("strategy", "timeframe"):
        if pd.isna(row[field]) or not str(row[field]).strip():
            raise ValueError(f"Leaderboard {path} has invalid {field}")
    for field in required[2:]:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Leaderboard {path} has invalid {field}") from exc
        # Positive infinity is a legitimate PF when there are no losing trades.
        valid = not isnan(value) and (isfinite(value) or (
            field in {"gross_profit_factor", "net_profit_factor"} and value > 0
        ))
        if field in {"total_trades", "total_fees", "gross_profit_factor", "net_profit_factor"}:
            valid = valid and value >= 0
        if field == "total_trades":
            valid = valid and isfinite(value) and value.is_integer()
        if field in {"suppression_rate", "profitable_month_percentage"}:
            valid = valid and 0 <= value <= 1
        if not valid:
            raise ValueError(f"Leaderboard {path} has invalid {field}")
        row[field] = value

    gross = row["gross_pnl_before_fees"]
    fees = row["total_fees"]
    net = row["net_pnl"]

    if gross <= 0:
        failure_mode = "NO_GROSS_EDGE"
    elif net <= 0:
        failure_mode = "FEE_DOMINATED"
    else:
        failure_mode = "POSITIVE_NET_EDGE"

    return {
        "source": str(Path(path)),
        "strategy": row["strategy"],
        "timeframe": row["timeframe"],
        "trades": int(row["total_trades"]),
        "gross_pnl_before_fees": gross,
        "total_fees": fees,
        "net_pnl": net,
        "gross_profit_factor": float(row["gross_profit_factor"]),
        "net_profit_factor": float(row["net_profit_factor"]),
        "gross_expectancy": float(row["gross_expectancy"]),
        "net_expectancy": float(row["net_expectancy"]),
        "suppression_rate": float(row["suppression_rate"]),
        "profitable_month_percentage": float(row["profitable_month_percentage"]),
        "failure_mode": failure_mode,
    }
