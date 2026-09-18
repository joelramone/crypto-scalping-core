"""Compact deterministic diagnostics for completed baseline leaderboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def diagnose_leaderboard(path: str | Path) -> dict[str, Any]:
    """Read the single best row and expose the metrics used for research triage."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Leaderboard is empty: {path}")

    row = frame.iloc[0]
    gross = float(row["gross_pnl_before_fees"])
    fees = float(row["total_fees"])
    net = float(row["net_pnl"])

    if gross <= 0:
        failure_mode = "NO_GROSS_EDGE"
    elif net <= 0:
        failure_mode = "FEE_DOMINATED"
    else:
        failure_mode = "POSITIVE_NET_EDGE"

    return {
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
