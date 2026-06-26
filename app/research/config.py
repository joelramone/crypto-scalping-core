from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ResearchBacktestConfig(BaseModel):
    """Configuration for the first V2 historical research backtester."""

    data_path: Path
    strategy: Literal["breakout", "rsi"] = "breakout"
    initial_balance: float = Field(default=1_000.0, gt=0)
    trade_size_usdt: float = Field(default=100.0, gt=0)
    fee_rate: float = Field(default=0.0004, ge=0)
    max_holding_bars: int = Field(default=60, ge=1)
    warmup_bars: int = Field(default=220, ge=1)
    timestamp_column: str = "timestamp"

    # TODO: add parameter optimization once baseline metrics are trustworthy.
    # TODO: add out-of-sample testing before promoting any researched setup.
    # TODO: add an ML probability model after enough labeled trade outcomes exist.
    # TODO: add multi-symbol testing when single-symbol flow is stable.
