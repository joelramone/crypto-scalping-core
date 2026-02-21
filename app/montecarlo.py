from __future__ import annotations

import os
import time
from typing import Any

from app.config import TRADING_CORE_CONFIG
from app.main import (
    DEFAULT_MAX_SIMULATION_MS,
    DEFAULT_MAX_TRADE_TICKS,
    _env_flag,
    _load_breakout_strategy_config_class,
    run_monte_carlo_profiled,
)
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.high_vol_engine import HighVolEngine
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def run_monte_carlo_mvp(
    runs: int = 10_000,
    progress: bool = False,
    max_iterations: int = TRADING_CORE_CONFIG.MONTE_CARLO_MAX_ITER,
    random_seed: int = 42,
) -> dict[str, Any]:
    BreakoutStrategyConfig = _load_breakout_strategy_config_class()
    config = BreakoutStrategyConfig(
        lookback_period=int(os.getenv("BREAKOUT_LOOKBACK", "18")),
        atr_sl_multiplier=float(os.getenv("ATR_SL_MULTIPLIER", "1.3")),
        atr_tp_multiplier=float(os.getenv("ATR_TP_MULTIPLIER", "2.0")),
        breakout_buffer_atr=float(os.getenv("BREAKOUT_SENSITIVITY", "0.12")),
        min_take_profit_r=float(os.getenv("MIN_TP_R", "1.5")),
        trailing_stop_enabled=os.getenv("TRAILING_STOP_ENABLED", "0") == "1",
        trailing_stop_atr_multiplier=float(os.getenv("TRAILING_STOP_ATR_MULTIPLIER", "1.0")),
    )

    strategy = BreakoutTrendStrategy(config=config)
    high_vol_engine = HighVolEngine(breakout_strategy=strategy)

    max_trade_ticks = int(os.getenv("MONTE_CARLO_MAX_TRADE_TICKS", str(DEFAULT_MAX_TRADE_TICKS)))
    max_simulation_ms = int(os.getenv("MONTE_CARLO_MAX_SIMULATION_MS", str(DEFAULT_MAX_SIMULATION_MS)))

    return run_monte_carlo_profiled(
        strategy=high_vol_engine,
        runs=runs,
        progress=progress,
        report_every=100,
        max_trade_ticks=max_trade_ticks,
        max_simulation_ms=max_simulation_ms,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    configure_logging()
    runs = int(os.getenv("MONTE_CARLO_SIMULATIONS", "10000"))
    show_progress = _env_flag("MONTE_CARLO_PROGRESS", True)
    max_iterations = int(os.getenv("MONTE_CARLO_MAX_ITER", str(TRADING_CORE_CONFIG.MONTE_CARLO_MAX_ITER)))
    random_seed = int(os.getenv("MONTE_CARLO_SEED", "42"))

    start = time.perf_counter()
    metrics = run_monte_carlo_mvp(runs=runs, progress=show_progress, max_iterations=max_iterations, random_seed=random_seed)
    elapsed = time.perf_counter() - start

    logger.info("monte_carlo_metrics", extra={"event_name": "monte_carlo_metrics", "parameters": metrics})
    logger.info("monte_carlo_execution", extra={"event_name": "monte_carlo_execution", "duration_s": elapsed})
