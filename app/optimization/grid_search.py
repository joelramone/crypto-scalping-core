from __future__ import annotations

import os
import time
from itertools import product
from statistics import mean
from typing import Dict, List

from app.main import _load_rsi_strategy_config_class, _sideways_market_generator, run_single_backtest
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from app.utils.logger import configure_logging, get_logger

COMBINATION_WARNING_THRESHOLD = 1000
HEARTBEAT_SECONDS = 5.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def run_grid_search() -> List[Dict[str, float]]:
    configure_logging()
    logger = get_logger(__name__)
    start_time = time.time()

    atr_sl_multipliers = [0.8, 1.0, 1.2, 1.5]
    atr_tp_multipliers = [1.5, 2.0, 2.5, 3.0]
    rsi_oversold_values = [25, 30, 35]
    rsi_overbought_values = [65, 70, 75]

    full_combinations = list(
        product(atr_sl_multipliers, atr_tp_multipliers, rsi_oversold_values, rsi_overbought_values)
    )
    full_count = len(full_combinations)
    max_combinations = _env_int("MAX_GRID_COMBINATIONS", full_count)
    combinations = full_combinations[:max_combinations]

    if full_count > COMBINATION_WARNING_THRESHOLD:
        logger.warning("Search space is very large: combinations=%s (> %s)", full_count, COMBINATION_WARNING_THRESHOLD)

    if len(combinations) < full_count:
        logger.warning(
            "Auto-reducing search space from %s to %s via MAX_GRID_COMBINATIONS=%s",
            full_count,
            len(combinations),
            max_combinations,
        )

    simulations_per_combination = _env_int("GRID_SIMULATIONS_PER_COMBINATION", 100)
    max_total_runs = _env_int("MAX_GRID_TOTAL_RUNS", len(combinations) * simulations_per_combination)

    logger.info(
        "Starting grid search combinations=%s simulations_per_combination=%s max_total_runs=%s",
        len(combinations),
        simulations_per_combination,
        max_total_runs,
    )
    print(f"Grid search started: {len(combinations)} combinations", flush=True)

    results: List[Dict[str, float]] = []
    RSIStrategyConfig = _load_rsi_strategy_config_class()
    total_runs_done = 0
    last_output_ts = time.time()

    for combo_index, (atr_sl_multiplier, atr_tp_multiplier, rsi_oversold, rsi_overbought) in enumerate(
        combinations, start=1
    ):
        if total_runs_done >= max_total_runs:
            logger.warning("Reached MAX_GRID_TOTAL_RUNS=%s. Stopping early to prevent runaway loops.", max_total_runs)
            break

        logger.info(
            "Evaluating combination %s/%s strategy=RSI params=%s",
            combo_index,
            len(combinations),
            {
                "atr_sl_multiplier": atr_sl_multiplier,
                "atr_tp_multiplier": atr_tp_multiplier,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            },
        )
        print(f"Evaluating combination {combo_index}/{len(combinations)}", flush=True)

        config = RSIStrategyConfig(
            atr_sl_multiplier=atr_sl_multiplier,
            atr_tp_multiplier=atr_tp_multiplier,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
        )

        run_results: List[Dict[str, float]] = []
        for simulation_index in range(1, simulations_per_combination + 1):
            if total_runs_done >= max_total_runs:
                logger.warning("Reached MAX_GRID_TOTAL_RUNS=%s during combination loop.", max_total_runs)
                break

            strategy = RSIMeanReversionStrategy(config=config)
            run_results.append(
                run_single_backtest(
                    market_generator_factory=_sideways_market_generator,
                    strategy=strategy,
                )
            )
            total_runs_done += 1

            if simulation_index == 1 or simulation_index % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "Combination progress combo=%s/%s run=%s/%s elapsed=%.2fs",
                    combo_index,
                    len(combinations),
                    simulation_index,
                    simulations_per_combination,
                    elapsed,
                )
                print(
                    f"  run {simulation_index}/{simulations_per_combination} | elapsed={elapsed:.2f}s",
                    flush=True,
                )
                last_output_ts = time.time()

            if time.time() - last_output_ts > HEARTBEAT_SECONDS:
                heartbeat_elapsed = time.time() - start_time
                print(f"  heartbeat... elapsed={heartbeat_elapsed:.2f}s", flush=True)
                logger.info("Heartbeat combo=%s elapsed=%.2fs", combo_index, heartbeat_elapsed)
                last_output_ts = time.time()

        if not run_results:
            continue

        total_gross_profit = sum(result["gross_profit"] for result in run_results)
        total_gross_loss = sum(result["gross_loss"] for result in run_results)
        total_net_profit = sum(result["net_profit"] for result in run_results)
        total_trades = sum(result["total_trades"] for result in run_results)
        total_wins = sum(result["total_wins"] for result in run_results)

        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
        expectancy_per_trade = total_net_profit / total_trades if total_trades > 0 else 0.0
        avg_max_drawdown = mean(result["max_drawdown"] for result in run_results)
        win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0.0

        results.append(
            {
                "atr_sl_multiplier": atr_sl_multiplier,
                "atr_tp_multiplier": atr_tp_multiplier,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
                "profit_factor": profit_factor,
                "expectancy_per_trade": expectancy_per_trade,
                "avg_max_drawdown": avg_max_drawdown,
                "win_rate": win_rate,
            }
        )

    ranked_results = sorted(
        results,
        key=lambda item: (item["profit_factor"], item["expectancy_per_trade"]),
        reverse=True,
    )

    logger.info("Grid search finished combinations_evaluated=%s elapsed=%.2fs", len(results), time.time() - start_time)
    print("Top 5 configurations (SIDEWAYS):\n", flush=True)
    for result in ranked_results[:5]:
        print(
            "SL={sl} TP={tp} OS={os} OB={ob}".format(
                sl=result["atr_sl_multiplier"],
                tp=result["atr_tp_multiplier"],
                os=result["rsi_oversold"],
                ob=result["rsi_overbought"],
            ),
            flush=True,
        )
        print(f"Profit Factor: {result['profit_factor']:.2f}", flush=True)
        print(f"Expectancy: {result['expectancy_per_trade']:.4f}", flush=True)
        print(f"Drawdown: {result['avg_max_drawdown']:.2f}", flush=True)
        print(f"Win Rate: {result['win_rate']:.0f}%", flush=True)
        print(flush=True)

    return ranked_results
