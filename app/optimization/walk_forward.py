from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from itertools import product
from statistics import mean
from typing import Dict, List, Tuple

from app.main import _load_rsi_strategy_config_class, _sideways_market_generator, run_single_backtest
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy

# Debug-friendly defaults (intentionally small to provide fast feedback in console)
WINDOWS = 3
TICKS_PER_BACKTEST = 300
TRAINING_RUNS_PER_COMBINATION = 3
VALIDATION_RUNS_PER_WINDOW = 2
MAX_PARAMETER_COMBINATIONS = 12


def _parameter_grid() -> List[Tuple[float, float, int, int]]:
    """Small grid by default for quick debugging and visible progress."""
    atr_sl_multipliers = [1.0, 1.2]
    atr_tp_multipliers = [1.5, 2.0]
    rsi_oversold_values = [28, 30, 32]
    rsi_overbought_values = [68, 70]

    combinations = list(
        product(
            atr_sl_multipliers,
            atr_tp_multipliers,
            rsi_oversold_values,
            rsi_overbought_values,
        )
    )
    return combinations[:MAX_PARAMETER_COMBINATIONS]


def _summarize_runs(run_results: List[Dict[str, float]]) -> Dict[str, float]:
    total_gross_profit = sum(result["gross_profit"] for result in run_results)
    total_gross_loss = sum(result["gross_loss"] for result in run_results)
    total_net_profit = sum(result["net_profit"] for result in run_results)
    total_trades = sum(result["total_trades"] for result in run_results)
    total_wins = sum(result["total_wins"] for result in run_results)

    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
    expectancy_per_trade = total_net_profit / total_trades if total_trades > 0 else 0.0
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0.0

    return {
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy_per_trade,
        "net_profit": total_net_profit,
        "win_rate": win_rate,
        "total_trades": float(total_trades),
    }


def run_walk_forward() -> List[Dict[str, float]]:
    start_ts = time.time()
    print("Starting walk-forward optimization...", flush=True)

    RSIStrategyConfig = _load_rsi_strategy_config_class()
    param_combinations = _parameter_grid()
    total_iterations = WINDOWS * len(param_combinations)

    print(
        f"Configured windows={WINDOWS}, combinations/window={len(param_combinations)}, "
        f"total optimization iterations={total_iterations}",
        flush=True,
    )

    if not param_combinations:
        print("No parameter combinations configured. Exiting.", flush=True)
        return []

    final_results: List[Dict[str, float]] = []
    global_iteration = 0

    for window_idx in range(WINDOWS):
        window_start = time.time()
        print(f"\nProcessing window {window_idx + 1}/{WINDOWS}", flush=True)

        ranking: List[Dict[str, float]] = []

        for combo_idx, (atr_sl, atr_tp, rsi_oversold, rsi_overbought) in enumerate(param_combinations, start=1):
            global_iteration += 1
            print(
                f"[Iteration {global_iteration}/{total_iterations}] "
                f"Window {window_idx + 1} - Combo {combo_idx}/{len(param_combinations)} -> "
                f"SL={atr_sl}, TP={atr_tp}, OS={rsi_oversold}, OB={rsi_overbought}",
                flush=True,
            )

            config = RSIStrategyConfig(
                atr_sl_multiplier=atr_sl,
                atr_tp_multiplier=atr_tp,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
            )

            training_runs: List[Dict[str, float]] = []
            for run_idx in range(TRAINING_RUNS_PER_COMBINATION):
                strategy = RSIMeanReversionStrategy(config=config)
                training_runs.append(
                    run_single_backtest(
                        market_generator_factory=_sideways_market_generator,
                        strategy=strategy,
                        ticks=TICKS_PER_BACKTEST,
                    )
                )
                print(
                    f"  Training run {run_idx + 1}/{TRAINING_RUNS_PER_COMBINATION} complete.",
                    flush=True,
                )

            summary = _summarize_runs(training_runs)
            ranking.append(
                {
                    "window": float(window_idx + 1),
                    "atr_sl_multiplier": atr_sl,
                    "atr_tp_multiplier": atr_tp,
                    "rsi_oversold": float(rsi_oversold),
                    "rsi_overbought": float(rsi_overbought),
                    **summary,
                }
            )

            print(
                "  Training metrics -> "
                f"PF={summary['profit_factor']:.3f}, "
                f"Expectancy={summary['expectancy_per_trade']:.4f}, "
                f"NetProfit={summary['net_profit']:.4f}, "
                f"WinRate={summary['win_rate']:.2f}%",
                flush=True,
            )

        ranked_window = sorted(
            ranking,
            key=lambda item: (item["profit_factor"], item["expectancy_per_trade"], item["net_profit"]),
            reverse=True,
        )
        best_training = ranked_window[0]

        best_config = RSIStrategyConfig(
            atr_sl_multiplier=best_training["atr_sl_multiplier"],
            atr_tp_multiplier=best_training["atr_tp_multiplier"],
            rsi_oversold=int(best_training["rsi_oversold"]),
            rsi_overbought=int(best_training["rsi_overbought"]),
        )

        print(
            "Best training config selected for validation -> "
            f"{asdict(best_config)}",
            flush=True,
        )

        validation_runs: List[Dict[str, float]] = []
        for validation_idx in range(VALIDATION_RUNS_PER_WINDOW):
            strategy = RSIMeanReversionStrategy(config=best_config)
            validation_runs.append(
                run_single_backtest(
                    market_generator_factory=_sideways_market_generator,
                    strategy=strategy,
                    ticks=TICKS_PER_BACKTEST,
                )
            )
            print(
                f"  Validation run {validation_idx + 1}/{VALIDATION_RUNS_PER_WINDOW} complete.",
                flush=True,
            )

        validation_summary = _summarize_runs(validation_runs)
        window_elapsed = time.time() - window_start

        result = {
            "window": float(window_idx + 1),
            "atr_sl_multiplier": best_training["atr_sl_multiplier"],
            "atr_tp_multiplier": best_training["atr_tp_multiplier"],
            "rsi_oversold": best_training["rsi_oversold"],
            "rsi_overbought": best_training["rsi_overbought"],
            **validation_summary,
            "window_duration_sec": window_elapsed,
        }
        final_results.append(result)

        print(
            "Window validation summary -> "
            f"PF={validation_summary['profit_factor']:.3f}, "
            f"Expectancy={validation_summary['expectancy_per_trade']:.4f}, "
            f"NetProfit={validation_summary['net_profit']:.4f}, "
            f"WinRate={validation_summary['win_rate']:.2f}%",
            flush=True,
        )
        print(f"Window {window_idx + 1} duration: {window_elapsed:.2f}s", flush=True)

    total_elapsed = time.time() - start_ts
    avg_pf = mean(item["profit_factor"] for item in final_results) if final_results else 0.0
    avg_expectancy = mean(item["expectancy_per_trade"] for item in final_results) if final_results else 0.0
    total_net = sum(item["net_profit"] for item in final_results)
    avg_win_rate = mean(item["win_rate"] for item in final_results) if final_results else 0.0

    print("\nFinal walk-forward summary", flush=True)
    print(f"  Windows processed: {len(final_results)}", flush=True)
    print(f"  Avg PF: {avg_pf:.3f}", flush=True)
    print(f"  Avg Expectancy: {avg_expectancy:.4f}", flush=True)
    print(f"  Total Net Profit: {total_net:.4f}", flush=True)
    print(f"  Avg Win Rate: {avg_win_rate:.2f}%", flush=True)
    print(f"Total duration: {total_elapsed:.2f}s", flush=True)

    return final_results


def main() -> None:
    print("Entering walk-forward main()", flush=True)
    overall_start = time.time()
    try:
        _ = run_walk_forward()
    except Exception as exc:
        print(f"ERROR in walk-forward optimization: {exc}", flush=True)
        traceback.print_exc()
    finally:
        print(f"walk_forward.py finished in {time.time() - overall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
