from __future__ import annotations

from itertools import product
from statistics import mean
from typing import Dict, List

from app.main import (
    _load_rsi_strategy_config_class,
    _sideways_market_generator,
    run_single_backtest,
)
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy


def run_grid_search() -> List[Dict[str, float]]:
    atr_sl_multipliers = [0.8, 1.0, 1.2, 1.5]
    atr_tp_multipliers = [1.5, 2.0, 2.5, 3.0]
    rsi_oversold_values = [25, 30, 35]
    rsi_overbought_values = [65, 70, 75]

    results: List[Dict[str, float]] = []
    RSIStrategyConfig = _load_rsi_strategy_config_class()

    for atr_sl_multiplier, atr_tp_multiplier, rsi_oversold, rsi_overbought in product(
        atr_sl_multipliers,
        atr_tp_multipliers,
        rsi_oversold_values,
        rsi_overbought_values,
    ):
        config = RSIStrategyConfig(
            atr_sl_multiplier=atr_sl_multiplier,
            atr_tp_multiplier=atr_tp_multiplier,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
        )
        _ = RSIMeanReversionStrategy(config=config)

        run_results = [
            run_single_backtest(
                market_generator_factory=_sideways_market_generator,
                strategy_config=config,
            )
            for _ in range(100)
        ]

        total_gross_profit = sum(result["gross_profit"] for result in run_results)
        total_gross_loss = sum(result["gross_loss"] for result in run_results)
        total_net_profit = sum(result["net_profit"] for result in run_results)
        total_trades = sum(result["total_trades"] for result in run_results)
        total_wins = sum(result["total_wins"] for result in run_results)

        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
        expectancy_per_trade = total_net_profit / total_trades if total_trades > 0 else 0.0
        avg_max_drawdown = mean(result["max_drawdown"] for result in run_results) if run_results else 0.0
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

    print("Top 5 configurations (SIDEWAYS):\n")
    for result in ranked_results[:5]:
        print(
            "SL={sl} TP={tp} OS={os} OB={ob}".format(
                sl=result["atr_sl_multiplier"],
                tp=result["atr_tp_multiplier"],
                os=result["rsi_oversold"],
                ob=result["rsi_overbought"],
            )
        )
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Expectancy: {result['expectancy_per_trade']:.4f}")
        print(f"Drawdown: {result['avg_max_drawdown']:.2f}")
        print(f"Win Rate: {result['win_rate']:.0f}%")
        print()

    return ranked_results
