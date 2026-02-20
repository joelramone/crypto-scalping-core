from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any

from importlib import util as importlib_util
from pathlib import Path

from app.data.real_data_loader import load_real_market_data
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.high_vol_engine import HighVolEngine
from app.trading.execution_metrics import compute_r_multiple
from app.utils.logger import configure_logging, get_logger

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

USE_REAL_DATA = True
DEFAULT_REAL_DATA_CSV = Path("data/binance_klines.csv")
ATR_WINDOW = 14


def _load_breakout_strategy_config_class():
    config_path = Path(__file__).resolve().parent / "config" / "breakout_config.py"
    spec = importlib_util.spec_from_file_location("app.config.breakout_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load BreakoutStrategyConfig")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BreakoutStrategyConfig


@dataclass
class Position:
    side: str
    entry: float
    sl: float
    tp: float
    initial_sl: float
    risk_per_trade: float
    reward_per_trade: float
    opened_tick: int
    regime_score: float
    atr: float
    trailing_stop_enabled: bool
    trailing_stop_atr_multiplier: float


logger = get_logger(__name__)


def _compute_atr(high: list[float], low: list[float], close: list[float], period: int = ATR_WINDOW) -> float | None:
    if len(close) <= period:
        return None

    true_ranges: list[float] = []
    prev_close = close[0]
    for h, l, c in zip(high, low, close):
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        true_ranges.append(tr)
        prev_close = c

    if len(true_ranges) < period:
        return None
    return sum(true_ranges[-period:]) / period


def _high_volatility_market_generator(initial_price: float):
    _ = initial_price
    direction = random.choice((-1.0, 1.0))

    def next_price(price: float) -> float:
        nonlocal direction
        if random.random() < 0.30:
            direction *= -1.0

        shock = random.uniform(0.002, 0.018)
        random_component = random.uniform(-0.012, 0.012)
        return max(100.0, price * (1 + direction * shock + random_component))

    return next_price


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def _finalize_backtest_metrics(
    r_multiples: list[float],
    trade_logs: list[dict[str, float | int | str]],
    equity_curve: list[float],
) -> dict[str, Any]:
    net_profit_r = sum(r_multiples)
    wins = sum(1 for value in r_multiples if value > 0)
    losses = sum(1 for value in r_multiples if value < 0)
    gross_profit = sum(value for value in r_multiples if value > 0)
    gross_loss = abs(sum(value for value in r_multiples if value < 0))
    max_drawdown = _compute_max_drawdown(equity_curve)

    return {
        "trades": len(r_multiples),
        "wins": wins,
        "losses": losses,
        "expectancy": mean(r_multiples) if r_multiples else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else 0.0,
        "net_profit_r": net_profit_r,
        "max_drawdown": max_drawdown,
        "r_distribution": r_multiples,
        "trade_logs": trade_logs,
    }


def _backtest_from_ohlcv_rows(strategy: HighVolEngine, rows: list[dict[str, float]]) -> dict[str, Any]:
    close: list[float] = []
    high: list[float] = []
    low: list[float] = []
    volume: list[float] = []
    atr_series: list[float] = []
    true_ranges: list[float] = []
    rolling_tr_sum = 0.0
    equity_curve: list[float] = [0.0]
    cumulative_r = 0.0
    r_multiples: list[float] = []
    trade_logs: list[dict[str, float | int | str]] = []

    active: Position | None = None

    for tick, row in enumerate(rows, start=1):
        price = float(row["close"])
        candle_high = float(row["high"])
        candle_low = float(row["low"])
        candle_volume = float(row["volume"])

        close.append(price)
        high.append(candle_high)
        low.append(candle_low)
        volume.append(candle_volume)

        if len(close) == 1:
            equity_curve.append(cumulative_r)
            continue

        previous_price = close[-2]
        tr = max(candle_high - candle_low, abs(candle_high - previous_price), abs(candle_low - previous_price))
        true_ranges.append(tr)

        if len(true_ranges) < ATR_WINDOW:
            equity_curve.append(cumulative_r)
            continue

        rolling_tr_sum += tr
        if len(true_ranges) > ATR_WINDOW:
            rolling_tr_sum -= true_ranges[-(ATR_WINDOW + 1)]
        atr = rolling_tr_sum / ATR_WINDOW
        atr_series.append(atr)

        market_data = {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "atr": atr_series,
        }

        signal = strategy.generate_signal(market_data)

        if active:
            if active.trailing_stop_enabled:
                trail_distance = atr * active.trailing_stop_atr_multiplier
                if active.side == "LONG":
                    active.sl = max(active.sl, price - trail_distance)
                else:
                    active.sl = min(active.sl, price + trail_distance)

            hit_sl = (active.side == "LONG" and price <= active.sl) or (active.side == "SHORT" and price >= active.sl)
            hit_tp = (active.side == "LONG" and price >= active.tp) or (active.side == "SHORT" and price <= active.tp)

            if hit_sl or hit_tp:
                exit_reason = "TP" if hit_tp else "SL"
                exit_price = active.tp if hit_tp else active.sl
                r_multiple = compute_r_multiple(
                    side=active.side,
                    entry_price=active.entry,
                    exit_price=exit_price,
                    stop_price=active.initial_sl,
                )

                if exit_reason == "SL" and r_multiple < -1.01:
                    logger.warning("Detected loss below -1.01R without slippage model: r_multiple=%.4f", r_multiple)
                    assert r_multiple >= -1.01, f"Stop-loss exceeded max tolerated loss: {r_multiple:.4f}R"

                r_multiples.append(r_multiple)
                trade_logs.append(
                    {
                        "regime_score": active.regime_score,
                        "atr": active.atr,
                        "entry_price": active.entry,
                        "stop": active.initial_sl,
                        "target": active.tp,
                        "risk_per_trade": active.risk_per_trade,
                        "reward_per_trade": active.reward_per_trade,
                        "r_multiple": r_multiple,
                        "duration_ticks": tick - active.opened_tick,
                        "exit_reason": exit_reason,
                    }
                )
                strategy.record_trade_outcome(trade_logs[-1])
                cumulative_r += r_multiple
                equity_curve.append(cumulative_r)
                active = None

        if active is None and signal:
            context = strategy.consume_last_signal_context() or {}
            entry_price = float(signal["entry"])
            stop_price = float(signal["sl"])
            target_price = float(signal["tp"])
            side = str(signal["side"])
            risk_per_trade = abs(entry_price - stop_price)
            reward_per_trade = abs(target_price - entry_price)

            if risk_per_trade <= 0:
                raise ValueError("risk_per_trade must be greater than zero")

            min_rr = signal.get("min_rr")
            if min_rr is not None:
                target_r = compute_r_multiple(side=side, entry_price=entry_price, exit_price=target_price, stop_price=stop_price)
                if target_r < float(min_rr):
                    logger.warning(
                        "Configured target is below min_rr: target_r=%.4f min_rr=%.4f side=%s",
                        target_r,
                        float(min_rr),
                        side,
                    )

            active = Position(
                side=side,
                entry=entry_price,
                sl=stop_price,
                tp=target_price,
                initial_sl=stop_price,
                risk_per_trade=risk_per_trade,
                reward_per_trade=reward_per_trade,
                opened_tick=tick,
                regime_score=float(context.get("regime_score", 0.0)),
                atr=float(context.get("atr", atr)),
                trailing_stop_enabled=bool(signal.get("trailing_stop_enabled", False)),
                trailing_stop_atr_multiplier=float(signal.get("trailing_stop_atr_multiplier", 1.0)),
            )

        equity_curve.append(cumulative_r)

    return _finalize_backtest_metrics(r_multiples=r_multiples, trade_logs=trade_logs, equity_curve=equity_curve)


def run_single_backtest(
    strategy: HighVolEngine,
    ticks: int = 1500,
    initial_price: float = 50000.0,
    market_df: Any = None,
) -> dict[str, Any]:
    if market_df is not None:
        rows = [
            {
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for row in market_df.itertuples(index=False)
        ]
        return _backtest_from_ohlcv_rows(strategy=strategy, rows=rows)

    price = initial_price
    generator = _high_volatility_market_generator(initial_price)

    rows: list[dict[str, float]] = []
    for _ in range(1, ticks + 1):
        previous_price = price
        price = generator(price)

        candle_high = max(previous_price, price) * (1 + random.uniform(0.0005, 0.003))
        candle_low = min(previous_price, price) * (1 - random.uniform(0.0005, 0.003))
        candle_volume = max(100.0, 1000.0 + random.uniform(-250, 400) + abs(price - previous_price) * 0.4)

        rows.append(
            {
                "open": previous_price,
                "high": candle_high,
                "low": candle_low,
                "close": price,
                "volume": candle_volume,
            }
        )

    return _backtest_from_ohlcv_rows(strategy=strategy, rows=rows)


def run_monte_carlo(strategy: HighVolEngine, runs: int = 10_000) -> dict[str, Any]:
    return run_monte_carlo_profiled(strategy=strategy, runs=runs)


def run_monte_carlo_profiled(
    strategy: HighVolEngine,
    runs: int = 10_000,
    ticks: int = 1500,
    initial_price: float = 50000.0,
    progress: bool = False,
    report_every: int = 100,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be > 0")

    iterator = range(runs)
    if progress and tqdm is not None:
        iterator = tqdm(iterator, total=runs, desc="Monte Carlo")

    results: list[dict[str, Any]] = []
    sim_times_ms: list[float] = []
    block_times_ms: list[float] = []
    block_start = time.perf_counter()

    for idx in iterator:
        strategy.reset()
        sim_start = time.perf_counter()
        results.append(run_single_backtest(strategy=strategy, ticks=ticks, initial_price=initial_price))
        sim_times_ms.append((time.perf_counter() - sim_start) * 1000.0)

        if report_every > 0 and (idx + 1) % report_every == 0:
            block_elapsed_ms = (time.perf_counter() - block_start) * 1000.0
            block_times_ms.append(block_elapsed_ms)
            block_start = time.perf_counter()

    net_profits = [float(r["net_profit_r"]) for r in results]
    trades_per_run = [int(r["trades"]) for r in results]
    all_r = [item for r in results for item in r["r_distribution"]]

    total_wins = sum(r["wins"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    gross_profit = sum(sum(x for x in r["r_distribution"] if x > 0) for r in results)
    gross_loss = abs(sum(sum(x for x in r["r_distribution"] if x < 0) for r in results))

    net_profits_std = pstdev(net_profits) if len(net_profits) > 1 else 0.0
    sharpe = (mean(net_profits) / net_profits_std) if net_profits_std > 0 else 0.0

    avg_sim_time_ms = mean(sim_times_ms) if sim_times_ms else 0.0
    avg_block_time_ms = mean(block_times_ms) if block_times_ms else 0.0

    return {
        "runs": runs,
        "expectancy_per_trade": mean(all_r) if all_r else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else 0.0,
        "sharpe_approx": sharpe,
        "avg_trades_per_simulation": mean(trades_per_run) if trades_per_run else 0.0,
        "max_drawdown": max((r["max_drawdown"] for r in results), default=0.0),
        "winrate": (total_wins / total_trades) * 100 if total_trades else 0.0,
        "r_distribution": {
            "p50": median(all_r) if all_r else 0.0,
            "p05": _percentile(all_r, 0.05),
            "p95": _percentile(all_r, 0.95),
        },
        "sample_trade_logs": results[0]["trade_logs"][:5] if results else [],
        "profiling": {
            "avg_simulation_ms": avg_sim_time_ms,
            "avg_time_per_100_sims_ms": avg_block_time_ms,
            "progress_enabled": bool(progress and tqdm is not None),
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(percentile * (len(ordered) - 1))))
    return ordered[index]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    configure_logging()
    logger = get_logger(__name__)

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

    use_real_data = _env_flag("USE_REAL_DATA", USE_REAL_DATA)

    if use_real_data:
        logger.info("RUNNING IN REAL DATA MODE")
        csv_path = Path(os.getenv("REAL_DATA_CSV_PATH", str(DEFAULT_REAL_DATA_CSV)))
        market_df = load_real_market_data(csv_path)

        start = time.time()
        metrics = run_single_backtest(strategy=high_vol_engine, market_df=market_df)
        elapsed = time.time() - start

        total_trades = int(metrics["trades"])
        wins = int(metrics["wins"])
        winrate = (wins / total_trades) * 100 if total_trades else 0.0

        logger.info("Total trades: %s", total_trades)
        logger.info("Winrate real: %.2f%%", winrate)
        logger.info("Profit factor real: %.4f", float(metrics["profit_factor"]))
        logger.info("Max drawdown real: %.4fR", float(metrics["max_drawdown"]))
        logger.info("Expectancy real: %.4fR", float(metrics["expectancy"]))
        logger.info("Elapsed: %.2fs", elapsed)

        print(metrics)
    else:
        runs = int(os.getenv("MONTE_CARLO_SIMULATIONS", "10000"))
        show_progress = os.getenv("MONTE_CARLO_PROGRESS", "0") == "1"
        verbose_output = os.getenv("MONTE_CARLO_VERBOSE", "1") == "1"
        start = time.time()
        metrics = run_monte_carlo_profiled(strategy=high_vol_engine, runs=runs, progress=show_progress)

        if verbose_output:
            logger.info("HIGH_VOL MVP metrics=%s", metrics)
            print(metrics)
            print(f"Elapsed: {time.time() - start:.2f}s")
