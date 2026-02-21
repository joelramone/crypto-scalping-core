from __future__ import annotations

import os
import random
import time
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any

from importlib import util as importlib_util
from pathlib import Path

import numpy as np

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


def _rolling_mean_np(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] < window:
        return output

    cumsum = np.cumsum(values, dtype=np.float64)
    output[window - 1 :] = (cumsum[window - 1 :] - np.concatenate(([0.0], cumsum[:-window]))) / float(window)
    return output


def _compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return _rolling_mean_np(true_range, period)


def _compute_regime_score_series(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    strategy: HighVolEngine,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vm = strategy.volatility_metrics
    sc = strategy.score_calculator
    classifier = strategy.classifier

    n = close.shape[0]
    score_values = np.full(n, np.nan, dtype=np.float64)
    regime_is_high = np.zeros(n, dtype=bool)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    returns = np.divide(close - prev_close, prev_close, out=np.zeros_like(close), where=prev_close != 0)
    squared_returns = returns * returns
    rv = np.sqrt(np.maximum(_rolling_mean_np(squared_returns[1:], vm.rv_window), 0.0))
    rv_full = np.full(n, np.nan, dtype=np.float64)
    rv_full[1:] = rv

    ranges = high - low
    range_mean = _rolling_mean_np(ranges, vm.range_window)
    prev_range_mean = np.full(n, np.nan, dtype=np.float64)
    for i in range((2 * vm.range_window) - 1, n):
        prev_range_mean[i] = range_mean[i - vm.range_window]

    atr_diff = np.abs(np.diff(atr, prepend=np.nan))
    atr_diff[np.isnan(atr_diff)] = 0.0
    vol_of_vol = _rolling_mean_np(atr_diff, vm.clustering_window)
    clustering_baseline = _rolling_mean_np(np.nan_to_num(vol_of_vol, nan=0.0), vm.lookback_window)
    volume_mean = _rolling_mean_np(volume, vm.volume_window)
    baseline_volume = np.full(n, np.nan, dtype=np.float64)
    for i in range((2 * vm.volume_window) - 1, n):
        baseline_volume[i] = volume_mean[i - vm.volume_window]

    atr_sorted: list[float] = []
    rv_sorted: list[float] = []
    atr_window_values: deque[float] = deque()
    rv_window_values: deque[float] = deque()

    min_required = max(vm.lookback_window + vm.atr_window, vm.rv_window + 1)

    for i in range(n):
        atr_value = atr[i]
        rv_value = rv_full[i]

        if not np.isnan(atr_value):
            atr_window_values.append(float(atr_value))
            insort(atr_sorted, float(atr_value))
            if len(atr_window_values) > vm.lookback_window:
                old_atr = atr_window_values.popleft()
                atr_sorted.pop(bisect_left(atr_sorted, old_atr))

        if not np.isnan(rv_value):
            rv_window_values.append(float(rv_value))
            insort(rv_sorted, float(rv_value))
            if len(rv_window_values) > vm.lookback_window:
                old_rv = rv_window_values.popleft()
                rv_sorted.pop(bisect_left(rv_sorted, old_rv))

        if i + 1 < min_required:
            continue
        if not atr_sorted or not rv_sorted or np.isnan(range_mean[i]) or np.isnan(prev_range_mean[i]):
            continue
        if np.isnan(vol_of_vol[i]) or np.isnan(volume_mean[i]) or np.isnan(baseline_volume[i]):
            continue

        atr_percentile = bisect_left(atr_sorted, float(atr_value)) + 1
        atr_percentile /= len(atr_sorted)
        rv_percentile = bisect_left(rv_sorted, float(rv_value)) + 1
        rv_percentile /= len(rv_sorted)

        range_to_mean_ratio = 0.0 if range_mean[i] == 0 else ranges[i] / range_mean[i]
        range_expansion_ratio = 0.0 if prev_range_mean[i] == 0 else range_mean[i] / prev_range_mean[i]

        clustering_den = clustering_baseline[i]
        clustering_ratio = 0.0 if clustering_den == 0 or np.isnan(clustering_den) else vol_of_vol[i] / clustering_den

        volume_expansion_ratio = 0.0 if baseline_volume[i] == 0 else volume_mean[i] / baseline_volume[i]

        atr_signal = max(0.0, min(1.0, atr_percentile))
        rv_signal = max(0.0, min(1.0, rv_percentile))
        range_signal = max(0.0, min(1.0, 0.5 * range_expansion_ratio + 0.5 * range_to_mean_ratio - 1.0))
        clustering_signal = max(0.0, min(1.0, clustering_ratio - 1.0))
        volume_signal = max(0.0, min(1.0, volume_expansion_ratio - 1.0))

        score = (
            atr_signal * sc.weights["atr"]
            + rv_signal * sc.weights["rv"]
            + range_signal * sc.weights["range"]
            + clustering_signal * sc.weights["clustering"]
            + volume_signal * sc.weights["volume"]
        )
        active_count = sum(v >= 0.60 for v in (atr_signal, rv_signal, range_signal, clustering_signal, volume_signal))

        score_values[i] = score
        is_positive = (
            score >= classifier.threshold
            and active_count >= classifier.min_active_features
            and volume_expansion_ratio >= classifier.min_volume_expansion_ratio
        )
        if is_positive:
            classifier._positive_streak += 1
        else:
            classifier._positive_streak = 0

        regime_is_high[i] = classifier._positive_streak >= classifier.confirmation_bars

    classifier.reset()
    return score_values, regime_is_high, rv_full


def _rolling_prev_extrema(close: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    prev_high = np.full(n, np.nan, dtype=np.float64)
    prev_low = np.full(n, np.nan, dtype=np.float64)
    max_deque: deque[int] = deque()
    min_deque: deque[int] = deque()

    for i in range(n):
        while max_deque and max_deque[0] < i - window_size:
            max_deque.popleft()
        while min_deque and min_deque[0] < i - window_size:
            min_deque.popleft()

        if i > 0 and i >= window_size:
            prev_high[i] = close[max_deque[0]]
            prev_low[i] = close[min_deque[0]]

        while max_deque and close[max_deque[-1]] <= close[i]:
            max_deque.pop()
        max_deque.append(i)

        while min_deque and close[min_deque[-1]] >= close[i]:
            min_deque.pop()
        min_deque.append(i)

    return prev_high, prev_low


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
        close = market_df["close"].to_numpy(dtype=np.float64)
        high = market_df["high"].to_numpy(dtype=np.float64)
        low = market_df["low"].to_numpy(dtype=np.float64)
        volume = market_df["volume"].to_numpy(dtype=np.float64)

        atr_period = strategy.volatility_metrics.atr_window
        if "atr" in market_df.columns:
            atr = market_df["atr"].to_numpy(dtype=np.float64)
        else:
            atr = _compute_atr_np(high=high, low=low, close=close, period=atr_period)

        n = close.shape[0]
        equity_curve = np.zeros(n, dtype=np.float64)
        positions = np.zeros(n, dtype=np.int8)
        score_values, regime_is_high, _ = _compute_regime_score_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            atr=atr,
            strategy=strategy,
        )

        lookback_period = int(getattr(strategy.breakout_strategy.config, "lookback_period", 20))
        window_size = max(1, lookback_period - 1)
        prev_window_high, prev_window_low = _rolling_prev_extrema(close, window_size)
        breakout_buffer_atr = float(getattr(strategy.breakout_strategy.config, "breakout_buffer_atr", 0.0))

        cumulative_r = 0.0
        r_multiples: list[float] = []
        trade_logs: list[dict[str, float | int | str]] = []
        active: Position | None = None

        start_index = max(lookback_period, atr_period, strategy.volatility_metrics.rv_window + 1)
        for i in range(start_index, n):
            price = close[i]
            current_atr = atr[i]
            if np.isnan(current_atr) or current_atr <= 0:
                equity_curve[i] = cumulative_r
                continue

            if active is not None:
                positions[i] = 1 if active.side == "LONG" else -1
                if active.trailing_stop_enabled:
                    trail_distance = current_atr * active.trailing_stop_atr_multiplier
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
                    r_multiples.append(r_multiple)
                    trade_payload = {
                        "regime_score": active.regime_score,
                        "atr": active.atr,
                        "entry_price": active.entry,
                        "stop": active.initial_sl,
                        "target": active.tp,
                        "risk_per_trade": active.risk_per_trade,
                        "reward_per_trade": active.reward_per_trade,
                        "r_multiple": r_multiple,
                        "duration_ticks": i - active.opened_tick,
                        "exit_reason": exit_reason,
                    }
                    trade_logs.append(trade_payload)
                    strategy.record_trade_outcome(trade_payload)
                    cumulative_r += r_multiple
                    active = None

            if active is None and regime_is_high[i] and i >= lookback_period:
                buffer = current_atr * breakout_buffer_atr
                window_high = prev_window_high[i]
                window_low = prev_window_low[i]
                if np.isnan(window_high) or np.isnan(window_low):
                    equity_curve[i] = cumulative_r
                    continue

                signal_side = None
                if price >= (window_high - buffer):
                    signal_side = "LONG"
                elif price <= (window_low + buffer):
                    signal_side = "SHORT"

                if signal_side is not None:
                    signal = strategy.breakout_strategy._build_signal(side=signal_side, entry=float(price), atr_value=float(current_atr))
                    entry_price = float(signal["entry"])
                    stop_price = float(signal["sl"])
                    target_price = float(signal["tp"])
                    risk_per_trade = abs(entry_price - stop_price)
                    reward_per_trade = abs(target_price - entry_price)
                    active = Position(
                        side=signal_side,
                        entry=entry_price,
                        sl=stop_price,
                        tp=target_price,
                        initial_sl=stop_price,
                        risk_per_trade=risk_per_trade,
                        reward_per_trade=reward_per_trade,
                        opened_tick=i,
                        regime_score=float(score_values[i]) if not np.isnan(score_values[i]) else 0.0,
                        atr=float(current_atr),
                        trailing_stop_enabled=bool(signal.get("trailing_stop_enabled", False)),
                        trailing_stop_atr_multiplier=float(signal.get("trailing_stop_atr_multiplier", 1.0)),
                    )

            equity_curve[i] = cumulative_r

        return _finalize_backtest_metrics(
            r_multiples=r_multiples,
            trade_logs=trade_logs,
            equity_curve=equity_curve.tolist(),
        )

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
        logger.info("Execution time: %.2fs", elapsed)

        print("Execution time:", elapsed)
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
