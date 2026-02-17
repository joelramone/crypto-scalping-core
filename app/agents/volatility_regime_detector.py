from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np


@dataclass(frozen=True)
class VolatilitySnapshot:
    atr: float
    atr_percentile: float
    realized_vol: float
    realized_vol_percentile: float
    range_expansion_ratio: float
    range_to_mean_ratio: float
    clustering_ratio: float
    volume_expansion_ratio: float


@dataclass(frozen=True)
class RegimeScore:
    value: float
    active_features: tuple[str, ...]
    snapshot: VolatilitySnapshot


class VolatilityMetrics:
    """Calcula métricas robustas para expansión estructural de volatilidad."""

    def __init__(
        self,
        atr_window: int = 14,
        lookback_window: int = 120,
        rv_window: int = 30,
        range_window: int = 30,
        clustering_window: int = 10,
        volume_window: int = 30,
    ) -> None:
        self.atr_window = atr_window
        self.lookback_window = lookback_window
        self.rv_window = rv_window
        self.range_window = range_window
        self.clustering_window = clustering_window
        self.volume_window = volume_window

    def compute(
        self,
        close: list[float],
        high: list[float],
        low: list[float],
        volume: list[float],
    ) -> VolatilitySnapshot | None:
        min_required = max(self.lookback_window + self.atr_window, self.rv_window + 1)
        if min(len(close), len(high), len(low), len(volume)) < min_required:
            return None

        true_ranges = self._true_ranges(close, high, low)
        atr_series = self._rolling_mean(true_ranges, self.atr_window)
        if not atr_series:
            return None

        atr = atr_series[-1]
        atr_percentile = self._percentile_rank(atr_series[-self.lookback_window :], atr)

        rv_series = self._realized_vol_series(close, self.rv_window)
        if not rv_series:
            return None
        realized_vol = rv_series[-1]
        realized_vol_percentile = self._percentile_rank(rv_series[-self.lookback_window :], realized_vol)

        range_series = [h - l for h, l in zip(high, low)]
        range_mean = mean(range_series[-self.range_window :])
        range_to_mean_ratio = self._safe_ratio(range_series[-1], range_mean)

        prev_range_mean = mean(range_series[-2 * self.range_window : -self.range_window])
        range_expansion_ratio = self._safe_ratio(range_mean, prev_range_mean)

        vol_of_vol = self._rolling_mean(
            [abs(a - b) for a, b in zip(atr_series[1:], atr_series[:-1])],
            self.clustering_window,
        )
        clustering_ratio = self._safe_ratio(vol_of_vol[-1], mean(vol_of_vol[-self.lookback_window :]))

        volume_mean = mean(volume[-self.volume_window :])
        baseline_volume = mean(volume[-(2 * self.volume_window) : -self.volume_window])
        volume_expansion_ratio = self._safe_ratio(volume_mean, baseline_volume)

        return VolatilitySnapshot(
            atr=atr,
            atr_percentile=atr_percentile,
            realized_vol=realized_vol,
            realized_vol_percentile=realized_vol_percentile,
            range_expansion_ratio=range_expansion_ratio,
            range_to_mean_ratio=range_to_mean_ratio,
            clustering_ratio=clustering_ratio,
            volume_expansion_ratio=volume_expansion_ratio,
        )

    @staticmethod
    def _true_ranges(close: list[float], high: list[float], low: list[float]) -> list[float]:
        ranges: list[float] = []
        prev_close = close[0]
        for c, h, l in zip(close, high, low):
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            ranges.append(tr)
            prev_close = c
        return ranges

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> list[float]:
        if len(values) < window:
            return []
        output: list[float] = []
        rolling_sum = sum(values[:window])
        output.append(rolling_sum / window)
        for idx in range(window, len(values)):
            rolling_sum += values[idx] - values[idx - window]
            output.append(rolling_sum / window)
        return output

    @staticmethod
    def _realized_vol_series(close: list[float], window: int) -> list[float]:
        if len(close) < window + 1:
            return []

        close_array = np.asarray(close, dtype=np.float64)
        prev_close = close_array[:-1]
        curr_close = close_array[1:]

        returns = np.divide(
            curr_close - prev_close,
            prev_close,
            out=np.zeros_like(curr_close),
            where=prev_close != 0,
        )

        squared_returns = returns * returns
        window_kernel = np.ones(window, dtype=np.float64)
        rolling_squared_sum = np.convolve(squared_returns, window_kernel, mode="valid")

        return np.sqrt(rolling_squared_sum / window).tolist()

    @staticmethod
    def _percentile_rank(values: list[float], target: float) -> float:
        if not values:
            return 0.0
        below_or_equal = sum(1 for v in values if v <= target)
        return below_or_equal / len(values)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator


class RegimeScoreCalculator:
    """Score compuesto para evitar dependencia en un único umbral."""

    def __init__(
        self,
        atr_weight: float = 0.25,
        rv_weight: float = 0.20,
        range_weight: float = 0.20,
        clustering_weight: float = 0.15,
        volume_weight: float = 0.20,
    ) -> None:
        total_weight = atr_weight + rv_weight + range_weight + clustering_weight + volume_weight
        if round(total_weight, 8) != 1.0:
            raise ValueError("Las ponderaciones del score deben sumar 1.0")

        self.weights = {
            "atr": atr_weight,
            "rv": rv_weight,
            "range": range_weight,
            "clustering": clustering_weight,
            "volume": volume_weight,
        }

    def compute(self, snapshot: VolatilitySnapshot) -> RegimeScore:
        signals = {
            "atr": self._clip(snapshot.atr_percentile),
            "rv": self._clip(snapshot.realized_vol_percentile),
            "range": self._clip(0.5 * snapshot.range_expansion_ratio + 0.5 * snapshot.range_to_mean_ratio - 1.0),
            "clustering": self._clip(snapshot.clustering_ratio - 1.0),
            "volume": self._clip(snapshot.volume_expansion_ratio - 1.0),
        }

        value = sum(signals[key] * self.weights[key] for key in signals)
        active = tuple(sorted(key for key, score in signals.items() if score >= 0.60))
        return RegimeScore(value=value, active_features=active, snapshot=snapshot)

    @staticmethod
    def _clip(value: float, floor: float = 0.0, cap: float = 1.0) -> float:
        return max(floor, min(cap, value))


class RegimeClassifier:
    HIGH_VOL = "HIGH_VOLATILITY"
    NORMAL = "NORMAL_VOLATILITY"

    def __init__(
        self,
        threshold: float = 0.62,
        min_active_features: int = 3,
        confirmation_bars: int = 2,
        min_volume_expansion_ratio: float = 1.05,
    ) -> None:
        self.threshold = threshold
        self.min_active_features = min_active_features
        self.confirmation_bars = confirmation_bars
        self.min_volume_expansion_ratio = min_volume_expansion_ratio
        self._positive_streak = 0

    def classify(self, score: RegimeScore) -> str:
        score_is_strong = score.value >= self.threshold
        breadth_is_strong = len(score.active_features) >= self.min_active_features
        volume_is_confirmed = (
            score.snapshot.volume_expansion_ratio >= self.min_volume_expansion_ratio
        )

        if score_is_strong and breadth_is_strong and volume_is_confirmed:
            self._positive_streak += 1
        else:
            self._positive_streak = 0

        if self._positive_streak >= self.confirmation_bars:
            return self.HIGH_VOL
        return self.NORMAL

    def reset(self) -> None:
        self._positive_streak = 0
