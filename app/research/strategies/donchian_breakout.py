"""Donchian Breakout research strategy."""

from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING

import pandas as pd

from app.research.signal_quality.scoring import calculate_long_breakout_quality
from app.research.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from app.research.simulation import BacktestTrade


class DonchianBreakoutStrategy(BaseStrategy):
    """Long-only Donchian breakout strategy with optional quality filtering."""

    def __init__(
        self,
        lookback: int = 20,
        volume_ratio: float = 1.2,
        take_profit_pct: float = 0.004,
        stop_loss_pct: float = 0.0025,
        max_holding_candles: int = 30,
        min_quality_score: int = 0,
        min_body_to_range: float = 0.0,
        min_close_location: float = 0.0,
        min_range_expansion: float = 0.0,
        min_atr_expansion: float = 0.0,
        min_ema20_slope_pct: float = float("-inf"),
        min_ema_alignment_strength: float = float("-inf"),
        min_breakout_distance_pct: float = 0.0,
    ) -> None:
        self.lookback = lookback
        self.volume_ratio = volume_ratio
        self._take_profit_pct = take_profit_pct
        self._stop_loss_pct = stop_loss_pct
        self._max_holding_candles = max_holding_candles

        self.min_quality_score = min_quality_score
        self.min_body_to_range = min_body_to_range
        self.min_close_location = min_close_location
        self.min_range_expansion = min_range_expansion
        self.min_atr_expansion = min_atr_expansion
        self.min_ema20_slope_pct = min_ema20_slope_pct
        self.min_ema_alignment_strength = min_ema_alignment_strength
        self.min_breakout_distance_pct = min_breakout_distance_pct

        self.last_quality_scores = pd.Series(dtype="int64")
        self.last_quality_components = pd.DataFrame()
        self.last_breakout_distance_pct = pd.Series(dtype="float64")

    def name(self) -> str:
        """Return the strategy name for CLI output."""
        return "donchian_breakout"

    def quality_filter_active(self) -> bool:
        """Return whether explicit quality gating is enabled."""
        return self.min_quality_score > 0

    def _prepare_quality_scores(
        self,
        df: pd.DataFrame,
        donchian_high: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Calculate breakout quality scores for long candidates."""
        breakout_distance_pct = (
            (df["close"] - donchian_high)
            .div(df["close"].where(df["close"] != 0.0))
            .fillna(0.0)
        )
        quality_score, quality_components = calculate_long_breakout_quality(
            df,
            breakout_distance_pct,
            min_body_to_range=self.min_body_to_range,
            min_close_location=self.min_close_location,
            min_range_expansion=self.min_range_expansion,
            min_atr_expansion=self.min_atr_expansion,
            min_ema20_slope_pct=self.min_ema20_slope_pct,
            min_ema_alignment_strength=self.min_ema_alignment_strength,
            min_breakout_distance_pct=self.min_breakout_distance_pct,
        )
        return quality_score, quality_components, breakout_distance_pct

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return Donchian Breakout long-only entry signals."""
        donchian_high = (
            df["high"].rolling(window=self.lookback, min_periods=self.lookback).max().shift(1)
        )
        atr_median = df["atr14"].rolling(window=20, min_periods=20).median()

        base_entries = (
            (df["close"] > donchian_high)
            & (df["close"] > df["ema200"])
            & (df["ema20_slope"] > 0)
            & (df["volume_ratio"] > self.volume_ratio)
            & (df["atr14"] > atr_median)
        ).fillna(False)

        quality_score, quality_components, breakout_distance_pct = self._prepare_quality_scores(
            df,
            donchian_high,
        )
        self.last_quality_scores = quality_score
        self.last_quality_components = quality_components
        self.last_breakout_distance_pct = breakout_distance_pct

        if not self.quality_filter_active():
            return base_entries.fillna(False)

        return (base_entries & (quality_score >= self.min_quality_score)).fillna(False)

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return strategy-driven exits. Donchian exits rely on TP/SL/holding only."""
        return pd.Series(False, index=df.index)

    def take_profit_pct(self) -> float:
        """Return configured take-profit percentage."""
        return self._take_profit_pct

    def stop_loss_pct(self) -> float:
        """Return configured stop-loss percentage."""
        return self._stop_loss_pct

    def max_holding_candles(self) -> int:
        """Return configured maximum holding candles."""
        return self._max_holding_candles

    def summarize_trade_quality(self, trades: list["BacktestTrade"]) -> list[dict[str, float | int]]:
        """Aggregate simulated trade results by stored quality score."""
        if self.last_quality_scores.empty or not trades:
            return []

        grouped: dict[int, list[BacktestTrade]] = {}
        for trade in trades:
            score = int(self.last_quality_scores.iloc[trade.entry_index])
            grouped.setdefault(score, []).append(trade)

        summary: list[dict[str, float | int]] = []
        for score in sorted(grouped):
            score_trades = grouped[score]
            winning = [trade for trade in score_trades if trade.net_pnl > 0.0]
            losing = [trade for trade in score_trades if trade.net_pnl <= 0.0]
            gross_profit = sum(trade.net_pnl for trade in winning)
            gross_loss = abs(sum(trade.net_pnl for trade in losing))
            profit_factor = (
                (gross_profit / gross_loss)
                if gross_loss
                else (inf if gross_profit > 0.0 else 0.0)
            )
            summary.append(
                {
                    "score": score,
                    "trades": len(score_trades),
                    "win_rate": len(winning) / len(score_trades),
                    "profit_factor": profit_factor,
                    "expectancy": sum(trade.net_pnl for trade in score_trades) / len(score_trades),
                }
            )
        return summary
