from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import BaseModel, Field


class SignalRejectionCounters(BaseModel):
    rejected_by_score: int = 0
    rejected_by_volume: int = 0
    rejected_by_atr: int = 0
    rejected_by_regime: int = 0
    rejected_by_confidence: int = 0

    def increment_for_reasons(self, reasons: Iterable[str]) -> None:
        categories = {_reason_category(reason) for reason in reasons}
        if "score" in categories:
            self.rejected_by_score += 1
        if "volume" in categories:
            self.rejected_by_volume += 1
        if "atr" in categories:
            self.rejected_by_atr += 1
        if "regime" in categories:
            self.rejected_by_regime += 1
        if "confidence" in categories:
            self.rejected_by_confidence += 1

    def has_rejections(self) -> bool:
        return any(
            (
                self.rejected_by_score,
                self.rejected_by_volume,
                self.rejected_by_atr,
                self.rejected_by_regime,
                self.rejected_by_confidence,
            )
        )

    def reset(self) -> None:
        self.rejected_by_score = 0
        self.rejected_by_volume = 0
        self.rejected_by_atr = 0
        self.rejected_by_regime = 0
        self.rejected_by_confidence = 0


class SignalRejectionEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    strategy: str
    regime: str | None = None
    score: float | None = None
    required_score: float | None = None
    atr: float | None = None
    atr_status: str | None = None
    volume_ratio: float | None = None
    ema_distance: float | None = None
    reasons: list[str]

    @property
    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else "unknown"


class SignalRejectionTracker:
    def __init__(self, now_fn: Callable[[], datetime] | None = None) -> None:
        self.counters = SignalRejectionCounters()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._summary_hour = self._hour_key(self._now_fn())

    def record(self, event: SignalRejectionEvent) -> None:
        self.counters.increment_for_reasons(event.reasons)

    def maybe_hourly_summary_lines(self) -> list[str]:
        current_hour = self._hour_key(self._now_fn())
        if current_hour == self._summary_hour:
            return []

        self._summary_hour = current_hour
        if not self.counters.has_rejections():
            return []

        lines = self.format_summary_lines(self.counters)
        self.counters.reset()
        return lines

    @staticmethod
    def format_rejection_lines(event: SignalRejectionEvent) -> list[str]:
        return [
            "[REJECTED SIGNAL]",
            f"timestamp={event.timestamp}",
            f"strategy={event.strategy}",
            f"regime={event.regime or 'UNKNOWN'}",
            f"score={_format_optional_number(event.score)}",
            f"required_score={_format_optional_number(event.required_score)}",
            f"atr={_format_atr(event.atr, event.atr_status)}",
            f"volume_ratio={_format_optional_number(event.volume_ratio)}",
            f"ema_distance={_format_optional_number(event.ema_distance)}",
            f"reason={event.primary_reason}",
            f"rejection_reasons={','.join(event.reasons) if event.reasons else 'unknown'}",
        ]

    @staticmethod
    def format_summary_lines(counters: SignalRejectionCounters) -> list[str]:
        return [
            "SIGNAL REJECTION SUMMARY",
            "",
            f"score_rejections={counters.rejected_by_score}",
            f"volume_rejections={counters.rejected_by_volume}",
            f"atr_rejections={counters.rejected_by_atr}",
            f"regime_rejections={counters.rejected_by_regime}",
            f"confidence_rejections={counters.rejected_by_confidence}",
        ]

    @staticmethod
    def _hour_key(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")


def build_rejection_event(
    *,
    strategy: str,
    regime: str | None,
    reasons: list[str],
    score: float | None = None,
    required_score: float | None = None,
    atr: float | None = None,
    atr_status: str | None = None,
    volume_ratio: float | None = None,
    ema_distance: float | None = None,
) -> SignalRejectionEvent:
    return SignalRejectionEvent(
        strategy=strategy,
        regime=regime,
        score=score,
        required_score=required_score,
        atr=atr,
        atr_status=atr_status,
        volume_ratio=volume_ratio,
        ema_distance=ema_distance,
        reasons=reasons,
    )


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_value == 0.0:
        return None
    return numerator_value / denominator_value


def _reason_category(reason: str) -> str | None:
    normalized = reason.lower()
    if "score" in normalized:
        return "score"
    if "volume" in normalized:
        return "volume"
    if "atr" in normalized or "volatility" in normalized:
        return "atr"
    if "regime" in normalized or "low_activity" in normalized:
        return "regime"
    if "confidence" in normalized or "expected_move" in normalized or "profitability" in normalized:
        return "confidence"
    return None


def _format_atr(atr: float | None, atr_status: str | None) -> str:
    if atr_status and atr is None:
        return atr_status
    if atr_status and atr is not None:
        return f"{_format_optional_number(atr)} ({atr_status})"
    return _format_optional_number(atr)


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
