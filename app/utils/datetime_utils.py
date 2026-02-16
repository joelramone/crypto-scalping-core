from __future__ import annotations

from datetime import UTC, datetime


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Naive datetime detected")
    return dt.astimezone(UTC)
