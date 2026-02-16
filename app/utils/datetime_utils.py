from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a timezone-aware datetime in UTC."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Naive datetime detected")
    return dt.astimezone(UTC)


def utc_isoformat(dt: datetime) -> str:
    """Serialize datetimes in a UTC ISO-8601 representation."""
    return ensure_utc(dt).isoformat().replace("+00:00", "Z")
