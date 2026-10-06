"""Deterministic initial lifecycle policy; these values are not experiment optima."""

from datetime import datetime, timedelta, timezone
import math


CATEGORY_POLICIES = {
    "device_profile": (0.90, 180),
    "home_environment": (0.75, 90),
    "usage_preference": (0.65, 60),
}
DEFAULT_POLICY = (0.50, 30)  # Historical facts have no validated category.
REPEAT_BONUS_STEP = 0.02
REPEAT_BONUS_MAX = 0.10


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def checked_now(provider) -> datetime:
    value = provider()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now_provider must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def as_utc(value: datetime) -> datetime:
    # SQLite test drivers drop timezone metadata; production uses TIMESTAMPTZ.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def policy(category: str | None) -> tuple[float, int]:
    if category is not None and category not in CATEGORY_POLICIES:
        raise ValueError("Unsupported fact_category.")
    return CATEGORY_POLICIES.get(category, DEFAULT_POLICY)


def importance_for(category: str | None, seen_count: int) -> float:
    if type(seen_count) is not int or seen_count < 1:
        raise ValueError("seen_count must be positive.")
    base, _ttl = policy(category)
    return min(1.0, base + min(REPEAT_BONUS_MAX, REPEAT_BONUS_STEP * (seen_count - 1)))


def expiration_for(category: str | None, now: datetime) -> datetime:
    return now + timedelta(days=policy(category)[1])


def effective_score(*, importance: float, seen_count: int, last_seen_at: datetime,
                    fact_category: str | None, now: datetime,
                    expires_at: datetime | None = None) -> float:
    if not math.isfinite(importance) or not 0 <= importance <= 1:
        raise ValueError("importance must be finite and between 0 and 1.")
    if type(seen_count) is not int or seen_count < 1:
        raise ValueError("seen_count must be positive.")
    current = as_utc(now)
    if expires_at is not None and as_utc(expires_at) <= current:
        return 0.0
    age_days = max(0.0, (current - as_utc(last_seen_at)).total_seconds() / 86400)
    half_life_days = policy(fact_category)[1] / 2
    recency_decay = 0.5 ** (age_days / half_life_days)
    repetition_factor = 1.0 + min(0.20, 0.02 * (seen_count - 1))
    return min(1.0, max(0.0, importance * recency_decay * repetition_factor))
