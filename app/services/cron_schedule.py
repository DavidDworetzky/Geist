"""Five-field cron parsing and time-zone-aware next-run calculation."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_FIELD_SPECS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day of month", 1, 31),
    ("month", 1, 12),
    ("day of week", 0, 7),
)
_MAX_EXPRESSION_LENGTH = 100
_SEARCH_YEARS = 8


@dataclass(frozen=True)
class CronExpression:
    """Parsed five-field cron expression."""

    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    day_of_month_restricted: bool
    day_of_week_restricted: bool

    def matches_date(self, value: datetime.date) -> bool:
        """Return whether a local calendar date satisfies the cron date fields."""
        if value.month not in self.months:
            return False

        day_of_month_matches = value.day in self.days_of_month
        cron_weekday = (value.weekday() + 1) % 7
        day_of_week_matches = cron_weekday in self.days_of_week

        if self.day_of_month_restricted and self.day_of_week_restricted:
            return day_of_month_matches or day_of_week_matches
        if self.day_of_month_restricted:
            return day_of_month_matches
        if self.day_of_week_restricted:
            return day_of_week_matches
        return True

    def matches(self, local_datetime: datetime.datetime) -> bool:
        """Return whether a local datetime satisfies every cron field."""
        return (
            local_datetime.minute in self.minutes
            and local_datetime.hour in self.hours
            and self.matches_date(local_datetime.date())
        )


def _expand_part(part: str, minimum: int, maximum: int, field_name: str) -> set[int]:
    base, separator, step_text = part.partition("/")
    if separator:
        if not step_text or "/" in step_text:
            raise ValueError(f"Invalid {field_name} step: {part!r}")
        try:
            step = int(step_text)
        except ValueError as error:
            raise ValueError(f"Invalid {field_name} step: {step_text!r}") from error
        if step <= 0:
            raise ValueError(f"{field_name} step must be greater than zero")
    else:
        step = 1

    if base == "*":
        start, end = minimum, maximum
    elif "-" in base:
        start_text, range_separator, end_text = base.partition("-")
        if not range_separator or not start_text or not end_text or "-" in end_text:
            raise ValueError(f"Invalid {field_name} range: {base!r}")
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as error:
            raise ValueError(f"Invalid {field_name} range: {base!r}") from error
    else:
        try:
            start = int(base)
        except ValueError as error:
            raise ValueError(f"Invalid {field_name} value: {base!r}") from error
        end = maximum if separator else start

    if start < minimum or start > maximum or end < minimum or end > maximum:
        raise ValueError(f"{field_name} values must be between {minimum} and {maximum}")
    if start > end:
        raise ValueError(f"Invalid descending {field_name} range: {base!r}")
    return set(range(start, end + 1, step))


def _parse_field(value: str, minimum: int, maximum: int, field_name: str) -> frozenset[int]:
    if not value:
        raise ValueError(f"Missing {field_name} field")
    parts = value.split(",")
    if any(not part for part in parts):
        raise ValueError(f"Invalid empty item in {field_name}")

    values: set[int] = set()
    for part in parts:
        values.update(_expand_part(part, minimum, maximum, field_name))
    if not values:
        raise ValueError(f"{field_name} has no values")
    return frozenset(values)


def parse_cron(expression: str) -> CronExpression:
    """Parse and validate a standard numeric five-field cron expression."""
    normalized = " ".join(expression.strip().split())
    if not normalized:
        raise ValueError("Cron expression is required")
    if len(normalized) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("Cron expression is too long")

    fields = normalized.split(" ")
    if len(fields) != 5:
        raise ValueError("Cron expression must contain exactly five fields")

    parsed = [
        _parse_field(field, minimum, maximum, name)
        for field, (name, minimum, maximum) in zip(fields, _FIELD_SPECS, strict=True)
    ]
    days_of_week = frozenset(0 if value == 7 else value for value in parsed[4])
    return CronExpression(
        expression=normalized,
        minutes=parsed[0],
        hours=parsed[1],
        days_of_month=parsed[2],
        months=parsed[3],
        days_of_week=days_of_week,
        day_of_month_restricted=parsed[2] != frozenset(range(1, 32)),
        day_of_week_restricted=days_of_week != frozenset(range(0, 7)),
    )


def validate_timezone(name: str) -> ZoneInfo:
    """Return an IANA time zone or raise a user-facing validation error."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("Time zone is required")
    try:
        return ZoneInfo(normalized)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown IANA time zone: {normalized}") from error


def _as_utc(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def _valid_local_candidate(
    day: datetime.date,
    hour: int,
    minute: int,
    timezone: ZoneInfo,
) -> datetime.datetime | None:
    # fold=0 makes a repeated fall-back minute execute once. The UTC round trip
    # rejects nonexistent spring-forward wall-clock values.
    local = datetime.datetime.combine(day, datetime.time(hour, minute), timezone).replace(fold=0)
    utc_value = local.astimezone(datetime.UTC)
    round_trip = utc_value.astimezone(timezone)
    if (
        round_trip.date() != day
        or round_trip.hour != hour
        or round_trip.minute != minute
        or round_trip.fold != 0
    ):
        return None
    return utc_value


def next_fire_after(
    expression: CronExpression | str,
    after_utc: datetime.datetime,
    timezone_name: str = "UTC",
) -> datetime.datetime:
    """Return the first UTC minute strictly after ``after_utc`` that matches."""
    cron = parse_cron(expression) if isinstance(expression, str) else expression
    timezone = validate_timezone(timezone_name)
    after = _as_utc(after_utc)
    local_after = after.astimezone(timezone)
    current_day = local_after.date()
    end_day = current_day + datetime.timedelta(days=366 * _SEARCH_YEARS)

    hours = sorted(cron.hours)
    minutes = sorted(cron.minutes)
    while current_day <= end_day:
        if cron.matches_date(current_day):
            for hour in hours:
                for minute in minutes:
                    candidate = _valid_local_candidate(current_day, hour, minute, timezone)
                    if candidate is not None and candidate > after:
                        return candidate.replace(tzinfo=None)
        current_day += datetime.timedelta(days=1)

    raise ValueError("Cron expression has no future occurrence within eight years")
