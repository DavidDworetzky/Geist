"""Tests for the dependency-free five-field cron evaluator."""

import datetime

import pytest

from app.services.cron_schedule import next_fire_after, parse_cron, validate_timezone


def test_parse_lists_ranges_steps_and_sunday_alias():
    cron = parse_cron("*/15 8-10 * * 0,7")

    assert cron.minutes == frozenset({0, 15, 30, 45})
    assert cron.hours == frozenset({8, 9, 10})
    assert cron.days_of_week == frozenset({0})


@pytest.mark.parametrize(
    "expression",
    ["", "* * * *", "60 * * * *", "* 24 * * *", "*/0 * * * *", "a * * * *"],
)
def test_invalid_cron_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        parse_cron(expression)


def test_day_of_month_and_week_use_cron_or_semantics():
    cron = parse_cron("0 9 15 * 1")

    assert cron.matches(datetime.datetime(2026, 9, 14, 9, 0))  # Monday
    assert cron.matches(datetime.datetime(2026, 9, 15, 9, 0))  # Fifteenth
    assert not cron.matches(datetime.datetime(2026, 9, 16, 9, 0))


def test_next_fire_uses_requested_timezone():
    result = next_fire_after(
        "0 9 * * *",
        datetime.datetime(2026, 1, 5, 14, 30),
        "America/Chicago",
    )

    assert result == datetime.datetime(2026, 1, 5, 15, 0)


def test_spring_forward_nonexistent_minute_is_skipped():
    result = next_fire_after(
        "30 2 * * *",
        datetime.datetime(2026, 3, 8, 6, 0),
        "America/Chicago",
    )

    assert result == datetime.datetime(2026, 3, 9, 7, 30)


def test_fall_back_repeated_minute_fires_only_on_first_fold():
    first = next_fire_after(
        "30 1 * * *",
        datetime.datetime(2026, 11, 1, 5, 0),
        "America/Chicago",
    )
    after_first = next_fire_after("30 1 * * *", first, "America/Chicago")

    assert first == datetime.datetime(2026, 11, 1, 6, 30)
    assert after_first == datetime.datetime(2026, 11, 2, 7, 30)


def test_invalid_timezone_is_rejected():
    with pytest.raises(ValueError, match="Unknown IANA time zone"):
        validate_timezone("Mars/Olympus_Mons")
