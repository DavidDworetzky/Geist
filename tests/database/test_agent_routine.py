import datetime
import importlib

import pytest

from app.models.database.agent_routine import (
    AgentRoutine,
    claim_routine_run,
    create_routine,
    finish_routine_run,
    get_due_routines,
    get_routine,
    schedule_routine_once,
    update_routine,
)
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'routines.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    try:
        with SessionLocal() as session:
            session.add(GeistUser(user_id=41, username="owner", email="owner@example.com"))
            session.commit()
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def test_claim_routine_run_allows_only_one_scheduler(sqlite_database):
    routine = create_routine(
        user_id=41,
        name="digest",
        prompt="Summarize the week",
        interval_minutes=60,
    )
    due_at = datetime.datetime(2026, 8, 18, 12, 0, 0)
    with SessionLocal() as session:
        stored = session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).one()
        stored.next_run_at = due_at
        session.commit()
    routine.next_run_at = due_at
    claimed_at = due_at + datetime.timedelta(seconds=1)

    assert claim_routine_run(routine, now=claimed_at) is True
    assert claim_routine_run(routine, now=claimed_at) is False

    with SessionLocal() as session:
        stored = session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).one()
        assert stored.last_run_at == claimed_at
        assert stored.next_run_at == claimed_at + datetime.timedelta(minutes=60)


def test_disabled_routine_one_off_does_not_enable_recurrence(sqlite_database):
    routine = create_routine(41, "once", "hello", 60, enabled=False)
    assert routine.next_run_at is None
    assert get_due_routines() == []
    scheduled = schedule_routine_once(routine.routine_id)
    assert not scheduled.enabled and scheduled.run_once_requested
    assert [r.routine_id for r in get_due_routines()] == [routine.routine_id]
    assert claim_routine_run(scheduled)
    stored = get_routine(routine.routine_id)
    assert not stored.enabled and not stored.run_once_requested
    assert stored.next_run_at is None
    assert get_due_routines() == []


def test_legacy_disabled_timestamp_is_not_authorization_to_run(sqlite_database):
    routine = create_routine(41, "disabled", "hello", 60, enabled=False)
    with SessionLocal() as session:
        session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).update(
            {
                "next_run_at": datetime.datetime(2000, 1, 1),
            }
        )
        session.commit()
    assert get_due_routines() == []
    assert not claim_routine_run(get_routine(routine.routine_id))


def test_outcomes_are_bound_to_claim_and_timeout_cannot_be_overwritten(sqlite_database):
    routine = create_routine(41, "outcome", "hello", 60, enabled=False)
    claimed = schedule_routine_once(routine.routine_id)
    assert claim_routine_run(claimed)
    assert get_routine(routine.routine_id).last_status == "running"
    finish_routine_run(claimed, "timed_out", "Execution budget exhausted")
    finish_routine_run(claimed, "succeeded")
    assert get_routine(routine.routine_id).last_status == "timed_out"
    newer = schedule_routine_once(routine.routine_id)
    assert claim_routine_run(newer)
    finish_routine_run(claimed, "failed", "Old failure")
    assert get_routine(routine.routine_id).last_status == "running"
    finish_routine_run(newer, "succeeded")
    stored = get_routine(routine.routine_id)
    assert stored.last_status == "succeeded" and stored.last_error is None


def test_interval_edit_reschedules_before_first_run_and_rejects_internal_fields(sqlite_database):
    routine = create_routine(41, "interval", "hello", 60)
    updated = update_routine(routine.routine_id, {"interval_minutes": 10})
    assert updated.last_run_at is None
    assert updated.next_run_at < routine.next_run_at - datetime.timedelta(minutes=40)
    for update in ({"user_id": 9}, {"next_run_at": None}, {"enabled": None}):
        with pytest.raises(ValueError):
            update_routine(routine.routine_id, update)


def test_interval_edit_preserves_queued_one_off(sqlite_database):
    routine = create_routine(41, "queued", "hello", 60)
    queued = schedule_routine_once(routine.routine_id)
    updated = update_routine(routine.routine_id, {"interval_minutes": 120})
    assert updated.run_once_requested
    assert updated.next_run_at == queued.next_run_at
