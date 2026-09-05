"""Tests for prompt schedule persistence, enqueueing, and inference dispatch."""

import datetime
import importlib

import pytest

from adapters.cron_schedule_adapter import CronScheduleAdapter
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.models.database.job import JobStatus, get_job, get_jobs
from app.models.database.prompt_schedule import PromptScheduleRun
from app.schemas.prompt_schedule import PromptScheduleCreate, PromptScheduleUpdate
from app.services.job_queue import JobWorker
from app.services.prompt_scheduler import (
    PromptScheduler,
    create_prompt_schedule,
    enqueue_prompt_schedule_now,
    get_prompt_schedule,
    list_prompt_schedules,
    update_prompt_schedule,
)


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'geist.sqlite3'}",
    )
    engine = configure_database(config)
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=1,
                workspace_key="default",
                username="scheduler",
                name="Scheduler Test",
                email="scheduler@example.com",
                password="",
            )
        )
        session.commit()
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def _request(**overrides):
    values = {
        "name": "Morning briefing",
        "prompt": "Summarize today's priorities.",
        "cron_expression": "0 9 * * *",
        "timezone": "UTC",
    }
    values.update(overrides)
    return PromptScheduleCreate(**values)


def test_create_update_and_list_schedule(sqlite_database):
    now = datetime.datetime(2026, 9, 3, 8, 0)
    schedule = create_prompt_schedule(1, _request(), now=now)

    assert schedule.next_run_at == datetime.datetime(2026, 9, 3, 9, 0)
    assert [item.prompt_schedule_id for item in list_prompt_schedules(1)] == [
        schedule.prompt_schedule_id
    ]

    updated = update_prompt_schedule(
        schedule.prompt_schedule_id,
        1,
        PromptScheduleUpdate(cron_expression="30 10 * * *", enabled=True),
        now=now,
    )
    assert updated is not None
    assert updated.next_run_at == datetime.datetime(2026, 9, 3, 10, 30)


def test_scheduler_enqueues_due_occurrence_once_and_advances(sqlite_database):
    before = datetime.datetime(2026, 9, 3, 8, 0)
    schedule = create_prompt_schedule(1, _request(), now=before)
    due = datetime.datetime(2026, 9, 3, 9, 0)
    scheduler = PromptScheduler()

    assert scheduler.run_once(now=due) == 1
    assert scheduler.run_once(now=due) == 0

    jobs = get_jobs(kind="prompt.inference")
    assert len(jobs) == 1
    assert jobs[0].to_dict()["payload"]["prompt"] == "Summarize today's priorities."
    with SessionLocal() as session:
        occurrence = session.query(PromptScheduleRun).one()
        assert occurrence.scheduled_for == due
    refreshed = get_prompt_schedule(schedule.prompt_schedule_id, 1)
    assert refreshed is not None
    assert refreshed.last_enqueued_at == due
    assert refreshed.next_run_at == datetime.datetime(2026, 9, 4, 9, 0)


def test_missed_occurrences_are_coalesced(sqlite_database):
    schedule = create_prompt_schedule(
        1,
        _request(cron_expression="* * * * *"),
        now=datetime.datetime(2026, 9, 3, 8, 0),
    )
    scheduler = PromptScheduler()

    assert scheduler.run_once(now=datetime.datetime(2026, 9, 3, 9, 0)) == 1
    refreshed = get_prompt_schedule(schedule.prompt_schedule_id, 1)
    assert refreshed is not None
    assert refreshed.next_run_at == datetime.datetime(2026, 9, 3, 9, 1)
    assert len(get_jobs(kind="prompt.inference")) == 1


def test_invalid_persisted_schedule_is_disabled_without_enqueueing(sqlite_database):
    schedule = create_prompt_schedule(
        1,
        _request(),
        now=datetime.datetime(2026, 9, 3, 8, 0),
    )
    with SessionLocal() as session:
        persisted = session.get(type(schedule), schedule.prompt_schedule_id)
        persisted.cron_expression = "invalid"
        session.commit()

    scheduler = PromptScheduler()
    assert scheduler.run_once(now=datetime.datetime(2026, 9, 3, 9, 0)) == 0
    refreshed = get_prompt_schedule(schedule.prompt_schedule_id, 1)
    assert refreshed is not None
    assert refreshed.enabled is False
    assert refreshed.next_run_at is None
    assert get_jobs(kind="prompt.inference") == []


def test_run_now_queues_inference_and_worker_records_result(sqlite_database, monkeypatch):
    schedule = create_prompt_schedule(
        1,
        _request(enabled=False),
        now=datetime.datetime(2026, 9, 3, 8, 0),
    )
    monkeypatch.setattr(
        "app.services.inference.run_prompt_inference",
        lambda **kwargs: {"message": ["done"], "completion_id": "test", "chat_id": 12},
    )
    job = enqueue_prompt_schedule_now(schedule)

    assert JobWorker(include_kinds={"prompt.inference"}).run_once() is True
    finished = get_job(job.job_id)
    assert finished is not None
    assert finished.status == JobStatus.SUCCEEDED.value
    assert finished.to_dict()["result"]["chat_id"] == 12


def test_chat_adapter_creates_schedule_for_current_user(sqlite_database, monkeypatch):
    from app.models.database.geist_user import WorkspaceModel

    monkeypatch.setattr(
        "app.models.database.geist_user.get_default_workspace",
        lambda: WorkspaceModel(1, "default", "Scheduler Test"),
    )

    result = CronScheduleAdapter().create_prompt_schedule(
        name="Chat-created task",
        prompt="Prepare a daily note.",
        cron_expression="15 8 * * *",
        timezone="UTC",
    )

    assert result["name"] == "Chat-created task"
    assert len(list_prompt_schedules(1)) == 1
