"""Tests for the database-backed job queue model."""
import datetime
import importlib

import pytest

from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.models.database.job import (
    Job,
    JobStatus,
    claim_next_job,
    enqueue_job,
    enqueue_or_reschedule_job,
    get_job,
    get_jobs,
    mark_job_failed,
    mark_job_succeeded,
    recover_stale_jobs,
)


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    sqlite_config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'geist.sqlite3'}",
    )

    engine = configure_database(sqlite_config)

    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def _set_run_after(job_id: int, when: datetime.datetime) -> None:
    with SessionLocal() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        job.run_after = when
        session.commit()


def test_enqueue_and_claim_fifo(sqlite_database):
    first = enqueue_job("noop", payload={"n": 1})
    second = enqueue_job("noop", payload={"n": 2})

    claimed = claim_next_job()
    assert claimed.job_id == first.job_id
    assert claimed.status == JobStatus.RUNNING.value
    assert claimed.attempts == 1
    assert claimed.to_dict()["payload"] == {"n": 1}

    claimed_second = claim_next_job()
    assert claimed_second.job_id == second.job_id


def test_claim_returns_none_when_queue_empty(sqlite_database):
    assert claim_next_job() is None


def test_claim_respects_run_after_delay(sqlite_database):
    delayed = enqueue_job("noop", delay_seconds=3600)
    assert claim_next_job() is None

    _set_run_after(delayed.job_id, datetime.datetime.utcnow() - datetime.timedelta(seconds=1))
    claimed = claim_next_job()
    assert claimed.job_id == delayed.job_id


def test_mark_succeeded_records_result(sqlite_database):
    job = enqueue_job("noop")
    claim_next_job()

    done = mark_job_succeeded(job.job_id, result={"answer": 42})
    assert done.status == JobStatus.SUCCEEDED.value
    assert done.to_dict()["result"] == {"answer": 42}
    assert done.error is None


def test_failed_job_requeues_with_backoff_then_fails(sqlite_database):
    job = enqueue_job("noop", max_attempts=2)

    claim_next_job()
    retried = mark_job_failed(job.job_id, "boom", retry_backoff_seconds=60)
    assert retried.status == JobStatus.QUEUED.value
    assert retried.error == "boom"
    assert retried.run_after > datetime.datetime.utcnow()

    # Not visible until the backoff elapses.
    assert claim_next_job() is None
    _set_run_after(job.job_id, datetime.datetime.utcnow() - datetime.timedelta(seconds=1))

    claimed = claim_next_job()
    assert claimed.attempts == 2
    failed = mark_job_failed(job.job_id, "boom again", retry_backoff_seconds=60)
    assert failed.status == JobStatus.FAILED.value
    assert failed.error == "boom again"

    # Terminal: nothing left to claim.
    assert claim_next_job() is None


def test_get_job_and_list_filtering(sqlite_database):
    queued = enqueue_job("noop", payload={"n": 1})
    finished = enqueue_job("noop", payload={"n": 2})
    # Claim and finish the second job (FIFO claims the first, so claim twice).
    claim_next_job()
    claim_next_job()
    mark_job_succeeded(finished.job_id, result="ok")

    fetched = get_job(queued.job_id)
    assert fetched is not None
    assert fetched.job_id == queued.job_id
    assert get_job(99999) is None

    succeeded = get_jobs(status=JobStatus.SUCCEEDED.value)
    assert [j.job_id for j in succeeded] == [finished.job_id]
    assert len(get_jobs()) == 2


def test_enqueue_or_reschedule_coalesces_latest_payload(sqlite_database):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=7,
                username="queue-user",
                name="Queue User",
                email="queue@example.com",
                password="",
            )
        )
        session.commit()

    first = enqueue_or_reschedule_job(
        "chat.memory.digest",
        {"expected_revision": 1},
        user_id=7,
        dedupe_key="chat-memory:7:9",
        delay_seconds=20,
    )
    second = enqueue_or_reschedule_job(
        "chat.memory.digest",
        {"expected_revision": 2},
        user_id=7,
        dedupe_key="chat-memory:7:9",
        delay_seconds=20,
    )

    assert second.job_id == first.job_id
    assert second.to_dict()["payload"]["expected_revision"] == 2
    assert get_jobs(user_id=8) == []
    assert [job.job_id for job in get_jobs(user_id=7)] == [first.job_id]


def test_recover_stale_running_job(sqlite_database):
    job = enqueue_job("stale")
    claim_next_job()
    with SessionLocal() as session:
        row = session.query(Job).filter_by(job_id=job.job_id).first()
        row.locked_at = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
        session.commit()

    assert recover_stale_jobs(lease_seconds=60) == 1
    recovered = get_job(job.job_id)
    assert recovered.status == JobStatus.QUEUED.value
    assert recovered.locked_at is None
