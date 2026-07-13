"""Tests for the job queue worker and handler registry."""
import datetime
import importlib
import time

import pytest

from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.job import Job, JobStatus, enqueue_job, get_job
from app.services.job_queue import (
    JobWorker,
    _handlers,
    enqueue,
    register_job_handler,
    registered_kinds,
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


@pytest.fixture()
def temp_handler():
    """Register a throwaway handler kind and clean it up afterwards."""
    registered = []

    def _register(kind, fn):
        register_job_handler(kind, fn)
        registered.append(kind)

    try:
        yield _register
    finally:
        for kind in registered:
            _handlers.pop(kind, None)


def test_worker_dispatches_and_records_result(sqlite_database, temp_handler):
    temp_handler("test.echo", lambda payload: {"echoed": payload["value"]})

    job = enqueue("test.echo", payload={"value": "hi"})
    worker = JobWorker()

    assert worker.run_once() is True
    done = get_job(job.job_id)
    assert done.status == JobStatus.SUCCEEDED.value
    assert done.to_dict()["result"]["echoed"] == "hi"

    # Queue drained.
    assert worker.run_once() is False


def test_worker_retries_handler_exception_then_fails(sqlite_database, temp_handler):
    calls = []

    def flaky(payload):
        calls.append(1)
        raise RuntimeError("kaboom")

    temp_handler("test.flaky", flaky)
    job = enqueue("test.flaky", max_attempts=2)
    worker = JobWorker(retry_backoff_seconds=60)

    assert worker.run_once() is True
    after_first = get_job(job.job_id)
    assert after_first.status == JobStatus.QUEUED.value
    assert "kaboom" in after_first.error

    # Pull the retry forward and let it exhaust max_attempts.
    with SessionLocal() as session:
        row = session.query(Job).filter_by(job_id=job.job_id).first()
        row.run_after = datetime.datetime.utcnow() - datetime.timedelta(seconds=1)
        session.commit()

    assert worker.run_once() is True
    final = get_job(job.job_id)
    assert final.status == JobStatus.FAILED.value
    assert len(calls) == 2


def test_worker_fails_unknown_kind(sqlite_database):
    job = enqueue_job("test.unregistered", max_attempts=1)
    worker = JobWorker()

    assert worker.run_once() is True
    failed = get_job(job.job_id)
    assert failed.status == JobStatus.FAILED.value
    assert "No handler registered" in failed.error


def test_worker_start_stop_processes_queued_job(sqlite_database, temp_handler):
    temp_handler("test.threaded", lambda payload: "done")
    job = enqueue("test.threaded")

    worker = JobWorker(poll_interval=0.05)
    worker.start()
    try:
        deadline = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        while datetime.datetime.utcnow() < deadline:
            if get_job(job.job_id).status == JobStatus.SUCCEEDED.value:
                break
            time.sleep(0.05)
        assert get_job(job.job_id).status == JobStatus.SUCCEEDED.value
    finally:
        worker.stop()


def test_workflow_run_handler_is_registered_and_fails_cleanly(sqlite_database):
    """The built-in workflow.run handler surfaces missing workflows as job failures."""
    assert "workflow.run" in registered_kinds()

    job = enqueue(
        "workflow.run",
        payload={"workflow_id": 999999, "user_id": 1, "input_data": {}},
        max_attempts=1,
    )
    worker = JobWorker()
    assert worker.run_once() is True

    failed = get_job(job.job_id)
    assert failed.status == JobStatus.FAILED.value
    assert "not found" in failed.error
