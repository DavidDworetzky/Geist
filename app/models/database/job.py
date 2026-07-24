import datetime
import json
import logging
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.models.database.database import Base, SessionLocal


logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 30


class JobStatus(Enum):
    """Lifecycle states of a queued job."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(Base):
    '''
    SqlAlchemy class persisting queued background jobs.

    The job table is the whole queue: enqueue is an INSERT, claiming is a
    single transaction that flips the oldest visible queued row to running,
    and retries requeue the row with a later run_after. This works on both
    SQLite (single writer, so a plain transaction is safe) and PostgreSQL
    (claiming adds FOR UPDATE SKIP LOCKED so concurrent workers never
    double-claim).
    '''
    __tablename__ = "job"
    job_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=True, index=True)
    kind = Column(String, nullable=False, index=True)
    dedupe_key = Column(String, nullable=True, index=True)
    payload = Column(Text)
    status = Column(String, nullable=False, default=JobStatus.QUEUED.value, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=DEFAULT_MAX_ATTEMPTS)
    run_after = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    locked_at = Column(DateTime, nullable=True)
    result = Column(Text)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    @staticmethod
    def _loads(serialized: str | None) -> Any:
        if not serialized:
            return None
        try:
            return json.loads(serialized)
        except (TypeError, ValueError):
            logger.warning("Failed to deserialize job column; returning raw value")
            return serialized

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "dedupe_key": self.dedupe_key,
            "payload": self._loads(self.payload) or {},
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "run_after": self.run_after.isoformat() if self.run_after else None,
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "result": self._loads(self.result),
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def enqueue_job(
    kind: str,
    payload: dict[str, Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_seconds: int = 0,
    user_id: int | None = None,
    dedupe_key: str | None = None,
) -> Job:
    """Insert a queued job; delay_seconds hides it from workers until then."""
    run_after = datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds)
    with SessionLocal() as session:
        job = Job(
            user_id=user_id,
            kind=kind,
            dedupe_key=dedupe_key,
            payload=json.dumps(payload or {}),
            status=JobStatus.QUEUED.value,
            max_attempts=max_attempts,
            run_after=run_after,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def enqueue_or_reschedule_job(
    kind: str,
    payload: dict[str, Any],
    *,
    user_id: int,
    dedupe_key: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_seconds: int = 0,
) -> Job:
    """Coalesce a pending logical job while leaving running work untouched."""
    run_after = datetime.datetime.utcnow() + datetime.timedelta(seconds=delay_seconds)
    with SessionLocal() as session:
        job = (
            session.query(Job)
            .filter(
                Job.kind == kind,
                Job.user_id == user_id,
                Job.dedupe_key == dedupe_key,
                Job.status == JobStatus.QUEUED.value,
            )
            .order_by(Job.job_id.desc())
            .first()
        )
        if job is None:
            job = Job(
                user_id=user_id,
                kind=kind,
                dedupe_key=dedupe_key,
                payload=json.dumps(payload),
                status=JobStatus.QUEUED.value,
                max_attempts=max_attempts,
                run_after=run_after,
            )
            session.add(job)
        else:
            job.payload = json.dumps(payload)
            job.run_after = run_after
            job.attempts = 0
            job.error = None
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def claim_next_job() -> Job | None:
    """
    Atomically claim the oldest visible queued job, or None if the queue is
    empty. The claimed row is flipped to running with attempts incremented.
    """
    now = datetime.datetime.utcnow()
    with SessionLocal() as session:
        query = (
            session.query(Job)
            .filter(Job.status == JobStatus.QUEUED.value, Job.run_after <= now)
            .order_by(Job.job_id)
        )
        # PostgreSQL: lock the claimed row and skip rows other workers hold,
        # so multiple workers can share one queue. SQLite has a single writer,
        # so the plain transaction below is already race-free.
        if session.get_bind().dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        job = query.first()
        if job is None:
            return None
        job.status = JobStatus.RUNNING.value
        job.attempts += 1
        job.locked_at = now
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def mark_job_succeeded(job_id: int, result: Any = None) -> Job | None:
    """Record a successful run and its JSON-serializable result."""
    with SessionLocal() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job is None:
            return None
        job.status = JobStatus.SUCCEEDED.value
        job.result = json.dumps(result) if result is not None else None
        job.error = None
        job.locked_at = None
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def mark_job_failed(
    job_id: int,
    error: str,
    retry_backoff_seconds: int = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> Job | None:
    """
    Record a failed run. The job is requeued with exponential backoff until
    attempts reach max_attempts, then left in the terminal failed state.
    """
    with SessionLocal() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job is None:
            return None
        job.error = error
        job.locked_at = None
        if job.attempts >= job.max_attempts:
            job.status = JobStatus.FAILED.value
        else:
            backoff = retry_backoff_seconds * (2 ** (job.attempts - 1))
            job.status = JobStatus.QUEUED.value
            job.run_after = datetime.datetime.utcnow() + datetime.timedelta(seconds=backoff)
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def get_job(job_id: int, user_id: int | None = None) -> Job | None:
    """Fetch a single job by id."""
    with SessionLocal() as session:
        query = session.query(Job).filter_by(job_id=job_id)
        if user_id is not None:
            query = query.filter(Job.user_id == user_id)
        job = query.first()
        if job is not None:
            session.expunge(job)
        return job


def get_jobs(
    status: str | None = None,
    limit: int = 50,
    user_id: int | None = None,
) -> list[Job]:
    """List jobs, newest first, optionally filtered by status."""
    with SessionLocal() as session:
        query = session.query(Job)
        if status:
            query = query.filter(Job.status == status)
        if user_id is not None:
            query = query.filter(Job.user_id == user_id)
        jobs = query.order_by(Job.job_id.desc()).limit(limit).all()
        for job in jobs:
            session.expunge(job)
        return jobs


def recover_stale_jobs(lease_seconds: int = 300) -> int:
    """Requeue work abandoned by a stopped worker."""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=lease_seconds)
    with SessionLocal() as session:
        jobs = (
            session.query(Job)
            .filter(
                Job.status == JobStatus.RUNNING.value,
                Job.locked_at.is_not(None),
                Job.locked_at < cutoff,
            )
            .all()
        )
        for job in jobs:
            job.status = JobStatus.QUEUED.value
            job.locked_at = None
            job.run_after = datetime.datetime.utcnow()
            job.error = "Recovered after worker lease expired"
        session.commit()
        return len(jobs)
