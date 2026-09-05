"""CRUD and background scheduling for recurring prompt inference."""

from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.models.database.database import SessionLocal
from app.models.database.job import Job, add_job
from app.models.database.prompt_schedule import PromptSchedule, PromptScheduleRun
from app.schemas.prompt_schedule import PromptScheduleCreate, PromptScheduleUpdate
from app.services.cron_schedule import next_fire_after


logger = logging.getLogger(__name__)
PROMPT_INFERENCE_KIND = "prompt.inference"
PROMPT_SCHEDULE_SOURCE = "cron"
PROMPT_SCHEDULE_MANUAL_SOURCE = "manual"


def _utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _required_int(value: Any, field_name: str) -> int:
    if value is None:
        raise RuntimeError(f"Persisted schedule is missing {field_name}")
    return int(value)


def _job_payload(schedule: PromptSchedule) -> dict[str, Any]:
    return {
        "prompt_schedule_id": _required_int(schedule.prompt_schedule_id, "prompt_schedule_id"),
        "user_id": _required_int(schedule.user_id, "user_id"),
        "prompt": str(schedule.prompt),
        "inference_config": dict(schedule.inference_config or {}),
    }


def create_prompt_schedule(
    user_id: int,
    request: PromptScheduleCreate,
    now: datetime.datetime | None = None,
) -> PromptSchedule:
    """Create a validated schedule and calculate its first occurrence."""
    current_time = now or _utcnow()
    next_run_at = (
        next_fire_after(request.cron_expression, current_time, request.timezone)
        if request.enabled
        else None
    )
    with SessionLocal() as session:
        schedule = PromptSchedule(
            user_id=user_id,
            name=request.name,
            prompt=request.prompt,
            cron_expression=request.cron_expression,
            timezone=request.timezone,
            enabled=request.enabled,
            inference_config=request.inference_config.model_dump(exclude_none=True),
            next_run_at=next_run_at,
        )
        session.add(schedule)
        session.commit()
        session.refresh(schedule)
        session.expunge(schedule)
        return schedule


def get_prompt_schedule(prompt_schedule_id: int, user_id: int) -> PromptSchedule | None:
    """Return one user-owned schedule."""
    with SessionLocal() as session:
        schedule = (
            session.query(PromptSchedule)
            .filter(
                PromptSchedule.prompt_schedule_id == prompt_schedule_id,
                PromptSchedule.user_id == user_id,
            )
            .first()
        )
        if schedule is not None:
            session.expunge(schedule)
        return schedule


def list_prompt_schedules(user_id: int) -> list[PromptSchedule]:
    """List a user's schedules, ordered by next occurrence then ID."""
    with SessionLocal() as session:
        schedules = (
            session.query(PromptSchedule)
            .filter(PromptSchedule.user_id == user_id)
            .order_by(
                PromptSchedule.enabled.desc(),
                PromptSchedule.next_run_at,
                PromptSchedule.prompt_schedule_id,
            )
            .all()
        )
        for schedule in schedules:
            session.expunge(schedule)
        return schedules


def update_prompt_schedule(
    prompt_schedule_id: int,
    user_id: int,
    request: PromptScheduleUpdate,
    now: datetime.datetime | None = None,
) -> PromptSchedule | None:
    """Update an owned schedule and recompute its next run when needed."""
    changes = request.model_dump(exclude_unset=True)
    current_time = now or _utcnow()
    with SessionLocal() as session:
        schedule = (
            session.query(PromptSchedule)
            .filter(
                PromptSchedule.prompt_schedule_id == prompt_schedule_id,
                PromptSchedule.user_id == user_id,
            )
            .first()
        )
        if schedule is None:
            return None

        timing_changed = "cron_expression" in changes or "timezone" in changes
        for field, value in changes.items():
            if field == "inference_config" and value is not None:
                value = {key: item for key, item in value.items() if item is not None}
            setattr(schedule, field, value)

        if not schedule.enabled:
            schedule.next_run_at = None
        elif timing_changed or "enabled" in changes or schedule.next_run_at is None:
            schedule.next_run_at = next_fire_after(
                str(schedule.cron_expression), current_time, str(schedule.timezone)
            )

        schedule.updated_at = current_time
        session.commit()
        session.refresh(schedule)
        session.expunge(schedule)
        return schedule


def delete_prompt_schedule(prompt_schedule_id: int, user_id: int) -> bool:
    """Delete a user-owned schedule while preserving its job history."""
    with SessionLocal() as session:
        schedule = (
            session.query(PromptSchedule)
            .filter(
                PromptSchedule.prompt_schedule_id == prompt_schedule_id,
                PromptSchedule.user_id == user_id,
            )
            .first()
        )
        if schedule is None:
            return False
        session.delete(schedule)
        session.commit()
        return True


def enqueue_prompt_schedule_now(schedule: PromptSchedule) -> Job:
    """Queue an immediate inference using a detached schedule snapshot."""
    scheduled_for = _utcnow()
    with SessionLocal() as session:
        user_id = _required_int(schedule.user_id, "user_id")
        job = add_job(
            session,
            PROMPT_INFERENCE_KIND,
            payload=_job_payload(schedule),
            user_id=user_id,
        )
        session.add(
            PromptScheduleRun(
                prompt_schedule_id=_required_int(schedule.prompt_schedule_id, "prompt_schedule_id"),
                user_id=user_id,
                job_id=_required_int(job.job_id, "job_id"),
                trigger_type=PROMPT_SCHEDULE_MANUAL_SOURCE,
                scheduled_for=scheduled_for,
            )
        )
        session.commit()
        session.refresh(job)
        session.expunge(job)
        return job


def list_prompt_schedule_runs(
    prompt_schedule_id: int,
    user_id: int,
    limit: int = 50,
) -> list[tuple[Job, PromptScheduleRun]]:
    """List automatic and manual jobs belonging to an owned schedule."""
    schedule = get_prompt_schedule(prompt_schedule_id, user_id)
    if schedule is None:
        return []
    with SessionLocal() as session:
        rows = (
            session.query(Job, PromptScheduleRun)
            .join(PromptScheduleRun, PromptScheduleRun.job_id == Job.job_id)
            .filter(
                PromptScheduleRun.prompt_schedule_id == prompt_schedule_id,
                PromptScheduleRun.user_id == user_id,
                Job.kind == PROMPT_INFERENCE_KIND,
            )
            .order_by(Job.job_id.desc())
            .limit(limit)
            .all()
        )
        records: list[tuple[Job, PromptScheduleRun]] = []
        for job, run in rows:
            records.append((job, run))
            session.expunge(job)
            session.expunge(run)
        return records


class PromptScheduler:
    """Polling scheduler that converts due rows into durable jobs."""

    def __init__(self, poll_interval: float = 15.0, batch_size: int = 100):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self, now: datetime.datetime | None = None) -> int:
        """Enqueue one coalesced occurrence for each due schedule."""
        current_time = now or _utcnow()
        enqueued = 0
        with SessionLocal() as session:
            query = (
                session.query(PromptSchedule)
                .filter(
                    PromptSchedule.enabled.is_(True),
                    PromptSchedule.next_run_at.is_not(None),
                    PromptSchedule.next_run_at <= current_time,
                )
                .order_by(PromptSchedule.next_run_at, PromptSchedule.prompt_schedule_id)
                .limit(self.batch_size)
            )
            if session.get_bind().dialect.name == "postgresql":
                query = query.with_for_update(skip_locked=True)
            schedules = query.all()

            for schedule in schedules:
                scheduled_for = schedule.next_run_at
                if scheduled_for is None:
                    continue
                try:
                    next_run_at = next_fire_after(
                        str(schedule.cron_expression),
                        current_time,
                        str(schedule.timezone),
                    )
                except ValueError as error:
                    schedule.enabled = False
                    schedule.next_run_at = None
                    logger.error(
                        "Disabling invalid prompt schedule %s: %s",
                        schedule.prompt_schedule_id,
                        error,
                    )
                    continue

                try:
                    with session.begin_nested():
                        job = add_job(
                            session,
                            kind=PROMPT_INFERENCE_KIND,
                            payload=_job_payload(schedule),
                            run_after=current_time,
                            user_id=_required_int(schedule.user_id, "user_id"),
                        )
                        session.add(
                            PromptScheduleRun(
                                prompt_schedule_id=_required_int(
                                    schedule.prompt_schedule_id, "prompt_schedule_id"
                                ),
                                user_id=_required_int(schedule.user_id, "user_id"),
                                job_id=_required_int(job.job_id, "job_id"),
                                trigger_type=PROMPT_SCHEDULE_SOURCE,
                                scheduled_for=scheduled_for,
                            )
                        )
                    enqueued += 1
                except IntegrityError:
                    logger.info(
                        "Prompt schedule occurrence already queued: schedule_id=%s scheduled_for=%s",
                        schedule.prompt_schedule_id,
                        scheduled_for,
                    )

                schedule.last_enqueued_at = scheduled_for
                schedule.next_run_at = next_run_at
            session.commit()
        return enqueued

    def _run_loop(self) -> None:
        logger.info("Prompt scheduler started")
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Prompt scheduler loop error")
            self._stop_event.wait(self.poll_interval)
        logger.info("Prompt scheduler stopped")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="geist-prompt-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


_scheduler: PromptScheduler | None = None


def start_scheduler() -> PromptScheduler | None:
    """Start the process-wide prompt scheduler unless disabled."""
    global _scheduler
    enabled = os.getenv("GEIST_PROMPT_SCHEDULER_ENABLED", "true").strip().lower()
    if enabled in {"false", "0", "no"}:
        logger.info("Prompt scheduler disabled via GEIST_PROMPT_SCHEDULER_ENABLED")
        return None
    if _scheduler is None:
        try:
            poll_interval = float(os.getenv("GEIST_PROMPT_SCHEDULER_POLL_INTERVAL", "15"))
        except ValueError:
            poll_interval = 15.0
        try:
            batch_size = int(os.getenv("GEIST_PROMPT_SCHEDULER_BATCH_SIZE", "100"))
        except ValueError:
            batch_size = 100
        _scheduler = PromptScheduler(
            poll_interval=max(0.1, poll_interval),
            batch_size=max(1, min(batch_size, 1000)),
        )
    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    """Stop the process-wide scheduler if running."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
