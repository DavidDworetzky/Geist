"""AgentRoutine database model: user-defined recurring agent prompts."""

import datetime
from dataclasses import dataclass

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, or_

from app.models.database.database import Base, SessionLocal


class AgentRoutine(Base):
    """A prompt the agent runs on a fixed interval without a user present."""

    __tablename__ = "agent_routine"
    __table_args__ = (Index("ix_agent_routine_next_run", "enabled", "next_run_at"),)

    routine_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False)
    name = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    interval_minutes = Column(Integer, nullable=False, default=60)
    enabled = Column(Boolean, nullable=False, default=True)
    run_once_requested = Column(Boolean, nullable=False, default=False)
    last_run_at = Column(DateTime, nullable=True)
    last_status = Column(String(32), nullable=True)
    last_error = Column(String(500), nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


@dataclass
class AgentRoutineModel:
    routine_id: int
    user_id: int
    name: str
    prompt: str
    interval_minutes: int
    enabled: bool
    last_run_at: datetime.datetime | None
    next_run_at: datetime.datetime | None
    create_date: datetime.datetime
    update_date: datetime.datetime
    run_once_requested: bool = False
    last_status: str | None = None
    last_error: str | None = None


def _to_model(routine: AgentRoutine) -> AgentRoutineModel:
    return AgentRoutineModel(
        routine_id=routine.routine_id,
        user_id=routine.user_id,
        name=routine.name,
        prompt=routine.prompt,
        interval_minutes=routine.interval_minutes,
        enabled=routine.enabled,
        run_once_requested=routine.run_once_requested,
        last_run_at=routine.last_run_at,
        last_status=routine.last_status,
        last_error=routine.last_error,
        next_run_at=routine.next_run_at,
        create_date=routine.create_date,
        update_date=routine.update_date,
    )


def create_routine(
    user_id: int,
    name: str,
    prompt: str,
    interval_minutes: int,
    enabled: bool = True,
) -> AgentRoutineModel:
    with SessionLocal() as session:
        routine = AgentRoutine(
            user_id=user_id,
            name=name,
            prompt=prompt,
            interval_minutes=interval_minutes,
            enabled=enabled,
            # First run happens one interval from creation, not immediately:
            # creating a routine should not instantly fire an unattended run.
            next_run_at=(datetime.datetime.utcnow() + datetime.timedelta(minutes=interval_minutes))
            if enabled
            else None,
        )
        session.add(routine)
        session.commit()
        session.refresh(routine)
        return _to_model(routine)


def list_routines(user_id: int) -> list[AgentRoutineModel]:
    with SessionLocal() as session:
        routines = (
            session.query(AgentRoutine)
            .filter_by(user_id=user_id)
            .order_by(AgentRoutine.routine_id)
            .all()
        )
        return [_to_model(routine) for routine in routines]


def get_routine(routine_id: int) -> AgentRoutineModel | None:
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        return _to_model(routine) if routine else None


def update_routine(routine_id: int, updates: dict) -> AgentRoutineModel | None:
    allowed = {"name", "prompt", "interval_minutes", "enabled"}
    if set(updates) - allowed or any(value is None for value in updates.values()):
        raise ValueError("Invalid routine update fields")
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        if not routine:
            return None
        for field, value in updates.items():
            setattr(routine, field, value)
        if updates.get("enabled") is False:
            routine.next_run_at = None
            routine.run_once_requested = False
        elif (
            routine.enabled
            and not routine.run_once_requested
            and ("interval_minutes" in updates or updates.get("enabled") is True)
        ):
            routine.next_run_at = datetime.datetime.utcnow() + datetime.timedelta(
                minutes=routine.interval_minutes
            )
        routine.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(routine)
        return _to_model(routine)


def delete_routine(routine_id: int) -> bool:
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        if not routine:
            return False
        session.delete(routine)
        session.commit()
        return True


def schedule_routine_once(routine_id: int) -> AgentRoutineModel | None:
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        if routine is None:
            return None
        routine.next_run_at = datetime.datetime.utcnow()
        routine.run_once_requested = True
        session.commit()
        session.refresh(routine)
        return _to_model(routine)


def get_due_routines(
    now: datetime.datetime | None = None, *, limit: int = 8
) -> list[AgentRoutineModel]:
    now = now or datetime.datetime.utcnow()
    with SessionLocal() as session:
        routines = (
            session.query(AgentRoutine)
            .filter(or_(AgentRoutine.enabled.is_(True), AgentRoutine.run_once_requested.is_(True)))
            .filter(AgentRoutine.next_run_at.isnot(None))
            .filter(AgentRoutine.next_run_at <= now)
            .order_by(AgentRoutine.next_run_at)
            .limit(max(1, min(limit, 64)))
            .all()
        )
        return [_to_model(routine) for routine in routines]


def claim_routine_run(
    routine: AgentRoutineModel,
    now: datetime.datetime | None = None,
) -> bool:
    """Atomically claim one due occurrence and schedule its next run.

    The due timestamp and interval are part of the compare-and-swap so another
    scheduler process, or a concurrent user edit, wins cleanly instead of
    launching a duplicate or stale occurrence.
    """
    if routine.next_run_at is None or not (routine.enabled or routine.run_once_requested):
        return False
    now = now or datetime.datetime.utcnow()
    with SessionLocal() as session:
        claimed = (
            session.query(AgentRoutine)
            .filter(AgentRoutine.routine_id == routine.routine_id)
            .filter(AgentRoutine.enabled == routine.enabled)
            .filter(AgentRoutine.run_once_requested == routine.run_once_requested)
            .filter(AgentRoutine.user_id == routine.user_id)
            .filter(AgentRoutine.prompt == routine.prompt)
            .filter(AgentRoutine.name == routine.name)
            .filter(AgentRoutine.next_run_at == routine.next_run_at)
            .filter(AgentRoutine.next_run_at <= now)
            .filter(AgentRoutine.interval_minutes == routine.interval_minutes)
            .update(
                {
                    AgentRoutine.last_run_at: now,
                    AgentRoutine.last_status: "running",
                    AgentRoutine.last_error: None,
                    AgentRoutine.run_once_requested: False,
                    AgentRoutine.next_run_at: (
                        now + datetime.timedelta(minutes=routine.interval_minutes)
                        if routine.enabled
                        else None
                    ),
                },
                synchronize_session=False,
            )
        )
        session.commit()
        if claimed == 1:
            routine.last_run_at = now
        return claimed == 1


def finish_routine_run(routine: AgentRoutineModel, status: str, error: str | None = None) -> None:
    if routine.last_run_at is None:
        return
    with SessionLocal() as session:
        session.query(AgentRoutine).filter_by(
            routine_id=routine.routine_id,
            user_id=routine.user_id,
            last_run_at=routine.last_run_at,
            last_status="running",
        ).update(
            {AgentRoutine.last_status: status, AgentRoutine.last_error: error},
            synchronize_session=False,
        )
        session.commit()
