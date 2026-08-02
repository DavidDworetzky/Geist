"""AgentRoutine database model: user-defined recurring agent prompts."""
import datetime
from dataclasses import dataclass

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String

from app.models.database.database import Base, SessionLocal


class AgentRoutine(Base):
    """A prompt the agent runs on a fixed interval without a user present."""

    __tablename__ = 'agent_routine'

    routine_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False)
    name = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    interval_minutes = Column(Integer, nullable=False, default=60)
    enabled = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime, nullable=True)
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


def _to_model(routine: AgentRoutine) -> AgentRoutineModel:
    return AgentRoutineModel(
        routine_id=routine.routine_id,
        user_id=routine.user_id,
        name=routine.name,
        prompt=routine.prompt,
        interval_minutes=routine.interval_minutes,
        enabled=routine.enabled,
        last_run_at=routine.last_run_at,
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
            next_run_at=datetime.datetime.utcnow()
            + datetime.timedelta(minutes=interval_minutes),
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
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        if not routine:
            return None
        for field, value in updates.items():
            if hasattr(routine, field):
                setattr(routine, field, value)
        if "interval_minutes" in updates and routine.last_run_at is not None:
            routine.next_run_at = routine.last_run_at + datetime.timedelta(
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


def get_due_routines(now: datetime.datetime | None = None) -> list[AgentRoutineModel]:
    now = now or datetime.datetime.utcnow()
    with SessionLocal() as session:
        routines = (
            session.query(AgentRoutine)
            .filter(AgentRoutine.enabled.is_(True))
            .filter(AgentRoutine.next_run_at.isnot(None))
            .filter(AgentRoutine.next_run_at <= now)
            .order_by(AgentRoutine.next_run_at)
            .all()
        )
        return [_to_model(routine) for routine in routines]


def mark_routine_run(routine_id: int, now: datetime.datetime | None = None) -> None:
    """Record a completed run and schedule the next one."""
    now = now or datetime.datetime.utcnow()
    with SessionLocal() as session:
        routine = session.query(AgentRoutine).filter_by(routine_id=routine_id).first()
        if not routine:
            return
        routine.last_run_at = now
        routine.next_run_at = now + datetime.timedelta(minutes=routine.interval_minutes)
        session.commit()
