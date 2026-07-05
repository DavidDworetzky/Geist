import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, func

from app.models.database.database import Base, SessionLocal

logger = logging.getLogger(__name__)


class AgentSnapshot(Base):
    '''
    SqlAlchemy class persisting point-in-time snapshots of agent execution state.

    Snapshots are append-only: one row per (agent, step), written after each
    tick and on phase_out, so agent runs can be resumed after a crash,
    inspected tick by tick, or rolled back to an earlier step.
    '''
    __tablename__ = "agent_snapshot"
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    agent_identifier = Column(String, nullable=False, index=True)
    step = Column(Integer, nullable=False, default=0)
    reason = Column(String, default="manual")
    world_context = Column(Text)
    task_context = Column(Text)
    execution_context = Column(Text)
    function_log = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    @staticmethod
    def _loads(serialized: Optional[str]) -> List[Any]:
        if not serialized:
            return []
        try:
            value = json.loads(serialized)
            return value if isinstance(value, list) else []
        except (TypeError, ValueError):
            logger.warning("Failed to deserialize snapshot column; returning empty list")
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "agent_identifier": self.agent_identifier,
            "step": self.step,
            "reason": self.reason,
            "world_context": self._loads(self.world_context),
            "task_context": self._loads(self.task_context),
            "execution_context": self._loads(self.execution_context),
            "function_log": self._loads(self.function_log),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def create_snapshot(
    agent_identifier: str,
    world_context: List[Any],
    task_context: List[Any],
    execution_context: List[Any],
    function_log: Optional[List[Any]] = None,
    reason: str = "manual",
) -> AgentSnapshot:
    """Persist a new snapshot with the next step number for the agent."""
    with SessionLocal() as session:
        max_step = (
            session.query(func.max(AgentSnapshot.step))
            .filter(AgentSnapshot.agent_identifier == agent_identifier)
            .scalar()
        )
        snapshot = AgentSnapshot(
            agent_identifier=agent_identifier,
            step=(max_step or 0) + 1,
            reason=reason,
            world_context=json.dumps(world_context, default=str),
            task_context=json.dumps(task_context, default=str),
            execution_context=json.dumps(execution_context, default=str),
            function_log=json.dumps(function_log or [], default=str),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)
        session.expunge(snapshot)
        return snapshot


def get_snapshots(agent_identifier: str, limit: int = 50) -> List[AgentSnapshot]:
    """Return snapshots for an agent, most recent step first."""
    with SessionLocal() as session:
        snapshots = (
            session.query(AgentSnapshot)
            .filter(AgentSnapshot.agent_identifier == agent_identifier)
            .order_by(AgentSnapshot.step.desc())
            .limit(limit)
            .all()
        )
        for snapshot in snapshots:
            session.expunge(snapshot)
        return snapshots


def get_latest_snapshot(agent_identifier: str) -> Optional[AgentSnapshot]:
    snapshots = get_snapshots(agent_identifier, limit=1)
    return snapshots[0] if snapshots else None


def get_snapshot_by_id(snapshot_id: int) -> Optional[AgentSnapshot]:
    with SessionLocal() as session:
        snapshot = (
            session.query(AgentSnapshot)
            .filter(AgentSnapshot.snapshot_id == snapshot_id)
            .first()
        )
        if snapshot is not None:
            session.expunge(snapshot)
        return snapshot


def prune_snapshots(agent_identifier: str, keep_last: int = 100) -> int:
    """Delete all but the most recent `keep_last` snapshots for an agent."""
    with SessionLocal() as session:
        steps = [
            row[0]
            for row in session.query(AgentSnapshot.step)
            .filter(AgentSnapshot.agent_identifier == agent_identifier)
            .order_by(AgentSnapshot.step.desc())
            .limit(keep_last)
            .all()
        ]
        if not steps:
            return 0
        deleted = (
            session.query(AgentSnapshot)
            .filter(
                AgentSnapshot.agent_identifier == agent_identifier,
                AgentSnapshot.step < min(steps),
            )
            .delete(synchronize_session=False)
        )
        session.commit()
        return deleted
