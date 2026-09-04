"""Persistent state for one optional autonomous chat goal."""

from __future__ import annotations

import datetime
import json
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.models.database.database import Base, SessionLocal


class AgentGoal(Base):
    __tablename__ = "agent_goal"

    goal_id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, index=True)
    chat_id = Column(
        Integer,
        ForeignKey("chat_session.chat_session_id"),
        nullable=True,
        index=True,
    )
    run_id = Column(String(80), nullable=False, unique=True, index=True)
    objective = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="active")
    plan_json = Column(Text, nullable=False, default="[]")
    turns_used = Column(Integer, nullable=False, default=0)
    max_turns = Column(Integer, nullable=False, default=8)
    completion_summary = Column(Text, nullable=True)
    completion_evidence_json = Column(Text, nullable=False, default="[]")
    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )


def create_goal(snapshot: dict[str, Any], user_id: int, run_id: str) -> None:
    with SessionLocal() as session:
        goal = AgentGoal(
            goal_id=snapshot["goal_id"],
            user_id=user_id,
            run_id=run_id,
            objective=snapshot["objective"],
            status=snapshot["goal_status"],
            plan_json=json.dumps(snapshot["tasks"]),
            turns_used=int(snapshot["turns_used"]),
            max_turns=int(snapshot["max_turns"]),
            completion_summary=snapshot.get("completion_summary"),
            completion_evidence_json=json.dumps(snapshot.get("completion_evidence", [])),
        )
        session.add(goal)
        session.commit()


def update_goal(snapshot: dict[str, Any]) -> None:
    goal_id = snapshot.get("goal_id")
    if not goal_id:
        return
    with SessionLocal() as session:
        goal = session.query(AgentGoal).filter_by(goal_id=goal_id).first()
        if goal is None:
            return
        goal.status = snapshot["goal_status"]
        goal.plan_json = json.dumps(snapshot["tasks"])
        goal.turns_used = int(snapshot["turns_used"])
        goal.max_turns = int(snapshot["max_turns"])
        goal.completion_summary = snapshot.get("completion_summary")
        goal.completion_evidence_json = json.dumps(snapshot.get("completion_evidence", []))
        goal.update_date = datetime.datetime.utcnow()
        session.commit()


def attach_goal_to_chat(goal_id: str, chat_id: int) -> None:
    with SessionLocal() as session:
        goal = session.query(AgentGoal).filter_by(goal_id=goal_id).first()
        if goal is None:
            return
        goal.chat_id = chat_id
        goal.update_date = datetime.datetime.utcnow()
        session.commit()
