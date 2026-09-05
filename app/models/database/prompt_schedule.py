"""Persistence model for recurring prompt schedules."""

import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from app.models.database.database import Base


class PromptSchedule(Base):
    """A user-owned cron schedule that enqueues prompt inference jobs."""

    __tablename__ = "prompt_schedule"

    prompt_schedule_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    cron_expression = Column(String, nullable=False)
    timezone = Column(String, nullable=False, default="UTC")
    enabled = Column(Boolean, nullable=False, default=True)
    inference_config = Column(JSON, nullable=False, default=dict)
    next_run_at = Column(DateTime, index=True)
    last_enqueued_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "prompt_schedule_id": self.prompt_schedule_id,
            "user_id": self.user_id,
            "name": self.name,
            "prompt": self.prompt,
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "enabled": self.enabled,
            "inference_config": self.inference_config or {},
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_enqueued_at": (
                self.last_enqueued_at.isoformat() if self.last_enqueued_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PromptScheduleRun(Base):
    """Ownership and occurrence metadata linking a schedule to a queued job."""

    __tablename__ = "prompt_schedule_run"
    __table_args__ = (
        UniqueConstraint(
            "prompt_schedule_id",
            "scheduled_for",
            name="uq_prompt_schedule_run_occurrence",
        ),
    )

    prompt_schedule_run_id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_schedule_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("job.job_id"), nullable=False, unique=True, index=True)
    trigger_type = Column(String, nullable=False)
    scheduled_for = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
