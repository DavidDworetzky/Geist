"""Structured state shared by task decomposition and Goal Mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PlanTaskStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
GoalStatus = Literal["active", "complete", "paused", "budget_limited", "failed"]


@dataclass
class PlanTask:
    id: str
    title: str
    acceptance_criteria: list[str] = field(default_factory=list)
    status: PlanTaskStatus = "pending"
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationSnapshot:
    objective: str
    agentic_mode: bool
    goal_id: str | None = None
    goal_status: GoalStatus | None = None
    turns_used: int = 0
    max_turns: int = 0
    tasks: list[PlanTask] = field(default_factory=list)
    completion_summary: str | None = None
    completion_evidence: list[str] = field(default_factory=list)
    decomposition_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tasks": [task.to_dict() for task in self.tasks],
        }
