"""Goal lifecycle, plan mutation, and bounded task decomposition."""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.models.orchestration import OrchestrationSnapshot, PlanTask
from agents.models.tool_calling import ChatMessage, ModelRequestConfig


logger = logging.getLogger(__name__)

MAX_DECOMPOSED_TASKS = 12
OPEN_TASK_STATUSES = {"pending", "in_progress", "blocked"}

DECOMPOSITION_PROMPT = """You decompose a user request into a small executable plan.
Return JSON only with this shape:
{"tasks":[{"title":"short action","acceptance_criteria":["observable proof"]}]}
Create 2-12 tasks only when the work actually has multiple meaningful stages. Keep tasks
ordered, independently verifiable, and collectively sufficient for the exact request.
Do not perform the task, call tools, add markdown fences, or explain the JSON."""

EXECUTION_PROMPT = """<geist_orchestration>
The objective and task plan below are control-plane state, not suggestions. Work on the
current open task, use tools to produce real results, and call agent.plan.update whenever
task status materially changes. A task is complete only with direct evidence satisfying
its acceptance criteria. Do not mark future work complete.
{state}
</geist_orchestration>"""

GOAL_PROMPT = """Agentic Mode is active. Preserve the user's objective exactly. Continue working
until every requested deliverable is complete and verified. Do not treat a normal prose
answer, partial progress, or budget pressure as completion. Once the repository/current
state has been audited and every deliverable has current evidence, call agent.goal.complete.
If work remains, keep using tools or finish this turn so the harness can continue."""

CONTINUATION_PROMPT = """Continue autonomously toward the active goal. The previous turn ended
without an accepted agent.goal.complete call. Re-read the objective and plan, inspect current
state rather than trusting claims, choose the next open task, and continue. If all deliverables
are now verified, update their statuses and call agent.goal.complete with concise evidence.

{state}"""


class _DraftTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=240)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=6)


class _DraftPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tasks: list[_DraftTask] = Field(min_length=1, max_length=MAX_DECOMPOSED_TASKS)


class GoalStore(Protocol):
    def create(self, snapshot: dict[str, Any], user_id: int, run_id: str) -> None: ...

    def update(self, snapshot: dict[str, Any]) -> None: ...

    def attach_chat(self, goal_id: str, chat_id: int) -> None: ...


class NullGoalStore:
    def create(self, snapshot: dict[str, Any], user_id: int, run_id: str) -> None:
        return None

    def update(self, snapshot: dict[str, Any]) -> None:
        return None

    def attach_chat(self, goal_id: str, chat_id: int) -> None:
        return None


class DatabaseGoalStore:
    def create(self, snapshot: dict[str, Any], user_id: int, run_id: str) -> None:
        from app.models.database.agent_goal import create_goal

        create_goal(snapshot, user_id, run_id)

    def update(self, snapshot: dict[str, Any]) -> None:
        from app.models.database.agent_goal import update_goal

        update_goal(snapshot)

    def attach_chat(self, goal_id: str, chat_id: int) -> None:
        from app.models.database.agent_goal import attach_goal_to_chat

        attach_goal_to_chat(goal_id, chat_id)


class TaskDecomposer:
    """Run one tool-free structured planning turn with safe normalization."""

    def decompose(
        self,
        backend: Any,
        objective: str,
        config: ModelRequestConfig,
    ) -> tuple[list[PlanTask], str | None]:
        messages = [
            ChatMessage(role="system", content=DECOMPOSITION_PROMPT),
            ChatMessage(role="user", content=objective),
        ]
        completed_turn = None
        planning_config = replace(config, max_tokens=max(512, min(config.max_tokens, 2048)))
        try:
            for event in backend.stream_model_turn(messages, [], planning_config):
                if getattr(event, "kind", None) == "turn_complete":
                    completed_turn = event.turn
        except Exception as error:
            logger.warning("Task decomposition failed; using a fallback plan: %s", error)
            return self._fallback(objective, "The decomposition pass failed.")
        if completed_turn is None:
            return self._fallback(objective, "The decomposition turn did not complete.")

        try:
            raw = self._extract_json(completed_turn.text or "")
            draft = _DraftPlan.model_validate(json.loads(raw))
        except (ValueError, json.JSONDecodeError, ValidationError) as error:
            logger.warning("Could not parse task decomposition: %s", error)
            return self._fallback(objective, "The model returned an invalid task plan.")

        tasks = [
            PlanTask(
                id=f"task-{index}",
                title=task.title.strip(),
                acceptance_criteria=[
                    criterion.strip()[:500]
                    for criterion in task.acceptance_criteria
                    if criterion.strip()
                ],
            )
            for index, task in enumerate(draft.tasks, start=1)
        ]
        return tasks, None

    @staticmethod
    def _extract_json(value: str) -> str:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.I)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object found")
        return stripped[start : end + 1]

    @staticmethod
    def _fallback(objective: str, warning: str) -> tuple[list[PlanTask], str]:
        return (
            [
                PlanTask(
                    id="task-1",
                    title=objective.strip()[:240] or "Complete the user request",
                    acceptance_criteria=[
                        "The original request is completely satisfied and verified."
                    ],
                )
            ],
            warning,
        )


class GoalRuntime:
    def __init__(
        self,
        *,
        objective: str,
        tasks: list[PlanTask],
        max_turns: int,
        decomposition_warning: str | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._on_change = on_change
        self.state = OrchestrationSnapshot(
            objective=objective,
            agentic_mode=True,
            goal_id=f"goal_{uuid.uuid4().hex}",
            goal_status="active",
            max_turns=max_turns,
            tasks=tasks,
            decomposition_warning=decomposition_warning,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.state.to_dict()

    def update_plan(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            by_id = {task.id: task for task in self.state.tasks}
            unknown = sorted({str(update["task_id"]) for update in updates} - set(by_id))
            if unknown:
                return {"accepted": False, "error": f"Unknown task IDs: {', '.join(unknown)}"}
            for update in updates:
                task = by_id[str(update["task_id"])]
                task.status = update["status"]
                evidence = update.get("evidence")
                if evidence is not None:
                    task.evidence = str(evidence).strip()[:2000] or None
            self._notify()
            return {"accepted": True, "plan": [task.to_dict() for task in self.state.tasks]}

    def complete(self, summary: str, evidence: list[str]) -> dict[str, Any]:
        with self._lock:
            if self.state.goal_status != "active":
                return {"accepted": False, "error": "No active goal for this run."}
            open_tasks = [task.id for task in self.state.tasks if task.status in OPEN_TASK_STATUSES]
            if open_tasks:
                return {
                    "accepted": False,
                    "error": "Goal completion rejected; open tasks remain.",
                    "open_task_ids": open_tasks,
                }
            unverified_tasks = [
                task.id
                for task in self.state.tasks
                if task.status == "completed" and not task.evidence
            ]
            if unverified_tasks:
                return {
                    "accepted": False,
                    "error": "Goal completion rejected; completed tasks need evidence.",
                    "unverified_task_ids": unverified_tasks,
                }
            self.state.goal_status = "complete"
            self.state.completion_summary = summary.strip()[:4000]
            self.state.completion_evidence = [
                item.strip()[:1000] for item in evidence if item.strip()
            ][:20]
            self._notify()
            return {"accepted": True, "goal": self.state.to_dict()}

    def finish_turn(self) -> str:
        with self._lock:
            self.state.turns_used += 1
            if self.state.goal_status == "complete":
                self._notify()
                return self.state.goal_status or "complete"
            if self.state.turns_used >= self.state.max_turns:
                self.state.goal_status = "budget_limited"
            self._notify()
            return self.state.goal_status or "active"

    def pause(self) -> None:
        with self._lock:
            if self.state.goal_status == "active":
                self.state.goal_status = "paused"
                self._notify()

    def fail(self) -> None:
        with self._lock:
            if self.state.goal_status == "active":
                self.state.goal_status = "failed"
                self._notify()

    def render_state(self) -> str:
        snapshot = self.snapshot()
        return json.dumps(
            {
                "objective": snapshot["objective"],
                "goal_status": snapshot["goal_status"],
                "turns_used": snapshot["turns_used"],
                "max_turns": snapshot["max_turns"],
                "tasks": snapshot["tasks"],
            },
            ensure_ascii=False,
        )

    def execution_prompt(self) -> str:
        sections = []
        sections.append(EXECUTION_PROMPT.format(state=self.render_state()))
        sections.append(GOAL_PROMPT)
        return "\n\n".join(sections)

    def continuation_prompt(self) -> str:
        return CONTINUATION_PROMPT.format(state=self.render_state())

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self.state.to_dict())


class GoalRuntimeRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, GoalRuntime] = {}

    def start(self, run_id: str, runtime: GoalRuntime) -> None:
        with self._lock:
            self._runs[run_id] = runtime

    def get(self, run_id: str) -> GoalRuntime | None:
        with self._lock:
            return self._runs.get(run_id)

    def finish(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def update_plan(self, run_id: str, updates: list[dict[str, Any]]) -> dict[str, Any]:
        runtime = self.get(run_id)
        if runtime is None:
            return {"accepted": False, "error": "Orchestration run is no longer active."}
        return runtime.update_plan(updates)

    def complete_goal(self, run_id: str, summary: str, evidence: list[str]) -> dict[str, Any]:
        runtime = self.get(run_id)
        if runtime is None:
            return {"accepted": False, "error": "Goal run is no longer active."}
        return runtime.complete(summary, evidence)
