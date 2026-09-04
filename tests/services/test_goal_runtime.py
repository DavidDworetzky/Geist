from agents.models.orchestration import PlanTask
from agents.models.tool_calling import ModelEvent, ModelRequestConfig, ModelTurn
from app.services.goal_runtime import GoalRuntime, TaskDecomposer


class PlanningBackend:
    def __init__(self, text: str):
        self.text = text
        self.tools = None

    def stream_model_turn(self, messages, tools, config):
        self.tools = tools
        yield ModelEvent.turn_complete(ModelTurn(text=self.text))


def test_decomposer_normalizes_tool_free_json_plan():
    backend = PlanningBackend(
        '{"tasks":[{"title":"Build UI","acceptance_criteria":["UI test passes"]},'
        '{"title":"Verify audio","acceptance_criteria":["Audio smoke passes"]}]}'
    )

    tasks, warning = TaskDecomposer().decompose(
        backend,
        "Add voice notes",
        ModelRequestConfig(max_tokens=4096),
    )

    assert backend.tools == []
    assert warning is None
    assert [task.id for task in tasks] == ["task-1", "task-2"]
    assert tasks[0].acceptance_criteria == ["UI test passes"]


def test_decomposer_falls_back_when_plan_is_invalid():
    tasks, warning = TaskDecomposer().decompose(
        PlanningBackend("not json"),
        "Do the exact request",
        ModelRequestConfig(),
    )

    assert warning == "The model returned an invalid task plan."
    assert len(tasks) == 1
    assert tasks[0].title == "Do the exact request"


def test_goal_completion_requires_closed_tasks_with_evidence():
    runtime = GoalRuntime(
        objective="Ship it",
        tasks=[PlanTask(id="task-1", title="Implement")],
        max_turns=3,
    )

    open_result = runtime.complete("Done", ["tests"])
    assert open_result["accepted"] is False
    assert open_result["open_task_ids"] == ["task-1"]

    runtime.update_plan([{"task_id": "task-1", "status": "completed"}])
    unverified_result = runtime.complete("Done", ["tests"])
    assert unverified_result["accepted"] is False
    assert unverified_result["unverified_task_ids"] == ["task-1"]

    runtime.update_plan([{"task_id": "task-1", "status": "completed", "evidence": "pytest passed"}])
    accepted = runtime.complete("Done", ["pytest passed"])
    assert accepted["accepted"] is True
    assert runtime.finish_turn() == "complete"
    assert runtime.snapshot()["turns_used"] == 1


def test_goal_stops_at_bounded_turn_budget():
    runtime = GoalRuntime(
        objective="Ship it",
        tasks=[PlanTask(id="task-1", title="Implement")],
        max_turns=2,
    )

    assert runtime.finish_turn() == "active"
    assert runtime.finish_turn() == "budget_limited"
    assert runtime.snapshot()["turns_used"] == 2
