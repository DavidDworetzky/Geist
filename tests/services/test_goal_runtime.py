import pytest

from agents.models.orchestration import PlanTask
from app.services.goal_runtime import GoalRuntime


def runtime():
    return GoalRuntime(objective="Implement voice notes", tasks=[], max_turns=10)


def test_history_windows_and_execution_prompt_are_bounded():
    goal = runtime()
    for index in range(300):
        goal.observe(str(index), "terminal.run", "succeeded", "x" * 1000)
        goal.add_instruction({"id": str(index), "text": "y" * 20_000, "status": "applied"})
    assert len(goal.state.observations) == 128
    assert len(goal.state.instructions) == 20
    assert len(goal.execution_prompt()) < 26_000
    restored = GoalRuntime(objective="Resume", tasks=[], max_turns=5, saved=goal.checkpoint())
    assert len(restored.state.observations) == 128
    assert len(restored.state.instructions) == 20


def test_failed_instruction_checkpoint_rolls_back_in_memory():
    def fail(snapshot):
        raise RuntimeError("database unavailable")

    goal = GoalRuntime(objective="Build", tasks=[], max_turns=5, on_change=fail)
    goal.state.goal_status = "paused"
    with pytest.raises(RuntimeError, match="database unavailable"):
        goal.add_instruction({"id": "one", "text": "Use local", "status": "queued"})
    assert goal.state.instructions == []
    assert goal.state.goal_status == "paused"


def test_reasoning_task_does_not_need_unrelated_tool_citation():
    goal = runtime()
    goal.observe("read-1", "workspace.read_file", "succeeded", "Read requirements")
    goal.update_plan(
        [
            {
                "task_id": "explain",
                "title": "Explain tradeoffs",
                "status": "completed",
                "evidence": "Explained from the user requirements",
            }
        ]
    )
    assert goal.complete("Explained", ["Read and reasoned"], ["read-1"])["accepted"]


def test_accepted_task_references_survive_observation_window_eviction():
    goal = runtime()
    goal.observe("first", "workspace.read_file", "succeeded", "Read")
    goal.update_plan(
        [
            {
                "task_id": "read",
                "title": "Read",
                "status": "completed",
                "evidence": "Read",
                "evidence_refs": ["first"],
            }
        ]
    )
    for index in range(200):
        goal.observe(str(index), "terminal.run", "succeeded", "Checked")
    assert goal.complete("Done", ["Verified"], ["first"])["accepted"]


def test_executor_can_create_and_revise_plan_after_discovery():
    goal = runtime()
    assert goal.update_plan([{"task_id": "audio", "title": "Add diarization"}])["accepted"]
    assert goal.update_plan(
        [
            {
                "task_id": "audio",
                "title": "Reuse existing diarization",
                "acceptance_criteria": ["Integration passes"],
                "status": "in_progress",
            }
        ]
    )["accepted"]
    assert goal.state.tasks[0].title == "Reuse existing diarization"
    assert goal.state.tasks[0].acceptance_criteria == ["Integration passes"]


def test_completion_rejects_skipping_every_deliverable_and_blank_evidence():
    goal = runtime()
    goal.update_plan(
        [{"task_id": "audio", "title": "Implement", "status": "skipped", "skip_reason": "Too hard"}]
    )
    assert not goal.complete("Done", ["Skipped it"])["accepted"]
    assert not runtime().complete("Done", [" "])["accepted"]


def test_completion_requires_real_observation_references():
    goal = runtime()
    goal.observe("result-1", "terminal.run", "succeeded", "exit 0")
    goal.update_plan(
        [
            {
                "task_id": "audio",
                "title": "Implement",
                "status": "completed",
                "evidence": "Tests passed",
            }
        ]
    )
    assert not goal.complete("Done", ["Tests passed"], ["invented"])["accepted"]
    assert not goal.update_plan([{"task_id": "audio", "evidence_refs": ["invented"]}])["accepted"]
    goal.update_plan([{"task_id": "audio", "evidence_refs": ["result-1"]}])
    assert goal.complete("Done", ["Tests passed"], ["result-1"])["accepted"]


def test_text_only_answers_need_no_plan_or_tool_observations():
    assert runtime().complete("The answer is 42.", ["Derived from the user's question"])["accepted"]


def test_skipped_tasks_require_explanations():
    goal = runtime()
    goal.state.tasks = [
        PlanTask(id="a", title="Required", status="completed", evidence="Answered"),
        PlanTask(id="b", title="Optional", status="skipped"),
    ]
    assert not goal.complete("Answered", ["Answered"])["accepted"]
    goal.update_plan([{"task_id": "b", "skip_reason": "User explicitly removed this requirement"}])
    assert goal.complete("Answered", ["Answered"])["accepted"]


def test_wait_resume_preserves_state_and_extends_budget():
    goal = runtime()
    goal.state.workspace_id = "workspace-1"
    goal.update_plan([{"task_id": "audio", "title": "Build voice notes"}])
    goal.observe("obs-1", "workspace.read_file", "succeeded", "Existing audio found")
    goal.finish_turn()
    assert goal.wait_for_user("Which provider?")["accepted"]
    saved = goal.checkpoint([{"role": "user", "content": "Implement voice notes"}])
    resumed = GoalRuntime(objective="Use local", tasks=[], max_turns=5, saved=saved)
    assert resumed.state.goal_id == goal.state.goal_id
    assert resumed.state.objective == "Implement voice notes"
    assert resumed.state.workspace_id == "workspace-1"
    assert resumed.state.max_turns == 6
    assert resumed.state.goal_status == "active"
    assert resumed.transcript == saved["transcript"]
    assert resumed.state.observations == goal.state.observations


def test_budget_pauses_instead_of_claiming_completion():
    goal = GoalRuntime(objective="Build", tasks=[], max_turns=1)
    assert goal.finish_turn() == "budget_limited"


def test_new_instruction_invalidates_a_stale_completion():
    goal = runtime()
    goal.complete("Done", ["Answer"])
    goal.add_instruction({"id": "i1", "text": "Also add tests", "status": "queued"})
    assert goal.state.goal_status == "active"


def test_checkpoint_ignores_future_task_fields():
    goal = runtime()
    saved = goal.checkpoint()
    saved["tasks"] = [{"id": "a", "title": "Do work", "future_field": "ignored"}]
    restored = GoalRuntime(objective="resume", tasks=[], max_turns=10, saved=saved)
    assert restored.state.tasks[0].title == "Do work"


def test_objective_revision_preserves_identity_progress_and_budget():
    goal = runtime()
    goal.update_plan(
        [{"task_id": "done", "title": "Inspect", "status": "completed", "evidence": "Read files"}]
    )
    goal.observe("read-1", "workspace.read_file", "succeeded", "Read files")
    goal.finish_turn()
    before = goal.snapshot()
    result = goal.update_plan(
        [{"task_id": "cli", "title": "Build a CLI"}], objective="Build a CLI instead of voice notes"
    )
    assert result["accepted"]
    after = goal.snapshot()
    for key in ("goal_id", "workspace_id", "turns_used", "max_turns", "observations"):
        assert after[key] == before[key]
    assert after["tasks"][0] == before["tasks"][0]
    assert after["objective"] == "Build a CLI instead of voice notes"
    assert len(after["tasks"]) == 2


def test_objective_only_update_and_invalid_revision_are_atomic():
    goal = runtime()
    assert goal.update_plan([], objective="Build a CLI")["accepted"]
    before = goal.snapshot()
    assert not goal.update_plan([{"task_id": "no-title"}], objective="Invalid replacement")[
        "accepted"
    ]
    assert not goal.update_plan([], objective="  ")["accepted"]
    assert not goal.update_plan([])["accepted"]
    assert goal.snapshot() == before


def test_existing_plan_tool_exposes_objective_revision_without_new_goal_tool():
    from agents.models.tool_calling import ToolCall, ToolContext
    from app.services.goal_runtime import GoalRuntimeRegistry
    from app.services.tool_registry import build_default_tool_registry

    runs = GoalRuntimeRegistry()
    goal = runtime()
    runs.start("revise-run", goal)
    registry = build_default_tool_registry(runs)
    context = ToolContext(workspace_id=1, chat_id=7, run_id="revise-run", agentic_mode=True)
    call = ToolCall.create("agent.plan.update", {"objective": "Build a CLI"})
    result = registry.execute(call, context)
    assert result.status == "succeeded"
    assert goal.state.objective == "Build a CLI"
    assert "agent.goal.create" not in {tool.name for tool in registry.catalog()}


@pytest.mark.parametrize("status", ["paused", "waiting_for_user", "budget_limited", "failed"])
def test_incomplete_goal_resumes_with_same_identity(status):
    goal = runtime()
    goal.state.goal_status = status
    resumed = GoalRuntime(objective="Follow-up", tasks=[], max_turns=5, saved=goal.checkpoint())
    assert resumed.state.goal_id == goal.state.goal_id
    assert resumed.state.objective == goal.state.objective
    assert resumed.state.goal_status == "active"
