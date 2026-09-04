from agents.models.orchestration import PlanTask
from app.services.goal_runtime import GoalRuntime


def runtime():
    return GoalRuntime(objective="Implement voice notes", tasks=[], max_turns=10)


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
    assert not goal.complete("Done", ["Tests passed"], ["result-1"])["accepted"]
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
