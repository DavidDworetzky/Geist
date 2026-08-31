from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock


def load_pitchblend() -> ModuleType:
    script_path = (
        Path(__file__).parents[2] / ".github" / "scripts" / "pitchblend_review.py"
    )
    spec = importlib.util.spec_from_file_location("pitchblend_review", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pitchblend = load_pitchblend()


def eligible_pull_request() -> dict:
    return {
        "base": {"ref": "main"},
        "draft": False,
        "mergeable": True,
    }


def passing_checks() -> list[dict]:
    return [{"name": "tests", "status": "completed", "conclusion": "success"}]


def pr_330_files() -> list[dict]:
    return [
        {"filename": "agents/model_catalog.py"},
        {"filename": "docs/agents.md"},
        {"filename": "docs/open_weight_models.md"},
        {"filename": "tests/agents/test_model_catalog.py"},
        {"filename": "tests/agents/test_online_agent.py"},
    ]


def test_exact_review_command() -> None:
    assert pitchblend.COMMAND_PATTERN.fullmatch("@pitchblend-ai review")
    assert pitchblend.COMMAND_PATTERN.fullmatch("@Pitchblend-AI   review")
    assert not pitchblend.COMMAND_PATTERN.fullmatch("please @pitchblend-ai review")
    assert not pitchblend.COMMAND_PATTERN.fullmatch("@pitchblend-ai approve")


def test_gate_accepts_small_localized_fix_with_test() -> None:
    files = [
        {
            "filename": "app/services/titles.py",
            "status": "modified",
            "additions": 12,
            "deletions": 4,
            "patch": "@@ -1 +1 @@",
        },
        {
            "filename": "tests/services/test_titles.py",
            "status": "modified",
            "additions": 15,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        },
    ]

    result = pitchblend.deterministic_gate(
        eligible_pull_request(), files, passing_checks(), []
    )

    assert result.eligible
    assert result.changed_lines == 32


def test_gate_blocks_migration_contract_security_and_dependency_paths() -> None:
    files = [
        {"filename": "migrations/001.sql", "status": "modified", "patch": "x"},
        {"filename": "agents/base_agent.py", "status": "modified", "patch": "x"},
        {"filename": ".github/workflows/ci.yml", "status": "modified", "patch": "x"},
        {"filename": "uv.lock", "status": "modified", "patch": "x"},
        {"filename": "tests/test_policy.py", "status": "modified", "patch": "x"},
    ]

    result = pitchblend.deterministic_gate(
        eligible_pull_request(), files, passing_checks(), []
    )

    assert not result.eligible
    assert any("data migration" in reason for reason in result.reasons)
    assert any("contract" in reason for reason in result.reasons)
    assert any("security" in reason for reason in result.reasons)
    assert any("dependency" in reason for reason in result.reasons)


def test_gate_blocks_size_limits_and_pending_checks() -> None:
    files = [
        {
            "filename": f"src/fix_{index}.py",
            "status": "modified",
            "additions": 50,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        }
        for index in range(21)
    ]
    files[0]["filename"] = "tests/test_fix.py"
    checks = [{"name": "tests", "status": "in_progress", "conclusion": None}]

    result = pitchblend.deterministic_gate(eligible_pull_request(), files, checks, [])

    assert not result.eligible
    assert any("20-file limit" in reason for reason in result.reasons)
    assert any("1000-line limit" in reason for reason in result.reasons)
    assert any("pending or unsuccessful" in reason for reason in result.reasons)


def test_gate_ignores_its_own_pending_status_but_not_other_statuses() -> None:
    files = [
        {
            "filename": "agents/model_catalog.py",
            "status": "modified",
            "additions": 4,
            "deletions": 2,
            "patch": "@@ -1 +1 @@",
        },
        {
            "filename": "tests/agents/test_model_catalog.py",
            "status": "modified",
            "additions": 8,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        },
    ]
    own_status = {
        "context": pitchblend.APPROVAL_GATE_CONTEXT,
        "state": "pending",
    }

    result = pitchblend.deterministic_gate(
        eligible_pull_request(), files, passing_checks(), [own_status]
    )

    assert result.eligible

    unrelated_status = {"context": "external policy", "state": "pending"}
    result = pitchblend.deterministic_gate(
        eligible_pull_request(),
        files,
        passing_checks(),
        [own_status, unrelated_status],
    )

    assert not result.eligible
    assert result.reasons == (
        "commit statuses are pending or unsuccessful: external policy",
    )


def test_model_catalog_scope_matches_pr_330_and_rejects_near_misses() -> None:
    assert pitchblend.is_model_catalog_only_change(pr_330_files())
    assert not pitchblend.is_model_catalog_only_change(
        [{"filename": "tests/agents/test_model_catalog.py"}]
    )
    assert not pitchblend.is_model_catalog_only_change(
        pr_330_files() + [{"filename": "agents/online_agent.py"}]
    )


def base_classification(**overrides: object) -> dict:
    classification = {
        "change_type": "bugfix",
        "complexity": "low",
        "is_frontend": False,
        "data_migration": False,
        "contract_change": False,
        "security_change": False,
        "regression_test_present": True,
        "reason": "Localized correction.",
        "risk_flags": [],
    }
    classification.update(overrides)
    return classification


def test_classifier_matrix_covers_major_decision_boundaries() -> None:
    for change_type in ("bugfix", "feature", "refactor"):
        for complexity in ("low", "medium"):
            classification = base_classification(
                change_type=change_type,
                complexity=complexity,
                is_frontend=True,
                contract_change=True,
                risk_flags=["subjective diagnostic"],
            )
            assert pitchblend.classification_pass_rule(classification) == (
                f"frontend-{change_type}-low-or-medium"
            )

    assert pitchblend.classification_pass_rule(
        base_classification(is_frontend=True, change_type="feature", complexity="high")
    ) is None
    assert pitchblend.classification_pass_rule(
        base_classification(is_frontend=True, change_type="docs", complexity="low")
    ) is None
    assert pitchblend.classification_pass_rule(
        base_classification(complexity="medium")
    ) == "non-frontend-bugfix-low-or-medium"
    assert pitchblend.classification_pass_rule(
        base_classification(complexity="medium", contract_change=True)
    ) is None


def test_catalog_only_matrix_route_accepts_pr_330_classifier_shape() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        contract_change=True,
        reason="Adds a hosted OpenRouter model entry and exposes its metadata.",
    )

    assert pitchblend.classification_pass_rule(classification) is None
    assert (
        pitchblend.classification_pass_rule(
            classification,
            model_catalog_only=pitchblend.is_model_catalog_only_change(pr_330_files()),
        )
        == "model-catalog-low-or-medium"
    )
    approved = pitchblend.format_approved_comment(
        330,
        "8383580aa22b0ff7",
        len(pr_330_files()),
        103,
        classification,
        "model-catalog-low-or-medium",
    )
    assert "Medium-complexity model catalog feature" in approved

    classification["security_change"] = True
    assert (
        pitchblend.classification_pass_rule(
            classification, model_catalog_only=True
        )
        is None
    )


def test_geist_pr_325_classifier_shape_passes_despite_subjective_diagnostics() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        is_frontend=True,
        contract_change=True,
        reason=(
            "Adds local-model selection, shared artifact loading, and immediate "
            "settings persistence."
        ),
        risk_flags=[
            "new_persisted_settings_field",
            "shared_global_artifact_fetch_and_listener_state",
            "async_save_failure_and_rollback_behavior",
            "backend_settings_contract_compatibility",
            "full_stack_verification_incomplete",
        ],
    )

    assert (
        pitchblend.classification_pass_rule(classification)
        == "frontend-feature-low-or-medium"
    )


def test_classifier_schema_is_forward_and_backward_compatible() -> None:
    legacy = base_classification()
    legacy.pop("is_frontend")
    legacy["frontend_only"] = True
    legacy["future_field"] = {"ignored": True}

    normalized = pitchblend.normalize_classification(legacy)

    assert normalized["is_frontend"] is True
    assert normalized["future_field"] == {"ignored": True}
    assert pitchblend.CLASSIFICATION_SCHEMA["additionalProperties"] is True


def test_classifier_uses_frontend_boundary_prompt_and_safe_response_budget() -> None:
    classification = base_classification()
    client = mock.Mock()
    client.request.return_value = {
        "status": "completed",
        "output": [
            {"content": [{"type": "output_text", "text": json.dumps(classification)}]}
        ],
    }
    with mock.patch.object(
        pitchblend, "JsonHttpClient", return_value=client
    ) as client_type:
        result = pitchblend.classify_pull_request(
            "key", "gpt-5.6-luna", "high", {"title": "Fix"}, []
        )

    assert result["is_frontend"] is False
    assert client_type.call_args.kwargs["timeout_seconds"] == 300
    payload = client.request.call_args.args[2]
    assert payload["max_output_tokens"] == 16_000
    assert payload["text"]["format"]["strict"] is False
    assert "is_frontend" in payload["text"]["format"]["schema"]["required"]
    assert "confidence" not in payload["text"]["format"]["schema"]["required"]
    assert "Frontend code remains frontend when it calls an existing API" in payload[
        "instructions"
    ]


def test_comments_include_classifier_json_and_objective_matrix_decision() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        is_frontend=True,
        contract_change=True,
        risk_flags=["Subjective note"],
    )
    matched_rule = pitchblend.classification_pass_rule(classification)
    assert matched_rule == "frontend-feature-low-or-medium"

    approved = pitchblend.format_approved_comment(
        123, "0123456789abcdef", 4, 120, classification, matched_rule
    )
    blocked = pitchblend.format_blocked_comment(124, ["High complexity"], classification)

    assert '"is_frontend": true' in approved
    assert '"approved": true' in approved
    assert '"matched_pass_rule": "frontend-feature-low-or-medium"' in approved
    assert '"approved": false' in blocked
    assert '"matched_pass_rule": null' in blocked


def gate_pull_request() -> dict:
    return {
        "number": 42,
        "head": {"sha": "current-head"},
        "user": {"login": "pull-author"},
        "html_url": "https://github.test/org/repo/pull/42",
    }


def review(login: str, user_type: str = "User", **overrides: object) -> dict:
    value = {
        "id": 1,
        "state": "APPROVED",
        "commit_id": "current-head",
        "user": {"login": login, "type": user_type},
    }
    value.update(overrides)
    return value


def test_approval_gate_accepts_current_pitchblend_or_write_human() -> None:
    github = mock.Mock()
    app_source = pitchblend.approval_gate_source(
        github,
        "org",
        "repo",
        gate_pull_request(),
        [review("pitchblend-ai[bot]", "Bot")],
    )
    assert app_source == "pitchblend"
    github.request.assert_not_called()

    github.request.return_value = {"permission": "write"}
    human_source = pitchblend.approval_gate_source(
        github, "org", "repo", gate_pull_request(), [review("maintainer")]
    )
    assert human_source == "human:maintainer"


def test_approval_gate_rejects_stale_author_read_and_superseded_reviews() -> None:
    github = mock.Mock()
    github.request.return_value = {"permission": "read"}
    reviews = [
        review("pitchblend-ai[bot]", "Bot", commit_id="old-head"),
        review("pull-author"),
        review("reader"),
        review("former-approver", id=2),
        review("former-approver", id=3, state="CHANGES_REQUESTED"),
    ]

    assert (
        pitchblend.approval_gate_source(
            github, "org", "repo", gate_pull_request(), reviews
        )
        is None
    )


def test_publish_approval_gate_sets_one_pending_or_success_context() -> None:
    github = mock.Mock()
    pull_request = gate_pull_request()

    pitchblend.publish_approval_gate(github, "org", "repo", pull_request, None)
    pending_payload = github.request.call_args.args[2]
    assert pending_payload == {
        "state": "pending",
        "context": "Pitchblend approval gate",
        "description": "Waiting for Pitchblend or human approval",
        "target_url": pull_request["html_url"],
    }

    pitchblend.publish_approval_gate(
        github, "org", "repo", pull_request, "human:maintainer"
    )
    success_payload = github.request.call_args.args[2]
    assert success_payload["state"] == "success"
    assert success_payload["description"] == "Approved by maintainer"


def test_review_workflow_recomputes_gate_and_requests_status_write() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/pitchblend-review.yml"
    ).read_text(encoding="utf-8")
    assert "pull_request_review:" in workflow
    assert "types: [submitted, dismissed]" in workflow
    assert "pull_request_target:" in workflow
    assert "types: [opened, reopened, synchronize, ready_for_review]" in workflow
    assert "permission-statuses: write" in workflow


def test_pull_request_review_event_recomputes_current_head_gate() -> None:
    event = {
        "repository": {"full_name": "org/repo"},
        "pull_request": {"number": 42},
    }
    pull_request = gate_pull_request()
    github = mock.Mock()
    github.request.side_effect = [pull_request, None]
    github.paginate.return_value = [review("pitchblend-ai[bot]", "Bot")]
    environment = {
        "GITHUB_EVENT_PATH": "/event.json",
        "GITHUB_EVENT_NAME": "pull_request_review",
        "PITCHBLEND_GITHUB_TOKEN": "token",
    }
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(event))),
        mock.patch.object(pitchblend, "JsonHttpClient", return_value=github),
    ):
        assert pitchblend.main() == 0

    status_call = github.request.call_args_list[-1]
    assert status_call.args[0] == "POST"
    assert status_call.args[1].endswith("/statuses/current-head")
    assert status_call.args[2]["state"] == "success"


def test_classifier_requires_every_non_frontend_low_risk_signal() -> None:
    classification = base_classification()

    assert pitchblend.classification_is_eligible(classification)

    classification["contract_change"] = True
    assert not pitchblend.classification_is_eligible(classification)
