from __future__ import annotations

import importlib.util
import json
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


def test_classifier_requires_every_non_frontend_low_risk_signal() -> None:
    classification = base_classification()

    assert pitchblend.classification_is_eligible(classification)

    classification["contract_change"] = True
    assert not pitchblend.classification_is_eligible(classification)
