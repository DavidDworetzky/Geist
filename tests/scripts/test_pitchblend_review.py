from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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


def test_classifier_requires_every_low_risk_signal() -> None:
    classification = {
        "change_type": "bugfix",
        "complexity": "low",
        "data_migration": False,
        "contract_change": False,
        "security_change": False,
        "regression_test_present": True,
        "confidence": 0.98,
        "reason": "Localized correction.",
        "risk_flags": [],
    }

    assert pitchblend.classification_is_eligible(classification)

    classification["contract_change"] = True
    assert not pitchblend.classification_is_eligible(classification)
