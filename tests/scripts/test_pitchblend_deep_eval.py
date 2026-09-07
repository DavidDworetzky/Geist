from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).parents[2]


def load_deep_eval() -> ModuleType:
    script_path = REPOSITORY_ROOT / "scripts" / "pitchblend_deep_eval.py"
    spec = importlib.util.spec_from_file_location("pitchblend_deep_eval", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deep_eval = load_deep_eval()


def test_fixture_is_small_frontend_feature_with_regression_tests() -> None:
    assert deep_eval.EXPECTED_PASS_RULE == "frontend-feature-low-or-medium"
    assert len(deep_eval.FILES) == 3
    assert all(file["filename"].startswith("client/") for file in deep_eval.FILES)
    assert any(file["filename"].endswith(".test.tsx") for file in deep_eval.FILES)
    evaluator = deep_eval.load_evaluator()
    classifier_input = evaluator.build_classifier_input(
        deep_eval.PULL_REQUEST, deep_eval.FILES
    )
    assert len(classifier_input) < 10_000
    assert "existing settings" in classifier_input
    assert "rolls back a failed save" in classifier_input


def test_workflow_is_manual_read_only_and_uses_eval_secret() -> None:
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/pitchblend-deep-eval.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "secrets.PITCHBLEND_EVAL_OPENAI_API_KEY" in workflow
