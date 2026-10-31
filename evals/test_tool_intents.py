from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest


pytestmark = pytest.mark.eval

CASES_PATH = Path(__file__).parent / "cases" / "tool_intents.json"
VALID_INTENTS = {"answer", "sensitive_answer", "action", "image_generation"}


def load_intent_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as case_file:
        payload = json.load(case_file)
    assert payload["version"] == 1
    return cast(list[dict[str, Any]], payload["cases"])


INTENT_CASES = load_intent_cases()


@pytest.mark.parametrize("case", INTENT_CASES, ids=lambda case: case["id"])
def test_tool_intent_case_contract(case: dict[str, Any]) -> None:
    assert case["prompt"].strip()
    assert case["expected_intent"] in VALID_INTENTS
    assert isinstance(case["expected_needs_retrieval"], bool)


def test_tool_intent_eval_set_covers_every_route() -> None:
    assert {case["expected_intent"] for case in INTENT_CASES} == VALID_INTENTS
