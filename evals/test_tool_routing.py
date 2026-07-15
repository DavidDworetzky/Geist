from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest


pytestmark = pytest.mark.eval

CASES_PATH = Path(__file__).parent / "cases" / "tool_routing.json"


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as case_file:
        payload = json.load(case_file)

    assert payload["version"] == 1
    return cast(list[dict[str, Any]], payload["cases"])


ROUTING_CASES = load_cases()


def score_case(
    case: dict[str, Any],
    called_tools: list[dict[str, Any]] | None = None,
) -> float:
    recorded_calls = case["called_tools"] if called_tools is None else called_tools
    return 1.0 if recorded_calls == case["expected_tools"] else 0.0


@pytest.mark.parametrize("case", ROUTING_CASES, ids=lambda case: case["id"])
def test_recorded_tool_route_matches_expected(
    case: dict[str, Any],
) -> None:
    score = score_case(case)

    called_names = {tool["name"] for tool in case["called_tools"]}
    assert called_names.isdisjoint(case["forbidden_tools"])
    assert score == 1.0


def test_swapped_routes_are_rejected() -> None:
    tax_case = next(case for case in ROUTING_CASES if case["id"] == "tax_return_document_search")
    news_case = next(case for case in ROUTING_CASES if case["id"] == "todays_news_web_search")

    score = score_case(
        tax_case,
        called_tools=news_case["called_tools"],
    )

    assert score == 0.0
