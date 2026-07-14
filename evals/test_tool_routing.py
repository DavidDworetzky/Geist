from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


# DeepEval otherwise loads local dotenv files during import. These evals use
# recorded calls only and must never discover credentials or contact a model.
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"


pytestmark = pytest.mark.eval

CASES_PATH = Path(__file__).parent / "cases" / "tool_routing.json"


@pytest.fixture(scope="module")
def deepeval_api() -> SimpleNamespace:
    pytest.importorskip(
        "deepeval",
        reason=(
            "DeepEval routing checks are optional; create the isolated environment "
            "with `conda env create -f eval_environment.yml`."
        ),
    )

    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

    class NoNetworkEvaluationModel(DeepEvalBaseLLM):
        """Fail loudly if a supposedly deterministic metric invokes an LLM."""

        def load_model(self) -> None:
            return None

        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("deterministic routing eval attempted an LLM call")

        async def a_generate(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("deterministic routing eval attempted an LLM call")

        def get_model_name(self) -> str:
            return "no-network-evaluation-model"

    return SimpleNamespace(
        LLMTestCase=LLMTestCase,
        NoNetworkEvaluationModel=NoNetworkEvaluationModel,
        ToolCall=ToolCall,
        ToolCallParams=ToolCallParams,
        ToolCorrectnessMetric=ToolCorrectnessMetric,
    )


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as case_file:
        payload = json.load(case_file)

    assert payload["version"] == 1
    return cast(list[dict[str, Any]], payload["cases"])


ROUTING_CASES = load_cases()


def to_tool_calls(
    tool_records: list[dict[str, Any]],
    deepeval_api: SimpleNamespace,
) -> list[Any]:
    return [
        deepeval_api.ToolCall(
            name=tool_record["name"],
            input_parameters=tool_record["arguments"],
        )
        for tool_record in tool_records
    ]


def score_case(
    case: dict[str, Any],
    deepeval_api: SimpleNamespace,
    called_tools: list[dict[str, Any]] | None = None,
) -> tuple[float, Any]:
    recorded_calls = case["called_tools"] if called_tools is None else called_tools
    test_case = deepeval_api.LLMTestCase(
        input=case["prompt"],
        actual_output=case["actual_output"],
        tools_called=to_tool_calls(recorded_calls, deepeval_api),
        expected_tools=to_tool_calls(case["expected_tools"], deepeval_api),
    )
    metric = deepeval_api.ToolCorrectnessMetric(
        model=deepeval_api.NoNetworkEvaluationModel(),
        evaluation_params=[deepeval_api.ToolCallParams.INPUT_PARAMETERS],
        include_reason=False,
        async_mode=False,
        strict_mode=True,
        should_exact_match=True,
    )
    score = metric.measure(
        test_case,
        _show_indicator=False,
        _log_metric_to_confident=False,
    )
    return score, metric


@pytest.mark.parametrize("case", ROUTING_CASES, ids=lambda case: case["id"])
def test_recorded_tool_route_matches_expected(
    case: dict[str, Any],
    deepeval_api: SimpleNamespace,
) -> None:
    score, metric = score_case(case, deepeval_api)

    called_names = {tool["name"] for tool in case["called_tools"]}
    assert called_names.isdisjoint(case["forbidden_tools"])
    assert score == 1.0
    assert metric.is_successful()


def test_swapped_routes_are_rejected(deepeval_api: SimpleNamespace) -> None:
    tax_case = next(case for case in ROUTING_CASES if case["id"] == "tax_return_document_search")
    news_case = next(case for case in ROUTING_CASES if case["id"] == "todays_news_web_search")

    score, metric = score_case(
        tax_case,
        deepeval_api,
        called_tools=news_case["called_tools"],
    )

    assert score == 0.0
    assert not metric.is_successful()
