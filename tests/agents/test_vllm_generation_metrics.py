"""Generation metrics for the backward-compatible Transformers pipeline shim."""

from unittest.mock import MagicMock, patch

import pytest

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.qwen3_runner import Qwen3Runner


def test_structured_completion_reports_pipeline_throughput():
    runner = Qwen3Runner()
    runner.model_id = "Qwen/Qwen3-8B"
    runner.model = MagicMock()
    runner.device = "cpu"
    runner.tokenizer = MagicMock()
    runner.tokenizer.apply_chat_template.return_value = "rendered prompt"
    runner.tokenizer.encode.side_effect = [list(range(8)), list(range(4))]
    runner._pipeline = MagicMock(
        return_value=[{"generated_text": "rendered promptmeasured response"}]
    )

    with patch(
        "agents.architectures.vllm_runner.time.perf_counter",
        side_effect=[10.0, 10.5],
    ):
        result = runner.complete_messages_with_stats(
            [{"role": "user", "content": "hello"}],
            GenerationConfig(max_tokens=4, temperature=0.0),
        )

    assert result.generation_stats is not None
    assert result.generation_stats.backend == "transformers-pipeline"
    assert result.generation_stats.prompt_tokens == 8
    assert result.generation_stats.completion_tokens == 4
    assert result.generation_stats.total_seconds == pytest.approx(0.5)
    assert result.generation_stats.completion_tps == pytest.approx(8.0)
    assert result.generation_stats.generation_tps is None
