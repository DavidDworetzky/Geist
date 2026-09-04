"""Opt-in smoke test for the packaged llama.cpp runtime and a real GGUF."""

from __future__ import annotations

import os

import pytest

from agents.architectures.llama_server_runner import LlamaServerRunner
from agents.models.tool_calling import ChatMessage, ModelRequestConfig
from app.services.local_models import get_local_model_manager


@pytest.mark.live_model
def test_packaged_llama_server_generates_with_installed_qwen() -> None:
    if os.getenv("GEIST_RUN_LLAMA_SERVER_SMOKE") != "1":
        pytest.skip("set GEIST_RUN_LLAMA_SERVER_SMOKE=1 to exercise real GGUF inference")

    artifact_id = os.getenv("GEIST_LLAMA_SMOKE_ARTIFACT_ID", "qwen3-4b-q4-k-m")
    model_manager = get_local_model_manager()
    artifact = model_manager.get_artifact(artifact_id)
    status = model_manager.status(artifact.id)
    assert status["supported"] is True
    assert (
        status["status"] == "installed"
    ), f"Install {artifact.display_name} before running the llama.cpp smoke test"

    runner = LlamaServerRunner(model_manager=model_manager)
    try:
        runner.load(artifact.model_id, {"artifact_id": artifact.id})
        assert runner.server_manager.public_status()["status"] == "ready"
        events = list(
            runner.stream_model_turn(
                [
                    ChatMessage(role="system", content="Respond directly and briefly."),
                    ChatMessage(
                        role="user",
                        content="/no_think\nReply with the single word READY.",
                    ),
                ],
                [],
                ModelRequestConfig(max_tokens=64, temperature=0.0),
            )
        )
        assistant_text = "".join(event.text or "" for event in events)
        assert assistant_text.strip()
        assert events[-1].kind == "turn_complete"
    finally:
        runner.cleanup()
