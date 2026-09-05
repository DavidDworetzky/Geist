"""Focused tests for scheduled prompt inference configuration and caching."""

import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.models.llama_completion import LlamaCompletion, Message
from app.services.inference import clear_inference_runtime_cache, run_prompt_inference


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    clear_inference_runtime_cache()
    yield
    clear_inference_runtime_cache()


def _settings():
    return SimpleNamespace(
        user_settings_id=1,
        user_id=7,
        default_agent_type="online",
        default_local_model="unused-local-model",
        default_online_model="test-online-model",
        default_online_provider="openai",
        default_file_archives=[],
        enable_rag_by_default=False,
        default_max_tokens=321,
        default_temperature=0.75,
        default_top_p=0.8,
        default_frequency_penalty=0.2,
        default_presence_penalty=0.3,
        backup_providers=[],
        ui_preferences={},
        create_date=datetime.datetime(2026, 1, 1),
        update_date=datetime.datetime(2026, 1, 1),
    )


def test_prompt_inference_resolves_overrides_and_reuses_runtime(monkeypatch):
    agent = Mock()
    agent.complete_text.side_effect = [
        LlamaCompletion([Message(role="assistant", content="first")], chat_id=10),
        LlamaCompletion([Message(role="assistant", content="second")], chat_id=11),
    ]
    create_agent = Mock(return_value=agent)
    monkeypatch.setattr(
        "app.services.inference.UserSettingsService.get_or_create_workspace_settings_by_id",
        lambda user_id: _settings(),
    )
    monkeypatch.setattr("app.services.inference.get_default_agent_context", Mock)
    monkeypatch.setattr("app.services.inference.AgentFactory.create_agent", create_agent)

    first = run_prompt_inference(
        "Summarize the day",
        {"max_tokens": 42, "temperature": 0, "top_p": 0},
        user_id=7,
    )
    second = run_prompt_inference(
        "Summarize tomorrow",
        {"max_tokens": 42, "temperature": 0, "top_p": 0},
        user_id=7,
    )

    assert first["message"] == ["first"]
    assert first["chat_id"] == 10
    assert second["message"] == ["second"]
    create_agent.assert_called_once()
    first_call = agent.complete_text.call_args_list[0]
    assert first_call.kwargs["max_tokens"] == 42
    assert first_call.kwargs["temperature"] == 0
    assert first_call.kwargs["top_p"] == 0
    assert first_call.kwargs["frequency_penalty"] == 0.2
    assert first_call.kwargs["presence_penalty"] == 0.3
