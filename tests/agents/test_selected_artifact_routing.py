"""Selected managed artifacts must determine the concrete local runner."""

from __future__ import annotations

import datetime
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.factory import AgentFactory
from app.models.user_settings import (
    AgentConfigRequest,
    AgentFactoryConfig,
    UserSettingsResponse,
)
from app.services.user_settings_service import UserSettingsService


def _artifact(artifact_id: str, model_id: str, backend: str):
    return SimpleNamespace(id=artifact_id, model_id=model_id, backend=backend)


def _settings(**overrides) -> UserSettingsResponse:
    now = datetime.datetime.now(datetime.UTC)
    values = {
        "user_settings_id": 1,
        "user_id": 1,
        "default_agent_type": "local",
        "default_local_model": "Qwen/Qwen3-4B",
        "default_local_artifact_id": "qwen-gguf",
        "default_online_model": "gpt-4o-mini",
        "default_online_provider": "openai",
        "default_file_archives": [],
        "enable_rag_by_default": True,
        "default_max_tokens": 64,
        "default_temperature": 0.0,
        "default_top_p": 1.0,
        "default_frequency_penalty": 0.0,
        "default_presence_penalty": 0.0,
        "backup_providers": [],
        "ui_preferences": {},
        "create_date": now,
        "update_date": now,
    }
    values.update(overrides)
    return UserSettingsResponse(**values)


def test_qwen_gguf_artifact_routes_to_llama_server_on_apple_silicon():
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("qwen-gguf", "Qwen/Qwen3-4B", "llama_server")

    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
        patch("agents.factory.sys.platform", "darwin"),
        patch("agents.factory.platform.machine", return_value="arm64"),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3-4B",
            device_config={"artifact_id": "qwen-gguf"},
        )

    assert local_agent.call_args.kwargs["runner_type"] == "llama_server"
    assert local_agent.call_args.kwargs["device_config"] == {"artifact_id": "qwen-gguf"}


def test_qwen3_8_gguf_artifact_routes_to_llama_server_on_linux():
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact(
        "qwen3.8-gguf", "Qwen/Qwen3.8-27B", "llama_server"
    )

    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
        patch("agents.factory.sys.platform", "linux"),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3.8-27B",
            device_config={"artifact_id": "qwen3.8-gguf"},
        )

    assert local_agent.call_args.kwargs["runner_type"] == "llama_server"
    assert local_agent.call_args.kwargs["device_config"] == {"artifact_id": "qwen3.8-gguf"}


def test_mlx_artifact_routes_to_mlx_even_when_platform_inference_prefers_server():
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("qwen-mlx", "Qwen/Qwen3.8-27B", "mlx_llama")

    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
        patch("agents.factory.sys.platform", "linux"),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3.8-27B",
            device_config={"artifact_id": "qwen-mlx"},
        )

    assert local_agent.call_args.kwargs["runner_type"] == "mlx_llama"
    assert local_agent.call_args.kwargs["device_config"] == {
        "artifact_id": "qwen-mlx",
        "implementation": "mlx_lm",
    }


@pytest.mark.parametrize(
    ("runner_type", "environment"),
    [
        ("mlx_llama", {"GEIST_LOCAL_RUNNER": ""}),
        (None, {"GEIST_LOCAL_RUNNER": "mlx_llama"}),
    ],
)
def test_incompatible_explicit_runner_and_artifact_fail_closed(runner_type, environment):
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("qwen-gguf", "Qwen/Qwen3-4B", "llama_server")

    with (
        patch.dict(os.environ, environment),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        pytest.raises(ValueError, match="incompatible with selected artifact"),
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3-4B",
            runner_type=runner_type,
            device_config={"artifact_id": "qwen-gguf"},
        )


def test_matching_explicit_runner_remains_authoritative():
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("qwen-gguf", "Qwen/Qwen3-4B", "llama_server")

    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": "transformers"}),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3-4B",
            runner_type="llama_server",
            device_config={"artifact_id": "qwen-gguf"},
        )

    assert local_agent.call_args.kwargs["runner_type"] == "llama_server"


def test_selected_artifact_must_belong_to_requested_model():
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("other-gguf", "other/model", "llama_server")

    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        pytest.raises(ValueError, match="belongs to other/model"),
    ):
        AgentFactory.create_agent(
            "local",
            MagicMock(),
            model="Qwen/Qwen3-4B",
            device_config={"artifact_id": "other-gguf"},
        )


def test_settings_preserve_selected_artifact_for_factory_resolution():
    settings = _settings()
    config = AgentFactoryConfig.from_user_settings(settings)

    assert config.runner_type is None
    assert config.device_config == {"artifact_id": "qwen-gguf"}

    with (
        patch.object(
            UserSettingsService,
            "get_or_create_workspace_settings_by_id",
            return_value=settings,
        ),
        patch("app.services.user_settings_service.AgentFactory.create_agent") as create_agent,
    ):
        UserSettingsService.create_agent_from_workspace_settings(
            1,
            MagicMock(),
            AgentConfigRequest(agent_type="local"),
        )

    assert create_agent.call_args.kwargs["runner_type"] is None
    assert create_agent.call_args.kwargs["device_config"] == {"artifact_id": "qwen-gguf"}


@pytest.mark.parametrize(
    ("settings", "artifact", "expected_engine"),
    [
        (
            _settings(),
            _artifact("qwen-gguf", "Qwen/Qwen3-4B", "llama_server"),
            "llama_server",
        ),
        (
            _settings(
                default_local_model="Qwen/Qwen3.8-27B",
                default_local_artifact_id="qwen-mlx",
            ),
            _artifact("qwen-mlx", "Qwen/Qwen3.8-27B", "mlx_llama"),
            "mlx_llama",
        ),
    ],
)
def test_runtime_info_reports_the_runner_the_factory_will_use(settings, artifact, expected_engine):
    from app import main

    manager = MagicMock()
    manager.get_artifact.return_value = artifact
    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": ""}),
        patch.object(
            main.UserSettingsService,
            "get_default_workspace_settings",
            return_value=settings,
        ),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
    ):
        info = main._configured_inference_info()

    assert info["engine"] == expected_engine


def test_runtime_info_does_not_claim_a_runner_for_an_incompatible_override():
    from app import main

    settings = _settings()
    manager = MagicMock()
    manager.get_artifact.return_value = _artifact("qwen-gguf", "Qwen/Qwen3-4B", "llama_server")
    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": "mlx_llama"}),
        patch.object(
            main.UserSettingsService,
            "get_default_workspace_settings",
            return_value=settings,
        ),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
    ):
        info = main._configured_inference_info()

    assert info["engine"] == "unavailable"
    assert info["acceleration"] is None
