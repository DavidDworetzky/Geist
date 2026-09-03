import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from agents.model_load_status import ModelLoadStatusRegistry, model_load_status_registry
from app.api.v1.endpoints.models import get_model_load_status, start_local_runtime


def test_model_load_registry_tracks_lifecycle() -> None:
    registry = ModelLoadStatusRegistry()

    unloaded = registry.get("org/model")
    loading = registry.mark_loading("org/model", "Loading model files.")
    ready = registry.mark_ready("org/model")

    assert unloaded.state == "unloaded"
    assert unloaded.started_at is None
    assert loading.state == "loading"
    assert loading.started_at is not None
    assert ready.state == "ready"
    assert ready.started_at == loading.started_at


def test_model_load_registry_records_failure() -> None:
    registry = ModelLoadStatusRegistry()
    registry.mark_loading("org/model", "Loading model files.")

    failed = registry.mark_failed("org/model", "Model failed to load.")

    assert failed.state == "failed"
    assert failed.detail == "Model failed to load."


def test_model_status_endpoint_reports_process_local_state() -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    unloaded = asyncio.run(get_model_load_status(model_id))
    model_load_status_registry.mark_loading(model_id, "Loading cached weights.")
    loading = asyncio.run(get_model_load_status(model_id))

    assert unloaded.state == "unloaded"
    assert loading.state == "loading"
    assert loading.detail == "Loading cached weights."


def test_model_status_endpoint_rejects_unknown_model() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(get_model_load_status("unknown/model"))

    assert error.value.status_code == 404


def test_remote_model_status_is_always_ready() -> None:
    status = asyncio.run(get_model_load_status("gpt-4"))

    assert status.state == "ready"
    assert status.started_at is None


def test_start_local_runtime_preflights_artifact_before_background_load() -> None:
    manager = MagicMock()
    manager.find_artifact.return_value = SimpleNamespace(
        id="test-artifact",
        display_name="Test Model",
    )
    manager.status.return_value = {
        "status": "installed",
        "path": "/models/test-artifact",
        "supported": True,
    }
    config = SimpleNamespace(
        model="test/model",
        device_config={"artifact_id": "test-artifact"},
    )
    background_tasks = BackgroundTasks()

    with (
        patch(
            "app.services.user_settings_service.UserSettingsService.get_default_workspace_settings",
            return_value=object(),
        ),
        patch(
            "app.models.user_settings.AgentFactoryConfig.from_user_settings",
            return_value=config,
        ),
        patch(
            "app.api.v1.endpoints.models.get_local_model_manager",
            return_value=manager,
        ),
    ):
        status = start_local_runtime(background_tasks)

    assert status.state == "loading"
    assert status.model_id == "test/model"
    assert len(background_tasks.tasks) == 1


def test_start_local_runtime_surfaces_artifact_state_mismatch_immediately() -> None:
    manager = MagicMock()
    manager.find_artifact.return_value = SimpleNamespace(
        id="test-artifact",
        display_name="Test Model",
    )
    manager.status.return_value = {
        "status": "not_installed",
        "path": None,
        "supported": True,
        "error": None,
    }
    config = SimpleNamespace(
        model="test/model",
        device_config={"artifact_id": "test-artifact"},
    )
    background_tasks = BackgroundTasks()

    with (
        patch(
            "app.services.user_settings_service.UserSettingsService.get_default_workspace_settings",
            return_value=object(),
        ),
        patch(
            "app.models.user_settings.AgentFactoryConfig.from_user_settings",
            return_value=config,
        ),
        patch(
            "app.api.v1.endpoints.models.get_local_model_manager",
            return_value=manager,
        ),
    ):
        status = start_local_runtime(background_tasks)

    assert status.state == "failed"
    assert "not downloaded" in status.detail
    assert background_tasks.tasks == []
