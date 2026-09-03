import asyncio
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from agents.agent_type import AgentType
from agents.model_load_status import ModelLoadStatusRegistry, model_load_status_registry
from app.api.v1.endpoints.models import (
    _initialize_configured_local_runtime,
    download_local_artifact,
    get_model_load_status,
    import_local_artifact,
    start_local_runtime,
)
from app.services.local_models import InsufficientStorageError


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


def test_background_readiness_uses_the_local_agent_enum() -> None:
    model_load_status_registry.mark_loading("test/model", "Loading test model.")
    with patch("app.main.get_active_agent") as get_active_agent:
        _initialize_configured_local_runtime("test/model")

    get_active_agent.assert_called_once_with(AgentType.LOCALAGENT)
    assert model_load_status_registry.get("test/model").state == "ready"


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
    assert status.detail == "Model not installed."
    assert background_tasks.tasks == []


def test_download_endpoint_reports_insufficient_storage() -> None:
    manager = MagicMock()
    manager.request_download.side_effect = InsufficientStorageError(
        "Not enough space to install Test Model. 16.2 GB needed; 512.0 MB available."
    )

    with (
        patch(
            "app.api.v1.endpoints.models.get_local_model_manager",
            return_value=manager,
        ),
        pytest.raises(HTTPException) as raised,
    ):
        download_local_artifact("test-artifact")

    assert raised.value.status_code == 507
    assert "512.0 MB available" in raised.value.detail


def test_download_endpoint_rejects_a_competing_install() -> None:
    manager = MagicMock()
    manager.request_download.side_effect = RuntimeError("Another model is already installing.")

    with (
        patch(
            "app.api.v1.endpoints.models.get_local_model_manager",
            return_value=manager,
        ),
        pytest.raises(HTTPException) as raised,
    ):
        download_local_artifact("test-artifact")

    assert raised.value.status_code == 409
    assert raised.value.detail == "Another model is already installing."


def test_import_endpoint_reports_insufficient_storage() -> None:
    manager = MagicMock()
    manager.import_stream.side_effect = InsufficientStorageError(
        "Not enough space to import this model."
    )
    upload = SimpleNamespace(file=io.BytesIO(b"GGUFmodel"), filename="model.gguf")

    with (
        patch(
            "app.api.v1.endpoints.models.get_local_model_manager",
            return_value=manager,
        ),
        pytest.raises(HTTPException) as raised,
    ):
        import_local_artifact(upload)

    assert raised.value.status_code == 507
    assert raised.value.detail == "Not enough space to import this model."
