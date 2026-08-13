from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models.database.user_settings import UserSettingsModel
from app.models.user_settings import UserSettingsUpdate
from app.services.user_settings_service import UserSettingsService


def _settings(**overrides) -> UserSettingsModel:
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    values = {
        "user_settings_id": 1,
        "user_id": 1,
        "default_agent_type": "local",
        "default_local_model": "old/model",
        "default_local_artifact_id": "old-artifact",
        "llama_backend": None,
        "llama_gpu_device_ids": [],
        "default_online_model": "gpt-4",
        "default_online_provider": "openai",
        "default_file_archives": [],
        "enable_rag_by_default": True,
        "default_max_tokens": 4096,
        "default_temperature": 1.0,
        "default_top_p": 1.0,
        "default_frequency_penalty": 0.0,
        "default_presence_penalty": 0.0,
        "backup_providers": [],
        "ui_preferences": {},
        "create_date": now,
        "update_date": now,
    }
    values.update(overrides)
    return UserSettingsModel(**values)


def test_changing_local_model_clears_stale_artifact_selection():
    current = _settings()
    updated = _settings(
        default_local_model="new/model",
        default_local_artifact_id=None,
    )
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch(
            "app.services.user_settings_service.update_user_settings",
            return_value=updated,
        ) as update,
    ):
        result = UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(default_local_model="new/model"),
        )

    assert result is not None
    update.assert_called_once_with(
        1,
        {
            "default_local_model": "new/model",
            "default_local_artifact_id": None,
            "default_agent_type": "local",
        },
    )


def test_artifact_selection_must_match_model_and_be_installed():
    current = _settings(default_local_artifact_id=None)
    artifact = MagicMock(id="new-artifact", model_id="other/model")
    manager = MagicMock()
    manager.get_artifact.return_value = artifact
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        pytest.raises(ValueError, match="belongs to other/model"),
    ):
        UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(
                default_local_model="new/model",
                default_local_artifact_id="new-artifact",
            ),
        )

    artifact.model_id = "new/model"
    manager.status.return_value = {"status": "not_installed", "supported": True}
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch("app.services.local_models.get_local_model_manager", return_value=manager),
        pytest.raises(ValueError, match="must be installed"),
    ):
        UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(
                default_local_model="new/model",
                default_local_artifact_id="new-artifact",
            ),
        )


def test_explicit_gpu_selection_requires_current_inventory_devices():
    current = _settings(llama_backend="cpu")
    updated = _settings(llama_backend="gpu", llama_gpu_device_ids=["gpu-integrated"])
    inventory = SimpleNamespace(
        managed_by_environment=False,
        available=True,
        devices=[SimpleNamespace(id="gpu-integrated")],
    )
    service = MagicMock()
    service.inventory.return_value = inventory
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch("app.services.user_settings_service.update_user_settings", return_value=updated),
        patch(
            "agents.architectures.llama_devices.get_llama_device_service",
            return_value=service,
        ),
    ):
        result = UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(
                llama_backend="gpu",
                llama_gpu_device_ids=["gpu-integrated"],
            ),
        )

    assert result is not None
    assert result.llama_backend == "gpu"
    assert result.llama_gpu_device_ids == ["gpu-integrated"]


def test_regular_update_cannot_rearm_detection_after_resolution():
    current = _settings(llama_backend="cpu")
    inventory = SimpleNamespace(managed_by_environment=False, available=True, devices=[])
    service = MagicMock()
    service.inventory.return_value = inventory
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch(
            "agents.architectures.llama_devices.get_llama_device_service",
            return_value=service,
        ),
        pytest.raises(ValueError, match="Reset to Defaults"),
    ):
        UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(llama_backend=None, llama_gpu_device_ids=[]),
        )


def test_full_reset_rearms_detection_without_consulting_environment_lock():
    current = _settings(llama_backend="gpu", llama_gpu_device_ids=["gpu-old"])
    updated = _settings(llama_backend=None, llama_gpu_device_ids=[])
    inventory = SimpleNamespace(managed_by_environment=False, available=True, devices=[])
    service = MagicMock()
    service.inventory.return_value = inventory
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=current),
        patch(
            "app.services.user_settings_service.update_user_settings", return_value=updated
        ) as update,
        patch(
            "agents.architectures.llama_devices.get_llama_device_service",
            return_value=service,
        ),
    ):
        result = UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(llama_backend=None, llama_gpu_device_ids=[]),
            allow_llama_redetection=True,
        )

    assert result is not None
    assert result.llama_backend is None
    assert result.llama_gpu_device_ids == []
    assert update.call_args.args[1]["llama_backend"] is None
    service.inventory.assert_not_called()


def test_first_use_persistence_does_not_overwrite_a_resolved_choice():
    resolved = _settings(llama_backend="cpu")
    with (
        patch("app.services.user_settings_service.get_user_settings", return_value=resolved),
        patch.object(
            UserSettingsService,
            "get_user_settings_by_id",
            return_value=MagicMock(llama_backend="cpu"),
        ) as get_response,
        patch(
            "app.services.user_settings_service.update_detected_llama_backend_if_unset"
        ) as update,
    ):
        result = UserSettingsService.persist_detected_llama_backend(1, "gpu", ("gpu-best",))

    assert result is not None
    get_response.assert_called_once_with(1)
    update.assert_not_called()
