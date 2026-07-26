from __future__ import annotations

import datetime
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
