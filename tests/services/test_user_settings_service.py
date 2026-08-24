from datetime import datetime
from unittest.mock import patch

from app.models.database.user_settings import UserSettingsModel
from app.models.user_settings import UserSettingsUpdate
from app.services.user_settings_service import UserSettingsService


def make_settings_model(local_model: str) -> UserSettingsModel:
    now = datetime.now()
    return UserSettingsModel(
        user_settings_id=1,
        user_id=1,
        default_agent_type="local",
        default_local_model=local_model,
        default_local_artifact_id=None,
        llama_backend=None,
        llama_gpu_device_ids=[],
        default_online_model="gpt-4",
        default_online_provider="openai",
        default_file_archives=[],
        enable_rag_by_default=True,
        default_max_tokens=4096,
        default_temperature=1.0,
        default_top_p=1.0,
        default_frequency_penalty=0.0,
        default_presence_penalty=0.0,
        backup_providers=[],
        ui_preferences={},
        create_date=now,
        update_date=now,
    )


def test_returns_canonical_local_model_for_legacy_settings():
    settings_model = make_settings_model("Meta-Llama-3.1-8B-Instruct")

    with patch(
        "app.services.user_settings_service.get_user_settings",
        return_value=settings_model,
    ):
        response = UserSettingsService.get_user_settings_by_id(1)

    assert response is not None
    assert response.default_local_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_persists_canonical_local_model_for_legacy_update():
    settings_model = make_settings_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    with (
        patch(
            "app.services.user_settings_service.get_user_settings",
            return_value=settings_model,
        ),
        patch(
            "app.services.user_settings_service.update_user_settings",
            return_value=settings_model,
        ) as update_settings,
    ):
        response = UserSettingsService.update_user_settings_by_id(
            1,
            UserSettingsUpdate(default_local_model="Meta-Llama-3.1-8B-Instruct"),
        )

    update_dict = update_settings.call_args.args[1]
    assert update_dict["default_local_model"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert update_dict["default_agent_type"] == "local"
    assert response is not None
    assert response.default_local_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
