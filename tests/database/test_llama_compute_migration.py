from unittest.mock import patch

from migrations.versions import c7d9e1f3a5b8_add_llama_compute_settings as migration


def test_upgrade_only_backfills_missing_llama_gpu_device_ids() -> None:
    with (
        patch.object(migration.op, "add_column"),
        patch.object(migration.op, "execute") as execute,
    ):
        migration.upgrade()

    execute.assert_called_once_with(
        "UPDATE user_settings SET llama_gpu_device_ids = '[]' WHERE llama_gpu_device_ids IS NULL"
    )
