import importlib
import sqlite3

import pytest
from alembic import command
from alembic.migration import MigrationContext
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

from app.database_upgrade import (
    LEGACY_AGENTIC_REVISION,
    _alembic_config,
    _backup_sqlite_database,
    _classify_legacy_schema,
    _validate_legacy_schema,
    upgrade_database,
)
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig


def test_sqlite_backup_is_consistent_and_replaced_atomically(tmp_path):
    database_path = tmp_path / "geist.sqlite3"
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)"))
        connection.execute(text("INSERT INTO sample (value) VALUES ('preserved')"))

    backup_path = _backup_sqlite_database(f"sqlite:///{database_path}", engine)

    assert backup_path == tmp_path / "geist.sqlite3.pre-upgrade.bak"
    assert backup_path.is_file()
    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == ("preserved",)
    assert list(tmp_path.glob("*.backup.tmp")) == []


def test_legacy_schema_validation_accepts_matching_metadata():
    metadata = MetaData()
    Table(
        "sample",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", String),
    )
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    _validate_legacy_schema(metadata, engine)


@pytest.mark.parametrize("versioned", [False, True])
def test_memory_preference_migrates_off_and_persists_without_changing_model(tmp_path, versioned):
    from app.models.database.geist_user import GeistUser
    from app.models.database.user_settings import (
        UserSettings,
        get_user_settings,
        update_user_settings,
    )

    original = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(provider="sqlite", database_url=f"sqlite:///{tmp_path / 'memory.sqlite3'}")
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(
                GeistUser.__table__.insert().values(
                    user_id=51, username="memory-test", workspace_key="local"
                )
            )
            connection.execute(
                UserSettings.__table__.insert().values(
                    user_id=51, default_local_model="preserved/model"
                )
            )
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_allow_system_ram"))
        if versioned:
            command.stamp(_alembic_config(), "f6a9b1c3d5e7")
        upgrade_database()
        assert get_user_settings(51).llama_allow_system_ram is False
        assert (
            update_user_settings(51, {"llama_allow_system_ram": True}).llama_allow_system_ram
            is True
        )
        assert get_user_settings(51).default_local_model == "preserved/model"
        upgrade_database()
        assert get_user_settings(51).llama_allow_system_ram is True
    finally:
        configure_database(original)


def test_legacy_schema_validation_refuses_to_stamp_missing_columns():
    expected_metadata = MetaData()
    Table(
        "sample",
        expected_metadata,
        Column("id", Integer, primary_key=True),
        Column("required_value", String),
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE sample (id INTEGER PRIMARY KEY)"))

    with pytest.raises(RuntimeError, match="missing columns required_value"):
        _validate_legacy_schema(expected_metadata, engine)


def test_unversioned_pre_artifact_schema_is_adopted_at_previous_revision():
    expected_metadata = MetaData()
    Table(
        "user_settings",
        expected_metadata,
        Column("user_settings_id", Integer, primary_key=True),
        Column("default_local_artifact_id", String),
        Column("llama_backend", String),
        Column("llama_gpu_device_ids", String),
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE user_settings (user_settings_id INTEGER PRIMARY KEY)")
        )

    assert _classify_legacy_schema(expected_metadata, engine) == "pre_local_artifact"


def test_unversioned_schema_with_any_additional_gap_is_rejected():
    expected_metadata = MetaData()
    Table(
        "user_settings",
        expected_metadata,
        Column("user_settings_id", Integer, primary_key=True),
        Column("default_local_artifact_id", String),
        Column("llama_backend", String),
        Column("llama_gpu_device_ids", String),
        Column("another_required_column", String),
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE user_settings (user_settings_id INTEGER PRIMARY KEY)")
        )

    with pytest.raises(RuntimeError, match="another_required_column"):
        _classify_legacy_schema(expected_metadata, engine)


def test_unversioned_pre_compute_schema_is_adopted_at_previous_head():
    expected_metadata = MetaData()
    Table(
        "user_settings",
        expected_metadata,
        Column("user_settings_id", Integer, primary_key=True),
        Column("default_local_artifact_id", String),
        Column("llama_backend", String),
        Column("llama_gpu_device_ids", String),
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE user_settings ("
                "user_settings_id INTEGER PRIMARY KEY, "
                "default_local_artifact_id TEXT)"
            )
        )

    assert _classify_legacy_schema(expected_metadata, engine) == "pre_llama_compute"


@pytest.mark.parametrize(
    ("removed_columns", "expected_kind"),
    [
        ({("geist_user", "workspace_key")}, "pre_workspace"),
        (
            {
                ("geist_user", "workspace_key"),
                ("user_settings", "default_local_artifact_id"),
            },
            "pre_local_artifact_and_workspace",
        ),
    ],
)
def test_real_metadata_recognizes_supported_pre_workspace_schemas(removed_columns, expected_kind):
    importlib.import_module("app.models.database")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE mcp_server"))
        if ("geist_user", "workspace_key") in removed_columns:
            connection.execute(text("DROP INDEX ix_geist_user_workspace_key"))
            connection.execute(text("ALTER TABLE geist_user DROP COLUMN workspace_key"))
        if ("user_settings", "default_local_artifact_id") in removed_columns:
            connection.execute(
                text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id")
            )

    assert _classify_legacy_schema(Base.metadata, engine) == expected_kind


def test_real_metadata_recognizes_pre_mcp_schema():
    importlib.import_module("app.models.database")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE mcp_server"))

    assert _classify_legacy_schema(Base.metadata, engine) == "pre_mcp"


@pytest.mark.parametrize("missing_permissions", [False, True])
@pytest.mark.parametrize("missing_mcp", [False, True])
def test_upgrade_adopts_pre_mcp_schema(tmp_path, missing_permissions, missing_mcp):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'pre-mcp.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            if missing_mcp:
                connection.execute(text("DROP TABLE mcp_server"))
            if missing_permissions:
                connection.execute(text("ALTER TABLE user_settings DROP COLUMN agent_permissions"))

        upgrade_database()

        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
            assert connection.execute(text("SELECT COUNT(*) FROM mcp_server")).scalar_one() == 0
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_versioned_compute_parent_upgrade_adds_permissions(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite", database_url=f"sqlite:///{tmp_path / 'versioned-parent.sqlite3'}"
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN agent_permissions"))
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN agentic_mode_enabled"))
            connection.execute(text("DROP TABLE agent_goal"))
            connection.execute(text("DROP TABLE agent_routine"))
        config = _alembic_config()
        command.stamp(config, "b2c5d7e9f1a3")
        upgrade_database()
        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
            connection.execute(text("SELECT agent_permissions FROM user_settings"))
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_upgrade_adopts_combined_unversioned_legacy_schema(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'legacy.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_backend"))
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_gpu_device_ids"))
            connection.execute(text("DROP TABLE mcp_server"))
            connection.execute(text("DROP INDEX ix_geist_user_workspace_key"))
            connection.execute(text("ALTER TABLE geist_user DROP COLUMN workspace_key"))
            connection.execute(
                text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id")
            )
            connection.execute(
                text(
                    "INSERT INTO geist_user "
                    "(user_id, username, name, email, password) "
                    "VALUES (52, 'custom-owner', 'Custom Workspace', "
                    "'custom@example.com', 'legacy-value')"
                )
            )

        upgrade_database()

        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
            assert connection.execute(text("SELECT COUNT(*) FROM mcp_server")).scalar_one() == 0
            row = connection.execute(
                text(
                    "SELECT user_id, workspace_key, email, password "
                    "FROM geist_user WHERE user_id = 52"
                )
            ).one()
            assert row == (52, "default", None, None)
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_one_off_migration_preserves_existing_disabled_routine(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite", database_url=f"sqlite:///{tmp_path / 'routine-migration.sqlite3'}"
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO geist_user (user_id, workspace_key) VALUES (1, 'default')")
            )
            connection.execute(text("ALTER TABLE agent_routine DROP COLUMN run_once_requested"))
            connection.execute(
                text(
                    "INSERT INTO agent_routine (user_id, name, prompt, interval_minutes, enabled, next_run_at) VALUES (1, 'Existing', 'Keep this prompt', 60, 0, '2026-01-01 00:00:00')"
                )
            )
        config = _alembic_config()
        command.stamp(config, ["d4e7f9a1b3c5", LEGACY_AGENTIC_REVISION])
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT name, prompt, enabled, next_run_at, run_once_requested FROM agent_routine"
                )
            ).one() == ("Existing", "Keep this prompt", 0, "2026-01-01 00:00:00", 0)
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_routine_outcome_upgrade_preserves_existing_schedule(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(provider="sqlite", database_url=f"sqlite:///{tmp_path / 'outcomes.sqlite3'}")
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE agent_routine DROP COLUMN last_status"))
            connection.execute(text("ALTER TABLE agent_routine DROP COLUMN last_error"))
            connection.execute(
                text("INSERT INTO geist_user (user_id, workspace_key) VALUES (1, 'default')")
            )
            connection.execute(
                text(
                    "INSERT INTO agent_routine (user_id, name, prompt, interval_minutes, "
                    "enabled, next_run_at, run_once_requested) VALUES "
                    "(1, 'Existing', 'Keep this prompt', 60, 0, '2026-01-01 00:00:00', 1)"
                )
            )
        config = _alembic_config()
        command.stamp(config, ["e5f8a0b2c4d6", LEGACY_AGENTIC_REVISION])
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT name, prompt, enabled, next_run_at, run_once_requested, "
                    "last_status, last_error FROM agent_routine"
                )
            ).one() == ("Existing", "Keep this prompt", 0, "2026-01-01 00:00:00", 1, None, None)
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_bare_alembic_upgrade_seeds_default_workspace(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'bare-migration.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_backend"))
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_gpu_device_ids"))
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN agent_permissions"))
            connection.execute(text("DROP TABLE mcp_server"))
            connection.execute(text("DROP INDEX ix_geist_user_workspace_key"))
            connection.execute(text("ALTER TABLE geist_user DROP COLUMN workspace_key"))

        alembic_config = _alembic_config()
        # This fixture already contains the complete agentic sibling branch.
        command.stamp(alembic_config, LEGACY_AGENTIC_REVISION)
        command.upgrade(alembic_config, "head")

        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT workspace_key, username, name, email, password " "FROM geist_user")
            ).one()
            assert row == ("default", None, "Local Workspace", None, None)
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
            assert connection.execute(text("SELECT COUNT(*) FROM mcp_server")).scalar_one() == 0
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


def test_upgrade_adopts_workspace_schema_missing_only_local_artifact(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'branch-legacy.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE mcp_server"))
            connection.execute(
                text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id")
            )

        upgrade_database()

        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "b9c2d4e6f8a0"
            }
            columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(user_settings)"))
            }
            assert "default_local_artifact_id" in columns
            assert (
                connection.execute(
                    text("SELECT COUNT(*) FROM geist_user WHERE workspace_key = 'default'")
                ).scalar_one()
                == 1
            )
            assert connection.execute(text("SELECT COUNT(*) FROM mcp_server")).scalar_one() == 0
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


@pytest.fixture
def legacy_database(tmp_path):
    from app.models.database import database

    original_config = database.DATABASE_CONFIG
    path = tmp_path / "legacy.sqlite3"
    engine = configure_database(DatabaseConfig(provider="sqlite", database_url=f"sqlite:///{path}"))
    importlib.import_module("app.models.database")
    Base.metadata.create_all(engine)
    try:
        yield path, engine
    finally:
        Session.remove()
        engine.dispose()
        configure_database(original_config)


@pytest.mark.parametrize("missing_artifact", [False, True])
@pytest.mark.parametrize("missing_compute", [False, True])
@pytest.mark.parametrize("main_schema", ["current", "pre_mcp", "pre_workspace"])
def test_legacy_adoption_preserves_data_and_backup(
    legacy_database, missing_artifact, missing_compute, main_schema
):
    from sqlalchemy import inspect

    from app.models.database.agent_goal import AgentGoal
    from app.models.database.geist_user import GeistUser
    from app.models.database.user_settings import UserSettings

    path, engine = legacy_database
    with engine.begin() as connection:
        connection.execute(GeistUser.__table__.insert().values(user_id=52, workspace_key="default"))
        connection.execute(
            UserSettings.__table__.insert().values(
                user_id=52,
                default_local_model="preserved-model",
                agentic_mode_enabled=False,
                agent_permissions={"mode": "require_approval"},
                llama_backend="gpu",
                llama_gpu_device_ids=["gpu-0"],
                default_local_artifact_id="preserved-artifact",
            )
        )
        connection.execute(
            AgentGoal.__table__.insert().values(
                goal_id="preserved-goal",
                user_id=52,
                run_id="preserved-run",
                objective="Preserve my work",
                max_turns=12,
                checkpoint_json='{"turns_used": 3}',
            )
        )
        if missing_artifact:
            connection.execute(
                text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id")
            )
        if missing_compute:
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_backend"))
            connection.execute(text("ALTER TABLE user_settings DROP COLUMN llama_gpu_device_ids"))
        if main_schema != "current":
            connection.execute(text("DROP TABLE mcp_server"))
        if main_schema == "pre_workspace":
            connection.execute(text("DROP INDEX ix_geist_user_workspace_key"))
            connection.execute(text("ALTER TABLE geist_user DROP COLUMN workspace_key"))

    with sqlite3.connect(path) as connection:
        original_dump = list(connection.iterdump())

    upgrade_database()

    _validate_legacy_schema(Base.metadata, engine)
    with engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_heads() == ("b9c2d4e6f8a0",)
        assert connection.execute(text("SELECT user_id, workspace_key FROM geist_user")).one() == (
            52,
            "default",
        )
        settings = connection.execute(UserSettings.__table__.select()).mappings().one()
        assert settings["default_local_model"] == "preserved-model"
        assert settings["agentic_mode_enabled"] is False
        assert settings["agent_permissions"] == {"mode": "require_approval"}
        assert settings["llama_backend"] == (None if missing_compute else "gpu")
        assert settings["llama_gpu_device_ids"] == ([] if missing_compute else ["gpu-0"])
        assert settings["default_local_artifact_id"] == (
            None if missing_artifact else "preserved-artifact"
        )
        goal = connection.execute(AgentGoal.__table__.select()).mappings().one()
        assert goal["objective"] == "Preserve my work"
        assert goal["max_turns"] == 12
        assert goal["checkpoint_json"] == '{"turns_used": 3}'
        budget_column = next(
            column
            for column in inspect(connection).get_columns("agent_goal")
            if column["name"] == "max_turns"
        )
        assert budget_column["default"].strip("'\"") == "48"

    backup_path = path.with_suffix(".sqlite3.pre-upgrade.bak")
    with sqlite3.connect(backup_path) as connection:
        assert list(connection.iterdump()) == original_dump
    backup_bytes = backup_path.read_bytes()
    upgrade_database()
    assert backup_path.read_bytes() == backup_bytes
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM geist_user")).scalar_one() == 1
        assert connection.execute(text("SELECT COUNT(*) FROM agent_goal")).scalar_one() == 1


@pytest.mark.parametrize("unsupported_gap", ["llama_backend", "agent_goal"])
def test_legacy_adoption_rejects_unrecognized_gaps_without_mutating(
    legacy_database, unsupported_gap
):
    path, engine = legacy_database
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id"))
        if unsupported_gap == "agent_goal":
            connection.execute(text("DROP TABLE agent_goal"))
        else:
            connection.execute(text(f"ALTER TABLE user_settings DROP COLUMN {unsupported_gap}"))
    with sqlite3.connect(path) as connection:
        original_dump = list(connection.iterdump())

    with pytest.raises(RuntimeError, match=unsupported_gap):
        upgrade_database()

    with sqlite3.connect(path) as connection:
        assert list(connection.iterdump()) == original_dump
    with sqlite3.connect(path.with_suffix(".sqlite3.pre-upgrade.bak")) as connection:
        assert list(connection.iterdump()) == original_dump


def test_legacy_adoption_aborts_before_writes_if_backup_fails(legacy_database, monkeypatch):
    path, engine = legacy_database
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE user_settings DROP COLUMN default_local_artifact_id"))
    with sqlite3.connect(path) as connection:
        original_dump = list(connection.iterdump())

    def fail_backup(*args):
        raise OSError("Backup destination unavailable")

    monkeypatch.setattr("app.database_upgrade._backup_sqlite_database", fail_backup)
    with pytest.raises(OSError, match="Backup destination unavailable"):
        upgrade_database()
    with sqlite3.connect(path) as connection:
        assert list(connection.iterdump()) == original_dump
