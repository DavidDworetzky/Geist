import importlib
import sqlite3

import pytest
from alembic import command
from alembic.migration import MigrationContext
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

from app.database_upgrade import (
    PRE_WORKSPACE_REVISION,
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
        Column("another_required_column", String),
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE user_settings (user_settings_id INTEGER PRIMARY KEY)")
        )

    with pytest.raises(RuntimeError, match="another_required_column"):
        _classify_legacy_schema(expected_metadata, engine)


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


def test_upgrade_adopts_pre_mcp_schema(tmp_path):
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
            connection.execute(text("DROP TABLE mcp_server"))

        upgrade_database()

        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "a4d9c7e2f6b1"
            }
            assert connection.execute(text("SELECT COUNT(*) FROM mcp_server")).scalar_one() == 0
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
                "a4d9c7e2f6b1"
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
            connection.execute(text("DROP TABLE mcp_server"))
            connection.execute(text("DROP INDEX ix_geist_user_workspace_key"))
            connection.execute(text("ALTER TABLE geist_user DROP COLUMN workspace_key"))

        alembic_config = _alembic_config()
        command.stamp(alembic_config, PRE_WORKSPACE_REVISION)
        command.upgrade(alembic_config, "head")

        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT workspace_key, username, name, email, password " "FROM geist_user")
            ).one()
            assert row == ("default", None, "Local Workspace", None, None)
            assert set(MigrationContext.configure(connection).get_current_heads()) == {
                "a4d9c7e2f6b1"
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
                "a4d9c7e2f6b1"
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
