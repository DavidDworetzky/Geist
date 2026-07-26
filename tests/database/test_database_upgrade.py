import sqlite3

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

from app.database_upgrade import (
    _backup_sqlite_database,
    _classify_legacy_schema,
    _validate_legacy_schema,
)


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
