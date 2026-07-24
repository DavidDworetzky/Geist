"""Alembic-backed database upgrades for native Geist startup."""

from __future__ import annotations

import importlib
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
MIGRATIONS_PATH = PROJECT_ROOT / "migrations"
PRE_LOCAL_ARTIFACT_REVISION = "c3a1f5e7d9b2"
PRE_LOCAL_ARTIFACT_GAP = ("user_settings", "default_local_artifact_id")


def upgrade_database() -> None:
    """Upgrade to the Alembic head, preserving legacy metadata databases.

    The historic initial revision assumes tables created before Alembic was
    adopted. A genuinely empty database is therefore created from current
    metadata once and stamped at head. Non-empty, unversioned legacy databases
    are only stamped after their tables and columns match current metadata.
    Every existing SQLite database is backed up before a migration or stamp.
    """
    from alembic import command
    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import inspect

    from app.models.database.database import DATABASE_CONFIG, Base, Engine
    from app.models.database.database_config import initialize_database

    initialize_database(DATABASE_CONFIG)
    importlib.import_module("app.models.database")

    alembic_config = _alembic_config()
    target_heads = set(ScriptDirectory.from_config(alembic_config).get_heads())
    with Engine.connect() as connection:
        table_names = set(inspect(connection).get_table_names())
        user_table_names = table_names - {"alembic_version"}
        current_heads = set(MigrationContext.configure(connection).get_current_heads())

    if not user_table_names:
        logger.info("Creating a new Geist database from current metadata")
        Base.metadata.create_all(bind=Engine)
        command.stamp(alembic_config, "head")
    elif not current_heads:
        _backup_sqlite_database(DATABASE_CONFIG.database_url, Engine)
        schema_kind = _classify_legacy_schema(Base.metadata, Engine)
        if schema_kind == "pre_local_artifact":
            logger.info(
                "Adopting an unversioned legacy Geist database at %s before upgrading",
                PRE_LOCAL_ARTIFACT_REVISION,
            )
            command.stamp(alembic_config, PRE_LOCAL_ARTIFACT_REVISION)
            command.upgrade(alembic_config, "head")
        else:
            logger.info("Adopting an unversioned legacy Geist database at Alembic head")
            command.stamp(alembic_config, "head")
    else:
        if current_heads != target_heads:
            _backup_sqlite_database(DATABASE_CONFIG.database_url, Engine)
        command.upgrade(alembic_config, "head")

    from scripts.insert_presets import main as insert_presets

    insert_presets(to_commit=True, overwrite=False)


def _alembic_config():
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    config.set_main_option("sqlalchemy.url", "sqlite://")
    return config


def _validate_legacy_schema(metadata, engine) -> None:
    """Refuse to stamp legacy data unless it already matches current models."""
    schema_kind, problems = _inspect_legacy_schema(metadata, engine)
    if schema_kind != "current":
        _raise_legacy_schema_error(problems)


def _classify_legacy_schema(metadata, engine) -> str:
    """Recognize current metadata or the one safe pre-artifact legacy shape."""

    schema_kind, problems = _inspect_legacy_schema(metadata, engine)
    if schema_kind in {"current", "pre_local_artifact"}:
        return schema_kind
    _raise_legacy_schema_error(problems)
    raise AssertionError("unreachable")


def _inspect_legacy_schema(metadata, engine) -> tuple[str, list[str]]:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    problems: list[str] = []
    missing_column_gaps: set[tuple[str, str]] = set()
    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            problems.append(f"missing table {table.name}")
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
        missing_columns = set(table.columns.keys()) - existing_columns
        if missing_columns:
            missing_column_gaps.update((table.name, column) for column in missing_columns)
            problems.append(
                f"table {table.name} missing columns {', '.join(sorted(missing_columns))}"
            )

    if not problems:
        return "current", problems
    if missing_column_gaps == {PRE_LOCAL_ARTIFACT_GAP} and len(problems) == 1:
        return "pre_local_artifact", problems
    return "unknown", problems


def _raise_legacy_schema_error(problems: list[str]) -> None:
    details = "; ".join(problems)
    raise RuntimeError(
        "Unversioned Geist database does not match the current schema; "
        f"the pre-upgrade backup was preserved. {details}"
    )


def _backup_sqlite_database(database_url: str, engine) -> Path | None:
    """Create an atomic, consistent backup beside an existing SQLite file."""
    from sqlalchemy.engine import make_url

    parsed_url = make_url(database_url)
    if parsed_url.get_backend_name() != "sqlite":
        return None
    database_name = parsed_url.database
    if not database_name or database_name == ":memory:":
        return None

    database_path = Path(database_name).expanduser().resolve()
    if not database_path.is_file() or database_path.stat().st_size == 0:
        return None

    engine.dispose()
    backup_path = database_path.with_suffix(f"{database_path.suffix}.pre-upgrade.bak")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.",
        suffix=".backup.tmp",
        dir=database_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        # sqlite3.Connection's context manager commits or rolls back, but does
        # not close the native file handle. Close both handles explicitly so
        # Windows can atomically replace the backup immediately afterward.
        source = sqlite3.connect(database_path)
        destination = sqlite3.connect(temporary_path)
        try:
            source.backup(destination)
            destination.commit()
        finally:
            destination.close()
            source.close()
        os.replace(temporary_path, backup_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    logger.info("Backed up SQLite database to %s", backup_path)
    return backup_path
