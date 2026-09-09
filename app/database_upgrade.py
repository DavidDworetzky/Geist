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
PRE_LOCAL_ARTIFACT_GAP = ("user_settings", "default_local_artifact_id")
LLAMA_COMPUTE_GAPS = {
    ("user_settings", "llama_backend"),
    ("user_settings", "llama_gpu_device_ids"),
}
PRE_WORKSPACE_REVISION = "f5c8a1d3e7b9"
PRE_WORKSPACE_GAP = ("geist_user", "workspace_key")
PRE_MCP_REVISION = "c6d9e2f4a7b1"
PRE_MCP_TABLE = "mcp_server"
# Validate goal tables before adoption; leave the budget data migration pending.
LEGACY_AGENTIC_REVISION = "e9f2a4b6c8d0"
PERMISSIONS_GAP = ("user_settings", "agent_permissions")
MCP_REVISION = "b3e5d7f9a1c3"


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
        _reject_unsupported_legacy_schema(Base.metadata, Engine)
        _complete_legacy_settings_columns(Engine)
        schema_kind = _classify_legacy_schema(Base.metadata, Engine)
        revisions = [LEGACY_AGENTIC_REVISION]
        if schema_kind in {"current", "pre_routines"}:
            revisions.append(MCP_REVISION)
        elif schema_kind == "pre_mcp":
            revisions.append(PRE_MCP_REVISION)
        elif schema_kind != "pre_workspace":
            raise RuntimeError("Legacy settings repair did not produce a supported schema")
        # The agentic branch already includes the shared ancestor. When workspace
        # identity is absent, its migration must still run (including data adoption).
        logger.info("Adopting an unversioned Geist database at %s", revisions)
        command.stamp(alembic_config, revisions)
        command.upgrade(alembic_config, "head")
    else:
        if current_heads != target_heads:
            _backup_sqlite_database(DATABASE_CONFIG.database_url, Engine)
        command.upgrade(alembic_config, "head")

    from app.models.database.geist_user import ensure_default_workspace
    from scripts.insert_presets import main as insert_presets

    ensure_default_workspace()
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
    """Recognize current metadata and the supported legacy upgrade shapes."""

    schema_kind, problems = _inspect_legacy_schema(metadata, engine)
    if schema_kind in {
        "current",
        "pre_routines",
        "pre_permissions",
        "pre_llama_compute",
        "pre_local_artifact",
        "pre_local_artifact_and_mcp",
        "pre_mcp",
        "pre_workspace",
        "pre_local_artifact_and_workspace",
    }:
        return schema_kind
    _raise_legacy_schema_error(problems)
    raise AssertionError("unreachable")


def _reject_unsupported_legacy_schema(metadata, engine) -> None:
    """Validate the entire legacy schema before any additive repair is allowed."""
    _classify_legacy_schema(metadata, engine)


def _complete_legacy_settings_columns(engine) -> None:
    """Fill validated settings gaps without replaying already-present branch tables."""
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with engine.begin() as connection:
        columns = {column["name"] for column in sa.inspect(connection).get_columns("user_settings")}
        operations = Operations(MigrationContext.configure(connection))
        if "agent_permissions" not in columns:
            operations.add_column(
                "user_settings", sa.Column("agent_permissions", sa.JSON(), nullable=True)
            )
        if "default_local_artifact_id" not in columns:
            operations.add_column(
                "user_settings",
                sa.Column("default_local_artifact_id", sa.String(), nullable=True),
            )
        if "llama_backend" not in columns:
            operations.add_column(
                "user_settings", sa.Column("llama_backend", sa.String(), nullable=True)
            )
            operations.add_column(
                "user_settings", sa.Column("llama_gpu_device_ids", sa.JSON(), nullable=True)
            )
            connection.execute(
                sa.text(
                    "UPDATE user_settings SET llama_gpu_device_ids = '[]' "
                    "WHERE llama_gpu_device_ids IS NULL"
                )
            )


def _inspect_legacy_schema(metadata, engine) -> tuple[str, list[str]]:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    problems: list[str] = []
    missing_tables: set[str] = set()
    missing_column_gaps: set[tuple[str, str]] = set()
    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            missing_tables.add(table.name)
            problems.append(f"missing table {table.name}")
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
        missing_columns = set(table.columns.keys()) - existing_columns
        if missing_columns:
            missing_column_gaps.update((table.name, column) for column in missing_columns)
            problems.append(
                f"table {table.name} missing columns {', '.join(sorted(missing_columns))}"
            )

    routine_gap = ("agent_routine", "run_once_requested")
    missing_column_gaps.difference_update(
        {("agent_routine", "last_status"), ("agent_routine", "last_error")}
    )
    routine_missing = "agent_routine" in missing_tables or routine_gap in missing_column_gaps
    # A goal checkpoint implies its ancestor routine schema already exists.
    # Never stamp past a missing ancestor table and silently leave it absent.
    missing_column_gaps.discard(routine_gap)
    if not missing_tables and not missing_column_gaps:
        return "pre_routines" if routine_missing else "current", problems
    if PERMISSIONS_GAP in missing_column_gaps:
        missing_column_gaps.remove(PERMISSIONS_GAP)
        if not missing_tables and not missing_column_gaps:
            return "pre_permissions", problems
    compute_gaps = missing_column_gaps & LLAMA_COMPUTE_GAPS
    if compute_gaps:
        if compute_gaps != LLAMA_COMPUTE_GAPS:
            return "unknown", problems
        missing_column_gaps -= LLAMA_COMPUTE_GAPS
        if not missing_tables and not missing_column_gaps:
            return "pre_llama_compute", problems
    if not missing_tables and missing_column_gaps == {PRE_LOCAL_ARTIFACT_GAP}:
        return "pre_local_artifact", problems
    if missing_tables != {PRE_MCP_TABLE}:
        return "unknown", problems
    if not missing_column_gaps:
        return "pre_mcp", problems
    if missing_column_gaps == {PRE_LOCAL_ARTIFACT_GAP}:
        return "pre_local_artifact_and_mcp", problems
    if missing_column_gaps == {PRE_WORKSPACE_GAP}:
        return "pre_workspace", problems
    if missing_column_gaps == {PRE_LOCAL_ARTIFACT_GAP, PRE_WORKSPACE_GAP}:
        return "pre_local_artifact_and_workspace", problems
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
