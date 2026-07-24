from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
)

from app.models.database.database import DATABASE_CONFIG, Session, configure_database
from app.models.database.database_config import DatabaseConfig


def _create_legacy_schema(database_url: str) -> None:
    metadata = MetaData()
    Table(
        "geist_user",
        metadata,
        Column("user_id", Integer, primary_key=True, autoincrement=True),
        Column("username", String),
        Column("name", String),
        Column("email", String),
        Column("password", String),
    )
    Table(
        "chat_session",
        metadata,
        Column("chat_session_id", Integer, primary_key=True, autoincrement=True),
        Column("chat_history", String),
        Column("create_date", DateTime),
        Column("update_date", DateTime),
        Column("user_id", Integer, ForeignKey("geist_user.user_id")),
    )
    Table(
        "job",
        metadata,
        Column("job_id", Integer, primary_key=True, autoincrement=True),
        Column("kind", String, nullable=False),
        Column("payload", Text),
        Column("status", String, nullable=False),
        Column("attempts", Integer, nullable=False),
        Column("max_attempts", Integer, nullable=False),
        Column("run_after", DateTime, nullable=False),
        Column("result", Text),
        Column("error", Text),
        Column("created_at", DateTime),
        Column("updated_at", DateTime),
    )
    Table(
        "user_settings",
        metadata,
        Column("user_settings_id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", Integer, ForeignKey("geist_user.user_id"), nullable=False),
    )
    engine = create_engine(database_url)
    metadata.create_all(engine)
    engine.dispose()


def _alembic_config() -> Config:
    config = Config()
    migrations_dir = Path(__file__).resolve().parents[2] / "migrations"
    config.set_main_option("script_location", str(migrations_dir))
    return config


def test_memory_migration_round_trip_from_job_revision(tmp_path):
    original_config = DATABASE_CONFIG
    database_url = f"sqlite:///{tmp_path / 'legacy.sqlite3'}"
    _create_legacy_schema(database_url)
    configure_database(DatabaseConfig(provider="sqlite", database_url=database_url))
    config = _alembic_config()

    try:
        command.stamp(config, "c3a1f5e7d9b2")
        command.upgrade(config, "head")

        engine = create_engine(database_url)
        inspector = inspect(engine)
        assert {"memory_folder", "memory_record", "memory_embedding"}.issubset(
            inspector.get_table_names()
        )
        assert {
            "memory_enabled",
            "memory_mode",
            "folder_id",
            "memory_revision",
            "memory_processed_revision",
            "memory_last_activity_at",
        }.issubset({column["name"] for column in inspector.get_columns("chat_session")})
        assert {"user_id", "dedupe_key", "locked_at"}.issubset(
            {column["name"] for column in inspector.get_columns("job")}
        )
        assert "default_local_artifact_id" in {
            column["name"] for column in inspector.get_columns("user_settings")
        }
        engine.dispose()

        command.downgrade(config, "c3a1f5e7d9b2")
        engine = create_engine(database_url)
        inspector = inspect(engine)
        assert "memory_folder" not in inspector.get_table_names()
        assert "memory_enabled" not in {
            column["name"] for column in inspector.get_columns("chat_session")
        }
        assert "dedupe_key" not in {
            column["name"] for column in inspector.get_columns("job")
        }
        assert "default_local_artifact_id" not in {
            column["name"] for column in inspector.get_columns("user_settings")
        }
        engine.dispose()
    finally:
        Session.remove()
        configure_database(original_config)
