import pytest
from sqlalchemy import create_engine

from app.models.database.database_config import (
    DatabaseConfig,
    DatabaseProviderRegistry,
    create_database_engine,
    load_database_config,
)


class TestDatabaseProvider:
    name = "test"
    aliases = ("testdb",)

    def build_database_url(self, environ):
        return environ.get("TEST_DATABASE_URL", "test://default")

    def create_engine(self, config):
        return create_engine("sqlite:///:memory:")

    def initialize_database(self, config):
        return None


def test_sqlite_is_the_default_provider():
    config = load_database_config(environ={})

    assert config.provider == "sqlite"
    assert config.database_url.startswith("sqlite:///")
    assert config.database_url.endswith("data/geist.sqlite3")


def test_postgresql_can_be_selected_from_environment():
    config = load_database_config(environ={"GEIST_DATABASE_PROVIDER": "postgresql"})

    assert config == DatabaseConfig(
        provider="postgresql",
        database_url="postgresql://geist:geist@db:5432/geist",
    )


def test_sqlite_can_be_selected_from_environment(tmp_path):
    database_path = tmp_path / "configured.sqlite3"

    config = load_database_config(
        environ={
            "GEIST_DATABASE_PROVIDER": "sqlite",
            "SQLITE_DATABASE_PATH": str(database_path),
        }
    )

    assert config.provider == "sqlite"
    assert config.database_url == f"sqlite:///{database_path}"


def test_database_config_can_be_injected_without_environment():
    config = DatabaseConfig(
        provider="sqlite",
        database_url="sqlite:///:memory:",
    )

    engine = create_database_engine(config)
    try:
        assert engine.dialect.name == "sqlite"
    finally:
        engine.dispose()


def test_provider_registry_is_injectable():
    registry = DatabaseProviderRegistry((TestDatabaseProvider(),))

    config = load_database_config(
        provider="testdb",
        environ={"TEST_DATABASE_URL": "test://injected"},
        registry=registry,
    )
    engine = create_database_engine(config, registry=registry)
    try:
        assert config == DatabaseConfig(provider="test", database_url="test://injected")
        assert engine.dialect.name == "sqlite"
    finally:
        engine.dispose()


def test_provider_must_match_explicit_database_url():
    with pytest.raises(ValueError, match="does not match URL provider"):
        load_database_config(
            provider="postgresql",
            database_url="sqlite:///:memory:",
            environ={},
        )
