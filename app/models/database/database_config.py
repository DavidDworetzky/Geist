import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine as SqlAlchemyEngine
from sqlalchemy.engine import make_url


DEFAULT_SQLITE_DATABASE_PATH = "data/geist.sqlite3"
DEFAULT_DATABASE_PROVIDER = "postgresql"


@dataclass(frozen=True)
class DatabaseConfig:
    provider: str
    database_url: str


class DatabaseProvider(Protocol):
    name: str
    aliases: Sequence[str]

    def build_database_url(self, environ: Mapping[str, str]) -> str:
        ...

    def create_engine(self, config: DatabaseConfig) -> SqlAlchemyEngine:
        ...

    def initialize_database(self, config: DatabaseConfig) -> None:
        ...


class PostgreSQLProvider:
    name = "postgresql"
    aliases = ("postgres", "postgresql")

    def build_database_url(self, environ: Mapping[str, str]) -> str:
        db_name = environ.get("POSTGRES_DB", "geist")
        db_user = environ.get("POSTGRES_USER", "geist")
        db_password = environ.get("POSTGRES_PASSWORD", "geist")
        db_host = environ.get("DB_HOST", "db")
        db_port = environ.get("DB_PORT", "5432")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def create_engine(self, config: DatabaseConfig) -> SqlAlchemyEngine:
        return create_engine(config.database_url)

    def initialize_database(self, config: DatabaseConfig) -> None:
        from psycopg2 import connect, sql
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        database_url = make_url(config.database_url)
        database_name = database_url.database or "geist"
        connection = connect(
            dbname="postgres",
            user=database_url.username,
            host=database_url.host,
            port=database_url.port,
            password=database_url.password,
        )
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            if cursor.fetchone():
                print(f"Database '{database_name}' already exists")
                return

            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            print(f"Database '{database_name}' created successfully")
        finally:
            cursor.close()
            connection.close()


class SQLiteProvider:
    name = "sqlite"
    aliases = ("sqlite", "sqlite3")

    def build_database_url(self, environ: Mapping[str, str]) -> str:
        configured_url = environ.get("SQLITE_DATABASE_URL")
        if configured_url:
            return configured_url

        database_path = Path(environ.get("SQLITE_DATABASE_PATH", DEFAULT_SQLITE_DATABASE_PATH))
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        return f"sqlite:///{database_path}"

    def create_engine(self, config: DatabaseConfig) -> SqlAlchemyEngine:
        self._ensure_parent_directory(config)
        engine = create_engine(config.database_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    def initialize_database(self, config: DatabaseConfig) -> None:
        self._ensure_parent_directory(config)

    @staticmethod
    def _ensure_parent_directory(config: DatabaseConfig) -> None:
        database_path = make_url(config.database_url).database
        if not database_path or database_path == ":memory:":
            return
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)


class DatabaseProviderRegistry:
    def __init__(self, providers: Sequence[DatabaseProvider]):
        self._providers = {}
        for provider in providers:
            for name in (provider.name, *provider.aliases):
                self._providers[name.lower()] = provider

    def get(self, name: str) -> DatabaseProvider:
        normalized_name = name.lower()
        if normalized_name not in self._providers:
            available = ", ".join(sorted({provider.name for provider in self._providers.values()}))
            raise ValueError(
                f"Unknown database provider '{name}'. Available providers: {available}"
            )
        return self._providers[normalized_name]

    def infer_from_url(self, database_url: str) -> DatabaseProvider:
        driver_name = make_url(database_url).drivername.split("+", 1)[0]
        return self.get(driver_name)


DEFAULT_DATABASE_PROVIDERS = DatabaseProviderRegistry(
    providers=(PostgreSQLProvider(), SQLiteProvider()),
)


def load_database_config(
    provider: Optional[str] = None,
    database_url: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    registry: DatabaseProviderRegistry = DEFAULT_DATABASE_PROVIDERS,
) -> DatabaseConfig:
    environment = os.environ if environ is None else environ
    selected_provider_name = (
        provider
        or environment.get("GEIST_DATABASE_PROVIDER")
        or environment.get("GEIST_DATABASE_BACKEND")
        or environment.get("DB_BACKEND")
    )
    selected_database_url = database_url or environment.get("SQLALCHEMY_DATABASE_URL")

    if selected_database_url:
        inferred_provider = registry.infer_from_url(selected_database_url)
        selected_provider = (
            registry.get(selected_provider_name) if selected_provider_name else inferred_provider
        )
        if selected_provider.name != inferred_provider.name:
            raise ValueError(
                f"Database provider '{selected_provider.name}' does not match URL "
                f"provider '{inferred_provider.name}'"
            )
    else:
        selected_provider = registry.get(selected_provider_name or DEFAULT_DATABASE_PROVIDER)
        selected_database_url = selected_provider.build_database_url(environment)

    return DatabaseConfig(
        provider=selected_provider.name,
        database_url=selected_database_url,
    )


def create_database_engine(
    config: DatabaseConfig,
    registry: DatabaseProviderRegistry = DEFAULT_DATABASE_PROVIDERS,
) -> SqlAlchemyEngine:
    return registry.get(config.provider).create_engine(config)


def initialize_database(
    config: DatabaseConfig,
    registry: DatabaseProviderRegistry = DEFAULT_DATABASE_PROVIDERS,
) -> None:
    registry.get(config.provider).initialize_database(config)
