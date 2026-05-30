import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine as SqlAlchemyEngine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

load_dotenv()

DEFAULT_SQLITE_DATABASE_PATH = "data/geist.sqlite3"


def _postgres_database_url() -> str:
    # Use environment variables set in docker-compose.yml.
    db_name = os.getenv("POSTGRES_DB", "geist")
    db_user = os.getenv("POSTGRES_USER", "geist")
    db_password = os.getenv("POSTGRES_PASSWORD", "geist")
    db_host = os.getenv("DB_HOST", "db")
    db_port = os.getenv("DB_PORT", "5432")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def _sqlite_database_url() -> str:
    sqlite_url = os.getenv("SQLITE_DATABASE_URL")
    if sqlite_url:
        return sqlite_url

    sqlite_path = Path(os.getenv("SQLITE_DATABASE_PATH", DEFAULT_SQLITE_DATABASE_PATH))
    if not sqlite_path.is_absolute():
        sqlite_path = Path.cwd() / sqlite_path
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


def _database_config(
    backend: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Tuple[str, str]:
    explicit_database_url = database_url or os.getenv("SQLALCHEMY_DATABASE_URL")
    if explicit_database_url:
        explicit_backend = explicit_database_url.split(":", 1)[0]
        if explicit_backend.startswith("sqlite"):
            explicit_backend = "sqlite"
        return explicit_database_url, explicit_backend

    selected_backend = (backend or os.getenv("GEIST_DATABASE_BACKEND") or os.getenv("DB_BACKEND") or "postgresql").lower()
    if selected_backend in {"sqlite", "sqlite3"}:
        return _sqlite_database_url(), "sqlite"

    return _postgres_database_url(), "postgresql"


def _create_engine(database_url: str, backend: str) -> SqlAlchemyEngine:
    connect_args = {"check_same_thread": False} if backend == "sqlite" else {}
    engine = create_engine(database_url, connect_args=connect_args)

    if backend == "sqlite":
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


SQLALCHEMY_DATABASE_URL, DATABASE_BACKEND = _database_config()

Engine = _create_engine(SQLALCHEMY_DATABASE_URL, DATABASE_BACKEND)
_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
SessionLocal = _session_factory
Session = scoped_session(_session_factory)

Base = declarative_base()


def configure_database(
    database_url: Optional[str] = None,
    backend: Optional[str] = None,
) -> SqlAlchemyEngine:
    """
    Rebind the process-wide SQLAlchemy session factory.

    This is primarily used by tests and alternate storage configurations. Keeping
    the same session factory lets existing model modules continue using their
    imported SessionLocal object after the engine changes.
    """
    global SQLALCHEMY_DATABASE_URL, DATABASE_BACKEND, Engine

    Session.remove()
    Engine.dispose()

    SQLALCHEMY_DATABASE_URL, DATABASE_BACKEND = _database_config(
        backend=backend,
        database_url=database_url,
    )
    Engine = _create_engine(SQLALCHEMY_DATABASE_URL, DATABASE_BACKEND)
    _session_factory.configure(bind=Engine)
    return Engine
