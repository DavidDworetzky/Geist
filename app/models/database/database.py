from dotenv import load_dotenv
from sqlalchemy.engine import Engine as SqlAlchemyEngine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from app.models.database.database_config import (
    DEFAULT_DATABASE_PROVIDERS,
    DatabaseConfig,
    DatabaseProviderRegistry,
    create_database_engine,
    load_database_config,
)

load_dotenv()

DATABASE_CONFIG = load_database_config()
SQLALCHEMY_DATABASE_URL = DATABASE_CONFIG.database_url
DATABASE_PROVIDER = DATABASE_CONFIG.provider
DATABASE_BACKEND = DATABASE_PROVIDER

Engine = create_database_engine(DATABASE_CONFIG)
_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
SessionLocal = _session_factory
Session = scoped_session(_session_factory)

Base = declarative_base()


def configure_database(
    config: DatabaseConfig,
    registry: DatabaseProviderRegistry = DEFAULT_DATABASE_PROVIDERS,
) -> SqlAlchemyEngine:
    """
    Rebind the process-wide SQLAlchemy session factory.

    This is primarily used by tests and alternate storage configurations. Keeping
    the same session factory lets existing model modules continue using their
    imported SessionLocal object after the engine changes.
    """
    global DATABASE_CONFIG, SQLALCHEMY_DATABASE_URL, DATABASE_PROVIDER, DATABASE_BACKEND, Engine

    Session.remove()
    Engine.dispose()

    DATABASE_CONFIG = config
    SQLALCHEMY_DATABASE_URL = config.database_url
    DATABASE_PROVIDER = config.provider
    DATABASE_BACKEND = DATABASE_PROVIDER
    Engine = create_database_engine(config, registry=registry)
    _session_factory.configure(bind=Engine)
    return Engine
