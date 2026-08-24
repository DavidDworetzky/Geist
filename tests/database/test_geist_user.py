import importlib

import pytest

from app.models.database.chat_session import ChatSession
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import (
    DEFAULT_WORKSPACE_KEY,
    GeistUser,
    ensure_default_user,
)


@pytest.fixture()
def workspace_database(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'workspace.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def test_bootstrap_creates_one_neutral_workspace_and_is_idempotent(workspace_database):
    first = ensure_default_user()
    second = ensure_default_user()

    assert first == second
    assert first.workspace_key == DEFAULT_WORKSPACE_KEY
    assert first.username == "local"
    assert first.name == "Local User"
    assert first.email is None
    with SessionLocal() as session:
        rows = session.query(GeistUser).all()
        assert len(rows) == 1
        assert rows[0].password is None


def test_bootstrap_adopts_legacy_row_without_changing_ownership(workspace_database):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=41,
                username="ddworetzky",
                name="David Dworetzky",
                email="david@phantasmal.ai",
                password="",
            )
        )
        session.flush()
        session.add(ChatSession(chat_session_id=73, user_id=41, chat_history="[]"))
        session.commit()

    workspace = ensure_default_user()

    assert workspace.user_id == 41
    assert workspace.workspace_key == DEFAULT_WORKSPACE_KEY
    assert workspace.username == "local"
    assert workspace.name == "Local User"
    assert workspace.email is None
    with SessionLocal() as session:
        chat = session.get(ChatSession, 73)
        database_user = session.get(GeistUser, 41)
        assert chat.user_id == 41
        assert database_user.password is None


def test_bootstrap_preserves_customized_legacy_display_metadata(workspace_database):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                user_id=7,
                username="custom-handle",
                name="Custom Display Name",
                email="david@phantasmal.ai",
                password="legacy-value",
            )
        )
        session.commit()

    workspace = ensure_default_user()

    assert workspace.user_id == 7
    assert workspace.username == "custom-handle"
    assert workspace.name == "Custom Display Name"
    assert workspace.email is None


def test_bootstrap_scrubs_duplicate_legacy_rows(workspace_database):
    with SessionLocal() as session:
        session.add_all(
            [
                GeistUser(
                    user_id=3,
                    username="ddworetzky",
                    name="David Dworetzky",
                    email="david@phantasmal.ai",
                    password="",
                ),
                GeistUser(
                    user_id=4,
                    username="duplicate",
                    name="Customized duplicate",
                    email="david@phantasmal.ai",
                    password="legacy-value",
                ),
            ]
        )
        session.commit()

    workspace = ensure_default_user()

    assert workspace.user_id == 3
    with SessionLocal() as session:
        rows = session.query(GeistUser).order_by(GeistUser.user_id).all()
        assert [row.workspace_key for row in rows] == ["default", None]
        assert [row.email for row in rows] == [None, None]
        assert [row.password for row in rows] == [None, None]
        assert rows[1].name == "Customized duplicate"
