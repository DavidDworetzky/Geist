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
    ensure_default_workspace,
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
    first = ensure_default_workspace()
    second = ensure_default_workspace()

    assert first == second
    assert first.workspace_key == DEFAULT_WORKSPACE_KEY
    assert first.display_name == "Local Workspace"
    with SessionLocal() as session:
        rows = session.query(GeistUser).all()
        assert len(rows) == 1
        assert rows[0].username is None
        assert rows[0].email is None
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

    workspace = ensure_default_workspace()

    assert workspace.workspace_id == 41
    assert workspace.workspace_key == DEFAULT_WORKSPACE_KEY
    assert workspace.display_name == "Local Workspace"
    with SessionLocal() as session:
        chat = session.get(ChatSession, 73)
        database_user = session.get(GeistUser, 41)
        assert chat.user_id == 41
        assert database_user.username is None
        assert database_user.name == "Local Workspace"
        assert database_user.email is None
        assert database_user.password is None


def test_bootstrap_normalizes_an_existing_default_workspace(workspace_database):
    with SessionLocal() as session:
        session.add(
            GeistUser(
                workspace_key="default",
                username="local",
                name="Local User",
                email="obsolete@example.com",
                password="obsolete",
            )
        )
        session.commit()

    workspace = ensure_default_workspace()

    assert workspace.display_name == "Local Workspace"
    with SessionLocal() as session:
        database_user = session.query(GeistUser).filter_by(workspace_key="default").one()
        assert database_user.username is None
        assert database_user.email is None
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

    workspace = ensure_default_workspace()

    assert workspace.workspace_id == 7
    assert workspace.display_name == "Custom Display Name"
    with SessionLocal() as session:
        database_user = session.get(GeistUser, 7)
        assert database_user.username == "custom-handle"
        assert database_user.email is None


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

    workspace = ensure_default_workspace()

    assert workspace.workspace_id == 3
    with SessionLocal() as session:
        rows = session.query(GeistUser).order_by(GeistUser.user_id).all()
        assert [row.workspace_key for row in rows] == ["default", None]
        assert [row.email for row in rows] == [None, None]
        assert [row.password for row in rows] == [None, None]
        assert rows[0].username is None
        assert rows[0].name == "Local Workspace"
        assert rows[1].name == "Customized duplicate"
