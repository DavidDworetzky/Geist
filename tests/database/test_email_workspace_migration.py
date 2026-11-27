import importlib

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def test_pre_workspace_email_database_preserves_connector_ownership():
    engine = create_engine("sqlite:///:memory:")
    migration = importlib.import_module(
        "migrations.versions.b9d2e4f6a8c0_merge_email_workspace_heads"
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE geist_user (user_id INTEGER PRIMARY KEY, username TEXT, "
                "name TEXT, email TEXT, password TEXT)"
            )
        )
        connection.execute(
            text("INSERT INTO geist_user (user_id, name) VALUES (7, 'My workspace')")
        )
        connection.execute(
            text(
                "CREATE TABLE mcp_server (mcp_server_id INTEGER PRIMARY KEY, "
                "user_id INTEGER NOT NULL REFERENCES geist_user(user_id), name TEXT NOT NULL)"
            )
        )
        connection.execute(text("INSERT INTO mcp_server VALUES (42, 7, 'proton')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
        assert connection.execute(
            text("SELECT workspace_key, name FROM geist_user WHERE user_id = 7")
        ).one() == ("default", "My workspace")
        assert connection.execute(
            text("SELECT mcp_server_id, workspace_id, name FROM mcp_server")
        ).one() == (42, 7, "proton")
