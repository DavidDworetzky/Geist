import datetime
import importlib

import pytest

from app.models.database.agent_routine import (
    AgentRoutine,
    claim_routine_run,
    create_routine,
)
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    engine = configure_database(
        DatabaseConfig(
            provider="sqlite",
            database_url=f"sqlite:///{tmp_path / 'routines.sqlite3'}",
        )
    )
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    try:
        with SessionLocal() as session:
            session.add(GeistUser(user_id=41, username="owner", email="owner@example.com"))
            session.commit()
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def test_claim_routine_run_allows_only_one_scheduler(sqlite_database):
    routine = create_routine(
        user_id=41,
        name="digest",
        prompt="Summarize the week",
        interval_minutes=60,
    )
    due_at = datetime.datetime(2026, 8, 18, 12, 0, 0)
    with SessionLocal() as session:
        stored = session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).one()
        stored.next_run_at = due_at
        session.commit()
    routine.next_run_at = due_at
    claimed_at = due_at + datetime.timedelta(seconds=1)

    assert claim_routine_run(routine, now=claimed_at) is True
    assert claim_routine_run(routine, now=claimed_at) is False

    with SessionLocal() as session:
        stored = session.query(AgentRoutine).filter_by(routine_id=routine.routine_id).one()
        assert stored.last_run_at == claimed_at
        assert stored.next_run_at == claimed_at + datetime.timedelta(minutes=60)
