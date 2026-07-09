"""Tests for agent execution state snapshots."""
import importlib

import pytest

from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.online_agent import OnlineAgent
from app.models.database.agent_snapshot import (
    AgentSnapshot,
    create_snapshot,
    get_latest_snapshot,
    get_snapshot_by_id,
    get_snapshots,
    prune_snapshots,
    prune_snapshots_older_than,
)
from app.models.database.database import (
    Base,
    DATABASE_CONFIG,
    Session,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    sqlite_config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'geist.sqlite3'}",
    )

    engine = configure_database(sqlite_config)

    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def make_context(agent_id=None):
    settings = AgentSettings(name="snap", version="1.0", description="snapshot test")
    return AgentContext(
        settings=settings,
        agent_id=agent_id,
        world_context=[],
        task_context=[],
        execution_context=[],
        function_log=[],
        execution_classes=[("InertAdapter", [])],
    )


class TestSnapshotModel:
    def test_steps_are_monotonic_per_agent(self, sqlite_database):
        first = create_snapshot("agent-a", ["w"], ["t"], ["e"], reason="tick")
        second = create_snapshot("agent-a", ["w2"], ["t2"], ["e2"], reason="tick")
        other = create_snapshot("agent-b", [], [], [], reason="manual")

        assert first.step == 1
        assert second.step == 2
        assert other.step == 1

    def test_get_snapshots_orders_most_recent_first(self, sqlite_database):
        create_snapshot("agent-a", ["w1"], [], [], reason="tick")
        create_snapshot("agent-a", ["w2"], [], [], reason="tick")

        snapshots = get_snapshots("agent-a")
        assert [snapshot.step for snapshot in snapshots] == [2, 1]

        latest = get_latest_snapshot("agent-a")
        assert latest.step == 2
        assert latest.to_dict()["world_context"] == ["w2"]

    def test_to_dict_round_trips_contexts(self, sqlite_database):
        snapshot = create_snapshot(
            "agent-a",
            ["world fact"],
            ["task one", "task two"],
            ["execute this"],
            function_log=['{"function": "add"}'],
            reason="phase_out",
        )
        state = get_snapshot_by_id(snapshot.snapshot_id).to_dict()

        assert state["world_context"] == ["world fact"]
        assert state["task_context"] == ["task one", "task two"]
        assert state["execution_context"] == ["execute this"]
        assert state["function_log"] == ['{"function": "add"}']
        assert state["reason"] == "phase_out"

    def test_prune_keeps_most_recent(self, sqlite_database):
        for index in range(5):
            create_snapshot("agent-a", [f"w{index}"], [], [], reason="tick")

        deleted = prune_snapshots("agent-a", keep_last=2)

        assert deleted == 3
        assert [snapshot.step for snapshot in get_snapshots("agent-a")] == [5, 4]


def backdate_snapshot(snapshot_id: int, days: int) -> None:
    import datetime

    from app.models.database.database import SessionLocal

    with SessionLocal() as session:
        session.query(AgentSnapshot).filter(
            AgentSnapshot.snapshot_id == snapshot_id
        ).update({
            AgentSnapshot.created_at: datetime.datetime.utcnow() - datetime.timedelta(days=days)
        })
        session.commit()


class TestSnapshotRetention:
    def test_prunes_snapshots_past_retention(self, sqlite_database):
        expired_one = create_snapshot("agent-a", ["old1"], [], [], reason="tick")
        expired_two = create_snapshot("agent-a", ["old2"], [], [], reason="tick")
        create_snapshot("agent-a", ["fresh"], [], [], reason="tick")
        backdate_snapshot(expired_one.snapshot_id, days=8)
        backdate_snapshot(expired_two.snapshot_id, days=30)

        deleted = prune_snapshots_older_than("agent-a", retention_days=7)

        assert deleted == 2
        assert [snapshot.to_dict()["world_context"] for snapshot in get_snapshots("agent-a")] == [["fresh"]]

    def test_latest_snapshot_survives_regardless_of_age(self, sqlite_database):
        only = create_snapshot("agent-a", ["ancient"], [], [], reason="phase_out")
        backdate_snapshot(only.snapshot_id, days=365)

        assert prune_snapshots_older_than("agent-a", retention_days=7) == 0
        assert get_latest_snapshot("agent-a").snapshot_id == only.snapshot_id

    def test_zero_retention_disables_pruning(self, sqlite_database):
        old = create_snapshot("agent-a", ["old"], [], [], reason="tick")
        create_snapshot("agent-a", ["new"], [], [], reason="tick")
        backdate_snapshot(old.snapshot_id, days=100)

        assert prune_snapshots_older_than("agent-a", retention_days=0) == 0
        assert len(get_snapshots("agent-a")) == 2

    def test_context_snapshot_prunes_on_write(self, sqlite_database):
        context = make_context(agent_id="retained-agent")
        context.task_context = ["stale"]
        stale = context.snapshot(reason="tick")
        backdate_snapshot(stale.snapshot_id, days=8)

        context.task_context = ["current"]
        context.snapshot(reason="tick")

        remaining = get_snapshots("retained-agent")
        assert [snapshot.to_dict()["task_context"] for snapshot in remaining] == [["current"]]

    def test_context_snapshot_respects_disabled_retention(self, sqlite_database):
        context = make_context(agent_id="unbounded-agent")
        context.settings.snapshot_retention_days = 0
        stale = context.snapshot(reason="tick")
        backdate_snapshot(stale.snapshot_id, days=100)

        context.snapshot(reason="tick")

        assert len(get_snapshots("unbounded-agent")) == 2


class TestAgentContextSnapshots:
    def test_snapshot_and_restore_round_trip(self, sqlite_database):
        context = make_context(agent_id="ctx-agent")
        context.world_context = ["the sky is blue"]
        context.task_context = ["write tests"]
        context.execution_context = ["call the tool"]
        context.function_log = ["logged call"]

        snapshot = context.snapshot(reason="tick")
        assert snapshot is not None
        assert snapshot.step == 1

        rehydrated = make_context(agent_id="ctx-agent")
        assert rehydrated.restore_snapshot()

        assert rehydrated.world_context == ["the sky is blue"]
        assert rehydrated.task_context == ["write tests"]
        assert rehydrated.execution_context == ["call the tool"]
        assert rehydrated.function_log == ["logged call"]

    def test_restore_specific_snapshot(self, sqlite_database):
        context = make_context(agent_id="ctx-agent")
        context.task_context = ["version one"]
        first = context.snapshot(reason="tick")
        context.task_context = ["version two"]
        context.snapshot(reason="tick")

        assert context.restore_snapshot(snapshot_id=first.snapshot_id)
        assert context.task_context == ["version one"]

    def test_restore_refuses_foreign_snapshot(self, sqlite_database):
        other = make_context(agent_id="other-agent")
        other.task_context = ["not yours"]
        foreign = other.snapshot()

        context = make_context(agent_id="ctx-agent")
        context.task_context = ["mine"]
        assert not context.restore_snapshot(snapshot_id=foreign.snapshot_id)
        assert context.task_context == ["mine"]

    def test_restore_without_snapshots_returns_false(self, sqlite_database):
        context = make_context(agent_id="never-snapshotted")
        assert not context.restore_snapshot()

    def test_snapshot_failure_does_not_raise(self, monkeypatch):
        # Persistence failures must be swallowed and reported as None rather
        # than break the agent loop.
        import agents.agent_context as agent_context_module

        def failing_create_snapshot(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(agent_context_module, "create_snapshot", failing_create_snapshot)
        context = make_context(agent_id="no-db-agent")
        context.world_context = ["irrelevant"]
        assert context.snapshot(reason="tick") is None


class TestAgentSnapshotIntegration:
    def make_agent(self, agent_id="online-agent"):
        context = make_context(agent_id=agent_id)
        return OnlineAgent(
            context, base_url="https://example.test/v1", model="test-model", api_key="key"
        )

    def test_tick_writes_snapshot(self, sqlite_database, monkeypatch):
        agent = self.make_agent()
        monkeypatch.setattr(agent, "tick_world", lambda: None)
        monkeypatch.setattr(agent, "tick_tasks", lambda: agent._agent_context.task_context.append("task"))
        monkeypatch.setattr(agent, "tick_execution", lambda: agent._agent_context.execution_context.append("run"))

        agent.tick()
        agent.tick()

        snapshots = get_snapshots(agent._agent_context.agent_id)
        assert [snapshot.step for snapshot in snapshots] == [2, 1]
        assert all(snapshot.reason == "tick" for snapshot in snapshots)
        assert get_latest_snapshot(agent._agent_context.agent_id).to_dict()["task_context"] == ["task", "task"]

    def test_phase_out_then_phase_in_restores_state(self, sqlite_database):
        agent = self.make_agent(agent_id="phased-agent")
        agent._agent_context.task_context = ["finish the report"]
        agent._agent_context.world_context = ["deadline is friday"]

        agent.phase_out()

        resumed_context = make_context(agent_id="phased-agent")
        resumed = OnlineAgent(
            resumed_context, base_url="https://example.test/v1", model="test-model", api_key="key"
        )
        resumed.phase_in()

        assert resumed._agent_context.task_context == ["finish the report"]
        assert resumed._agent_context.world_context == ["deadline is friday"]

        latest = get_latest_snapshot("phased-agent")
        assert latest.reason == "phase_out"
