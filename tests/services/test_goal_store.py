from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import agent_goal
from app.services.goal_runtime import DatabaseGoalStore, GoalRuntime


def test_database_checkpoint_roundtrip_and_owner_scope(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'goals.sqlite'}")
    agent_goal.AgentGoal.__table__.create(engine)
    monkeypatch.setattr(agent_goal, "SessionLocal", sessionmaker(bind=engine))
    store = DatabaseGoalStore()
    goal = GoalRuntime(objective="Voice notes", tasks=[], max_turns=10, workspace_id="workspace-1")
    store.create(goal.checkpoint(), 1, "run-1")
    store.attach_chat(goal.state.goal_id, 7)
    goal.wait_for_user("Which provider?")
    saved = goal.checkpoint([{"role": "user", "content": "Voice notes"}])
    store.update(saved)
    assert store.load_latest(1, 7) == saved
    assert store.load_latest(2, 7) is None
    assert store.load_latest(1, 8) is None
    engine.dispose()
