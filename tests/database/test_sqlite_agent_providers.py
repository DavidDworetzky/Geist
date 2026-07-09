import importlib
import json

import pytest

from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from app.models.database.database import (
    Base,
    DATABASE_CONFIG,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig


class SQLiteTestRunner:
    def __init__(self):
        self.loaded_model = None
        self.cleaned_up = False

    def load(self, model_id, device_config=None):
        self.loaded_model = model_id

    def complete(self, system_prompt, user_prompt, generation_config):
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"local response: {user_prompt}"},
        ]

    def cleanup(self):
        self.cleaned_up = True


ONLINE_RESPONSE = {
    "id": "sqlite-online-completion",
    "object": "chat.completion",
    "created": 0,
    "model": "test-online-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "online response: provider persistence",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


@pytest.fixture()
def sqlite_database(tmp_path):
    original_config = DATABASE_CONFIG
    sqlite_config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'agent-providers.sqlite3'}",
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


def create_persisted_context():
    from app.models.database.agent import Agent

    with SessionLocal() as session:
        agent_record = Agent(
            process_id=None,
            world_context="[]",
            task_context="[]",
            execution_context="[]",
        )
        session.add(agent_record)
        session.commit()
        session.refresh(agent_record)
        agent_id = agent_record.agent_id

    settings = AgentSettings(
        name="SQLite Provider Test",
        version="1.0",
        description="Exercises agent provider persistence",
        max_tokens=32,
        n=1,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        include_world_processing=True,
    )
    return AgentContext(
        settings=settings,
        agent_id=agent_id,
        world_context=[],
        task_context=[],
        execution_context=[],
        execution_classes=[],
        envs={},
    )


def create_local_agent(context, monkeypatch):
    import agents.local_agent as local_agent_module

    monkeypatch.setattr(local_agent_module, "ensure_runners_registered", lambda: None)
    monkeypatch.setattr(
        local_agent_module,
        "get_runner",
        lambda runner_type: SQLiteTestRunner if runner_type == "sqlite_test" else None,
    )
    return local_agent_module.LocalAgent(
        agent_context=context,
        model_id="test-local-model",
        runner_type="sqlite_test",
    )


def create_online_agent(context, monkeypatch):
    from agents.online_agent import OnlineAgent

    agent = OnlineAgent(
        agent_context=context,
        base_url="https://example.invalid/v1",
        model="test-online-model",
        api_key="test-key",
    )
    monkeypatch.setattr(agent, "_make_request", lambda payload: ONLINE_RESPONSE)
    return agent


@pytest.mark.parametrize(
    ("provider_name", "agent_factory", "expected_reply"),
    [
        ("local", create_local_agent, "local response: provider persistence"),
        ("online", create_online_agent, "online response: provider persistence"),
    ],
)
def test_agent_provider_common_operations_persist_with_sqlite(
    sqlite_database,
    monkeypatch,
    provider_name,
    agent_factory,
    expected_reply,
):
    from app.models.database.agent import Agent
    from app.models.database.chat_session import get_chat_history

    context = create_persisted_context()
    agent = agent_factory(context, monkeypatch)

    agent.initialize(task_prompt=f"{provider_name} initialized")
    completion = agent.complete_text(
        prompt="provider persistence",
        chat_id=8100 if provider_name == "local" else 8200,
    )

    assert completion.chat_id == (8100 if provider_name == "local" else 8200)
    assert agent.state()["task_context"] == [f"{provider_name} initialized"]

    context.world_context = [f"{provider_name} world"]
    context.task_context.append(f"{provider_name} task")
    context.execution_context = [f"{provider_name} execution"]
    agent.phase_out()

    with SessionLocal() as session:
        persisted_agent = session.query(Agent).filter_by(agent_id=context.agent_id).one()
        assert json.loads(persisted_agent.world_context) == [f"{provider_name} world"]
        assert json.loads(persisted_agent.task_context) == [
            f"{provider_name} initialized",
            f"{provider_name} task",
        ]
        assert json.loads(persisted_agent.execution_context) == [
            f"{provider_name} execution"
        ]

    history = get_chat_history(completion.chat_id)
    assert history.chat_history == [
        {
            "user": "provider persistence",
            "ai": expected_reply,
        }
    ]

    agent.phase_in()
    assert context.subprocess_id is None
