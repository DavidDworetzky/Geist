import os

import pytest
from fastapi.testclient import TestClient

from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.architectures.registry import register_runner
from agents.local_agent import LocalAgent
from agents.online_agent import OnlineAgent
from app.environment import load_environment_dictionary
from app.main import create_app


def create_agent_settings(include_world_processing: bool = True):
    return AgentSettings(
        name="default agent",
        version="1.0",
        description="default",
        include_world_processing=include_world_processing,
    )


openai_key = os.getenv("OPENAI_API_KEY")


def get_envs() -> dict[str, str]:
    return load_environment_dictionary()


class StubRunner(BaseRunner):
    """Deterministic in-memory runner so LocalAgent tests never load real weights."""

    def load(self, model_id, device_config=None):
        self.model_id = model_id

    def generate(self, prompt, generation_config: GenerationConfig):
        return {"generated_text": f"Generated: {prompt}"}

    def complete(self, system_prompt, user_prompt, generation_config: GenerationConfig):
        # Same shape as the real runners: a list of message dicts (see strings_to_message_dict)
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"Response to: {user_prompt}"},
        ]

    def complete_messages(self, messages, generation_config: GenerationConfig):
        user_prompt = next(
            message["content"] for message in reversed(messages) if message["role"] == "user"
        )
        return self.complete("", user_prompt, generation_config)


@pytest.fixture(scope="module")
def app():
    # Create a version of our fastapi instance for testing
    app = create_app()
    yield app


@pytest.fixture(scope="module")
def client(app):
    # Create a TestClient instance using the app fixture
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture(scope="module")
def online_agent():
    settings = create_agent_settings(include_world_processing=True)
    context = AgentContext(settings=settings, envs=get_envs())
    return OnlineAgent(
        agent_context=context, base_url="https://api.openai.com/v1", model="gpt-4", api_key=None
    )


@pytest.fixture(scope="module")
def process_world_variation_online_agents():
    variant_settings = [
        create_agent_settings(include_world_processing=True),
        create_agent_settings(include_world_processing=False),
    ]
    contexts = [AgentContext(settings=setting, envs=get_envs()) for setting in variant_settings]
    return [
        OnlineAgent(
            agent_context=context, base_url="https://api.openai.com/v1", model="gpt-4", api_key=None
        )
        for context in contexts
    ]


@pytest.fixture(scope="module")
def local_agent():
    register_runner("stub", StubRunner)
    settings = create_agent_settings(include_world_processing=True)
    context = AgentContext(settings=settings, envs=get_envs())
    return LocalAgent(agent_context=context, model_id="stub-model", runner_type="stub")
