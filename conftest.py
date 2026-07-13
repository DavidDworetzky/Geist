import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from agents.agent_settings import AgentSettings
from agents.agent_context import AgentContext
from agents.online_agent import OnlineAgent
from agents.local_agent import LocalAgent
from agents.architectures.registry import register_runner
from agents.architectures.base_runner import BaseRunner, GenerationConfig
import os
from app.environment import LoadEnvironmentDictionary

def CreateAgentSettings(include_world_processing: bool = True):
    return AgentSettings(name="default agent", version="1.0", description="default", include_world_processing=include_world_processing)

openai_key = os.getenv("OPENAI_API_KEY")
def get_envs() -> dict[str,str]:
    return LoadEnvironmentDictionary()

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
            {"role": "assistant", "content": f"Response to: {user_prompt}"}
        ]

@pytest.fixture(scope="module")
def app():
    # Create a version of our fastapi instance for testing
    app = create_app()
    yield app

@pytest.fixture(scope="module")
def client(app):
    # Create a TestClient instance using the app fixture
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def online_agent():
    settings = CreateAgentSettings(include_world_processing=True)
    context = AgentContext(settings=settings, envs=get_envs())
    return OnlineAgent(agent_context=context, base_url="https://api.openai.com/v1",
                       model="gpt-4", api_key=None)

@pytest.fixture(scope="module")
def process_world_variation_online_agents():
    variant_settings = [CreateAgentSettings(include_world_processing=True), CreateAgentSettings(include_world_processing=False)]
    contexts = [AgentContext(settings=setting, envs=get_envs()) for setting in variant_settings]
    return [OnlineAgent(agent_context=context, base_url="https://api.openai.com/v1",
                        model="gpt-4", api_key=None) for context in contexts]

@pytest.fixture(scope="module")
def local_agent():
    register_runner("stub", StubRunner)
    settings = CreateAgentSettings(include_world_processing=True)
    context = AgentContext(settings=settings, envs=get_envs())
    return LocalAgent(agent_context=context, model_id="stub-model", runner_type="stub")
