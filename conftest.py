import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import create_app
from agents.agent_settings import AgentSettings
from agents.gpt4_agent import GPT4Agent
from agents.agent_context import AgentContext
from dotenv import load_dotenv
import os
from app.environment import LoadEnvironmentDictionary

def CreateAgentSettings(include_world_processing: bool = True):
    return AgentSettings(name="default agent", version="1.0", description="default", include_world_processing=include_world_processing)

openai_key = os.getenv("OPENAI_TOKEN")
def get_envs() -> dict[str,str]:
    return LoadEnvironmentDictionary()

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
def gpt4agent():
    settings = CreateAgentSettings(include_world_processing=False)
    env = get_envs()
    context = AgentContext(settings=settings, envs=env)
    api_key = None
    return GPT4Agent(api_key=api_key, agent_context=context)

@pytest.fixture(scope="module")
def process_world_variation_gpt4agents():
    variant_settings = [CreateAgentSettings(include_world_processing=True), CreateAgentSettings(include_world_processing=False)] 
    env = get_envs()
    contexts = [AgentContext(settings=setting, envs=env) for setting in variant_settings]
    api_key = None
    return [GPT4Agent(api_key=api_key, agent_context=context) for context in contexts]

