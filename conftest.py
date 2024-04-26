import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import create_app
from agents.agent_settings import AgentSettings
from agents.gpt4_agent import GPT4Agent
from agents.agent_context import AgentContext

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
    settings = AgentSettings(name="default agent", version="1.0", description="default")
    context = AgentContext(settings, None, [], [], [], [])
    api_key = None
    return GPT4Agent(api_key=api_key, agent_context=context)
