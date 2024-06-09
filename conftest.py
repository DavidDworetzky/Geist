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

openai_key = os.getenv("OPENAI_TOKEN")
def get_envs() -> dict[str,str]:
    return {
        "openai_key" : openai_key,
    }
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
    env = get_envs()
    context = AgentContext(settings=settings, envs=env)
    api_key = None
    return GPT4Agent(api_key=api_key, agent_context=context)
