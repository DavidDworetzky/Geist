import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import create_app

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