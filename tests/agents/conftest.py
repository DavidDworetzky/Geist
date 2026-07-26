"""Shared isolation for agent tests."""

from collections.abc import Generator

import pytest

from agents.model_load_status import model_load_status_registry


@pytest.fixture(autouse=True)
def clear_model_load_status_registry() -> Generator[None, None, None]:
    """Prevent process-local model state from leaking between tests."""
    model_load_status_registry.clear()
    yield
    model_load_status_registry.clear()
