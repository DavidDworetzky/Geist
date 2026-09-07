"""Tests for Claude Fable and Mythos 5.1 in the online model catalog."""

import asyncio

import pytest

from agents.architectures.registry import (
    DISCOVERED_MODELS,
    STATIC_MODELS,
    OnlineModelNames,
    OnlineModelProviders,
    get_model_by_id,
)
from app.api.v1.endpoints.models import get_available_models
from scripts.model_filter_config import ALWAYS_INCLUDE, get_model_metadata


@pytest.mark.parametrize(
    ("model_name", "model_id", "recommended", "family"),
    [
        (
            OnlineModelNames.CLAUDE_FABLE_51,
            "claude-fable-5-1",
            True,
            "claude-fable",
        ),
        (
            OnlineModelNames.CLAUDE_MYTHOS_51,
            "claude-mythos-5-1",
            False,
            "claude-mythos",
        ),
    ],
)
def test_claude_5_1_models_are_registered(
    model_name: OnlineModelNames,
    model_id: str,
    recommended: bool,
    family: str,
):
    assert model_name.value == model_id
    for catalog in (STATIC_MODELS, DISCOVERED_MODELS):
        assert model_id in {model.id for model in catalog[OnlineModelProviders.ANTHROPIC]}

    model = get_model_by_id(model_id)
    assert model is not None
    assert model.provider is OnlineModelProviders.ANTHROPIC
    assert model.context_window == 1000000
    assert model.max_output_tokens == 128000
    assert model.supports_vision is True
    assert model.supports_function_calling is True
    assert model.supports_streaming is True
    assert model.recommended is recommended
    assert model.family == family

    assert model_id in ALWAYS_INCLUDE
    assert get_model_metadata(model_id) == {
        "name": model.name,
        "recommended": recommended,
        "context_window": 1000000,
        "max_output_tokens": 128000,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": family,
    }


def test_models_route_exposes_claude_5_1_models():
    response = asyncio.run(get_available_models())
    models = {model.id: model for model in response.providers["anthropic"]}

    assert "claude-fable-5-1" in models
    assert "claude-mythos-5-1" in models
    assert models["claude-fable-5-1"].name == "Claude Fable 5.1"
    assert models["claude-mythos-5-1"].name == "Claude Mythos 5.1 (Invite Only)"
