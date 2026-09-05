"""Shared prompt inference service for HTTP and background execution."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent
from agents.factory import AgentFactory
from agents.models.agent_completion import AgentCompletion
from agents.prompt.prompt import AGENT_PROMPTS
from app.models.database.geist_user import get_default_workspace
from app.models.user_settings import AgentConfigRequest, AgentFactoryConfig
from app.schemas.prompt_schedule import InferenceConfig
from app.services.agent_context_provider import get_default_agent_context
from app.services.user_settings_service import UserSettingsService


DEFAULT_SYSTEM_PROMPT = AGENT_PROMPTS["default"]
_MAX_CACHED_RUNTIMES = 4


@dataclass
class _Runtime:
    agent: BaseAgent
    lock: threading.Lock


_runtime_cache: OrderedDict[tuple[Any, ...], _Runtime] = OrderedDict()
_runtime_cache_lock = threading.Lock()


def _resolved_generation_config(
    settings,
    overrides: InferenceConfig,
) -> dict[str, int | float]:
    return {
        "max_tokens": (
            overrides.max_tokens
            if overrides.max_tokens is not None
            else settings.default_max_tokens
        ),
        "temperature": (
            overrides.temperature
            if overrides.temperature is not None
            else settings.default_temperature
        ),
        "top_p": overrides.top_p if overrides.top_p is not None else settings.default_top_p,
        "frequency_penalty": (
            overrides.frequency_penalty
            if overrides.frequency_penalty is not None
            else settings.default_frequency_penalty
        ),
        "presence_penalty": (
            overrides.presence_penalty
            if overrides.presence_penalty is not None
            else settings.default_presence_penalty
        ),
    }


def _runtime_key(user_id: int, config: AgentFactoryConfig) -> tuple[Any, ...]:
    return (
        user_id,
        config.agent_type,
        config.model,
        config.endpoint,
        config.runner_type,
        json.dumps(config.device_config, sort_keys=True, default=str),
        json.dumps(config.generation_config, sort_keys=True, default=str),
        tuple(
            (provider.name, provider.base_url, provider.model, provider.priority)
            for provider in config.backup_providers
        ),
    )


def _create_runtime(config: AgentFactoryConfig) -> _Runtime:
    factory_kwargs: dict[str, Any] = {}
    if config.agent_type == "online":
        factory_kwargs["backup_providers"] = [
            provider.model_dump() for provider in config.backup_providers
        ]
    agent = AgentFactory.create_agent(
        agent_type=config.agent_type,
        agent_context=get_default_agent_context(),
        model=config.model,
        endpoint=config.endpoint,
        api_key=config.api_key,
        runner_type=config.runner_type,
        device_config=config.device_config,
        generation_config=config.generation_config,
        **factory_kwargs,
    )
    return _Runtime(agent=agent, lock=threading.Lock())


def _get_runtime(user_id: int, config: AgentFactoryConfig) -> _Runtime:
    key = _runtime_key(user_id, config)
    with _runtime_cache_lock:
        runtime = _runtime_cache.pop(key, None)
        if runtime is None:
            runtime = _create_runtime(config)
        _runtime_cache[key] = runtime
        while len(_runtime_cache) > _MAX_CACHED_RUNTIMES:
            _, evicted = _runtime_cache.popitem(last=False)
            evicted.agent._cleanup_resources()
        return runtime


def clear_inference_runtime_cache() -> None:
    """Close and remove cached runtimes; primarily useful for tests and shutdown."""
    with _runtime_cache_lock:
        runtimes = list(_runtime_cache.values())
        _runtime_cache.clear()
    for runtime in runtimes:
        runtime.agent._cleanup_resources()


def run_prompt_inference(
    prompt: str,
    inference_config: dict[str, Any] | InferenceConfig | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Run one prompt using user settings and return a JSON-safe completion."""
    if user_id is None:
        user_id = get_default_workspace().workspace_id
    overrides = (
        inference_config
        if isinstance(inference_config, InferenceConfig)
        else InferenceConfig.model_validate(inference_config or {})
    )
    settings = UserSettingsService.get_or_create_workspace_settings_by_id(user_id)
    factory_overrides = AgentConfigRequest(**overrides.model_dump(exclude_none=True))
    factory_config = AgentFactoryConfig.from_user_settings(settings, factory_overrides)
    generation = _resolved_generation_config(settings, overrides)
    runtime = _get_runtime(user_id, factory_config)

    with runtime.lock:
        completion = runtime.agent.complete_text(
            prompt=prompt,
            max_tokens=int(generation["max_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            frequency_penalty=float(generation["frequency_penalty"]),
            presence_penalty=float(generation["presence_penalty"]),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
    response = AgentCompletion.from_completion(completion)
    return {
        "message": response.message,
        "completion_id": response.id,
        "chat_id": response.chat_id,
    }
