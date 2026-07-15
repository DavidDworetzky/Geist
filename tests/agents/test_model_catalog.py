"""Tests for generic model/provider catalog and runner routing."""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.factory import AgentFactory
from agents.model_catalog import (
    MODEL_SPECS,
    PROVIDERS,
    ProviderSpec,
    get_model_spec,
    get_provider_endpoint,
    infer_model_spec,
)


def test_catalog_covers_requested_families():
    families = {spec.family for spec in MODEL_SPECS}
    assert {
        "llama", "qwen", "mistral", "phi", "smollm", "gemma",
        "granite", "olmo", "glm", "gpt-oss", "kimi", "deepseek",
    }.issubset(families)


def test_family_inference_supports_future_finetunes():
    assert infer_model_spec("Qwen/custom-Qwen3-finetune").family == "qwen"
    assert infer_model_spec("mlx-community/Mistral-7B-Instruct-v0.3").family == "mistral"
    assert infer_model_spec("my-org/gemma-3-quantized").family == "gemma"


def test_heavyweight_models_are_server_backed():
    kimi = get_model_spec("kimi-k2.5")
    assert kimi.backend == "openai_compatible"
    assert kimi.local is False
    assert get_provider_endpoint(kimi.provider) == "https://api.moonshot.ai/v1"

    from agents.architectures.registry import get_all_models, provider_from_string

    assert provider_from_string("moonshot") == "moonshot"
    assert "moonshot" in get_all_models()

    hosted_glm = get_model_spec("glm-4.7-flash")
    assert hosted_glm.local is False
    assert get_provider_endpoint(hosted_glm.provider) == "https://api.z.ai/api/paas/v4"


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-1b-it",
    "ibm-granite/granite-3.3-8b-instruct",
    "allenai/Olmo-3-7B-Instruct",
    "zai-org/glm-4-9b-chat-hf",
    "openai/gpt-oss-20b",
])
def test_standard_local_models_use_generic_runner(model_id):
    assert AgentFactory._infer_runner_type(model_id) == "transformers"


def test_unknown_huggingface_model_uses_generic_runner():
    assert AgentFactory._infer_runner_type("new-org/future-causal-lm") == "transformers"
    assert AgentFactory._infer_runner_type("new-org/future-8B-instruct") == "transformers"


@pytest.mark.parametrize(
    "model_id, expected_parameters",
    [
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", "70B"),
        ("Qwen/Qwen3-32B", "32B"),
        ("future-org/Future-8x22B-Instruct", "176B"),
        ("future-org/Future-1T-Instruct", "1000B"),
    ],
)
def test_future_heavyweight_ids_default_to_server(model_id, expected_parameters):
    spec = infer_model_spec(model_id)
    assert spec.local is False
    assert spec.provider == "self-hosted"
    assert spec.parameter_count == expected_parameters
    with pytest.raises(ValueError, match="server-backed"):
        AgentFactory._infer_runner_type(model_id)


def test_explicit_runner_override_allows_large_local_model():
    context = MagicMock()
    with patch("agents.local_agent.LocalAgent") as local_agent:
        AgentFactory.create_agent(
            "local",
            context,
            model="future-org/Future-70B-Instruct",
            runner_type="transformers",
        )
    assert local_agent.call_args.kwargs["runner_type"] == "transformers"
    assert local_agent.call_args.kwargs["device_config"]["allow_server_backed"] is True


def test_existing_llama_id_preserves_optimized_runner():
    assert AgentFactory._infer_runner_type(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ) == "mlx_llama"


@pytest.mark.parametrize("model_id", [
    "kimi-k2.5",
    "moonshotai/Kimi-K2.5",
    "glm-4.7-flash",
    "zai-org/GLM-4.7-Flash",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-oss-120b",
    "zai-org/GLM-5.2",
    "deepseek-ai/DeepSeek-R1",
])
def test_server_model_cannot_be_accidentally_loaded_locally(model_id):
    with pytest.raises(ValueError, match="server-backed"):
        AgentFactory._infer_runner_type(model_id)


@pytest.mark.parametrize("model_id", ["kimi-k2.5", "moonshotai/Kimi-K2.5"])
def test_server_model_infers_openai_compatible_provider_endpoint(model_id):
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model=model_id)
    assert online_agent.call_args.kwargs["base_url"] == "https://api.moonshot.ai/v1"


def test_hosted_glm_infers_zai_endpoint():
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model="glm-4.7-flash")
    assert online_agent.call_args.kwargs["base_url"] == "https://api.z.ai/api/paas/v4"


def test_self_hosted_model_requires_endpoint_or_environment():
    context = MagicMock()
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="OPENAI_COMPATIBLE_BASE_URL"),
    ):
        AgentFactory.create_agent("online", context, model="openai/gpt-oss-120b")


def test_self_hosted_model_uses_environment_endpoint():
    context = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_BASE_URL": "http://inference:8000/v1/"},
            clear=True,
        ),
        patch("agents.online_agent.OnlineAgent") as online_agent,
    ):
        AgentFactory.create_agent("online", context, model="openai/gpt-oss-120b")
    assert online_agent.call_args.kwargs["base_url"] == "http://inference:8000/v1"


def test_lazy_registry_does_not_import_every_backend():
    from agents.architectures.registry import clear_registry, get_registry, register_all_runners

    clear_registry()
    register_all_runners()
    raw_values = get_registry().list()
    assert isinstance(raw_values["mlx_llama"], tuple)
    assert isinstance(raw_values["transformers"], tuple)


def test_model_api_metadata_contains_performance_fields():
    from agents.architectures.registry import (
        OnlineModelProviders,
        get_model_by_id,
        get_models_for_provider,
    )

    local_models = get_models_for_provider(OnlineModelProviders.OFFLINE)
    local_glm = next(
        model for model in local_models if model.id == "zai-org/glm-4-9b-chat-hf"
    )
    assert local_glm.backend == "transformers"
    assert local_glm.local is True

    hosted_models = get_models_for_provider("self-hosted")
    glm = next(model for model in hosted_models if model.id == "zai-org/GLM-4.7-Flash")
    payload = glm.to_dict()
    assert payload["backend"] == "openai_compatible"
    assert payload["supports_reasoning"] is True
    assert payload["activated_parameters"] == "3B"
    assert "vLLM/SGLang" in payload["performance_note"]

    gpt_oss = get_model_by_id("openai/gpt-oss-20b")
    assert gpt_oss.optional_dependencies == ("kernels",)

    # Mistral also exists in the legacy Hugging Face provider list. Direct
    # lookup should return the catalog-enriched local record.
    mistral = get_model_by_id("mistralai/Mistral-7B-Instruct-v0.3")
    assert mistral.backend == "transformers"
    assert mistral.local is True
    assert mistral.provider == OnlineModelProviders.OFFLINE


def test_future_provider_does_not_require_enum_change(monkeypatch):
    from agents.architectures.registry import (
        STATIC_MODELS,
        ModelInfo,
        get_all_models,
        get_provider_ids,
        provider_from_string,
    )

    monkeypatch.setitem(
        PROVIDERS,
        "future-provider",
        ProviderSpec(
            "future-provider",
            "Future Provider",
            "https://future.example/v1",
            "FUTURE_API_KEY",
        ),
    )
    model = ModelInfo(
        id="future-model",
        name="Future Model",
        provider="future-provider",
    )
    monkeypatch.setitem(STATIC_MODELS, "future-provider", [model])

    assert provider_from_string("future-provider") == "future-provider"
    assert "future-provider" in get_provider_ids()
    assert get_all_models()["future-provider"] == [model]
    assert model.to_dict()["provider"] == "future-provider"


def test_model_routes_serialize_string_backed_providers():
    from app.api.v1.endpoints.models import (
        get_available_models,
        get_models_by_provider,
        get_providers,
    )

    provider_ids = asyncio.run(get_providers())
    assert "self-hosted" in provider_ids
    assert "moonshot" in provider_ids

    response = asyncio.run(get_available_models())
    assert "self-hosted" in response.providers
    assert any(
        model.id == "openai/gpt-oss-120b"
        for model in response.providers["self-hosted"]
    )

    hosted_models = asyncio.run(get_models_by_provider("self-hosted"))
    assert all(model.provider == "self-hosted" for model in hosted_models)
