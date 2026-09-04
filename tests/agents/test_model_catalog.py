"""Tests for generic model/provider catalog and runner routing."""
import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from agents.factory import AgentFactory
from agents.model_catalog import (
    MODEL_SPECS,
    PROVIDERS,
    ProviderSpec,
    default_local_model_id,
    get_model_spec,
    get_provider_endpoint,
    infer_model_spec,
    resolve_request_spec,
)


def test_platform_default_uses_qwen38_mlx_and_uses_gguf_on_windows():
    with (
        patch("agents.model_catalog.platform.system", return_value="Darwin"),
        patch("agents.model_catalog.platform.machine", return_value="arm64"),
    ):
        assert default_local_model_id() == "Qwen/Qwen3.8-27B"

    with patch("agents.model_catalog.platform.system", return_value="Windows"):
        assert default_local_model_id() == "Qwen/Qwen3-4B"


def test_catalog_covers_requested_families():
    families = {spec.family for spec in MODEL_SPECS}
    assert {
        "llama", "qwen", "mistral", "phi", "smollm", "gemma",
        "granite", "olmo", "glm", "gpt-oss", "kimi", "deepseek", "gemini",
    }.issubset(families)


def test_qwen3_8_27b_declares_runtime_compatibility():
    spec = get_model_spec("Qwen/Qwen3.8-27B")

    assert spec.local is True
    assert spec.backend == "mlx_llama"
    assert spec.context_window == 262144
    assert spec.parameter_count == "27B"
    assert spec.min_transformers_version == "5.8.0"
    assert spec.supports_vision is False


@pytest.mark.parametrize(
    "model_id",
    ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B"],
)
def test_qwen3_models_declare_architecture_runtime_compatibility(model_id):
    assert get_model_spec(model_id).min_transformers_version == "4.51.0"


def test_registry_only_qwen3_model_declares_runtime_compatibility():
    from agents.architectures.registry import OnlineModelProviders, get_models_for_provider

    offline_models = get_models_for_provider(OnlineModelProviders.OFFLINE)
    qwen3_06b = next(model for model in offline_models if model.id == "Qwen/Qwen3-0.6B")

    assert qwen3_06b.min_transformers_version == "4.51.0"


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

    qwen_max = get_model_spec("qwen/qwen3.8-max")
    assert qwen_max.backend == "openai_compatible"
    assert qwen_max.local is False
    assert get_provider_endpoint(qwen_max.provider) == "https://openrouter.ai/api/v1"

    glm_flash = get_model_spec("z-ai/glm-5.3-flash")
    assert glm_flash.backend == "openai_compatible"
    assert glm_flash.provider == "openrouter"
    assert glm_flash.local is False
    assert glm_flash.context_window == 1048576
    assert glm_flash.max_output_tokens == 131072
    assert glm_flash.parameter_count == "320B"
    assert glm_flash.activated_parameters == "18B"
    assert glm_flash.supports_vision is True
    assert glm_flash.supports_function_calling is True
    assert glm_flash.supports_reasoning is True
    assert glm_flash.supports_streaming is True
    assert glm_flash.recommended is True
    assert glm_flash.mandatory_reasoning_effort == "max"
    assert glm_flash.unsupported_parameters == ()
    assert get_provider_endpoint(glm_flash.provider) == "https://openrouter.ai/api/v1"
    assert get_model_spec("stealth/ox-alpha") is None

    muse = get_model_spec("meta/muse-spark-1.2-contributor")
    assert muse.backend == "openai_compatible"
    assert muse.provider == "openrouter"
    assert muse.local is False
    assert muse.context_window == 1048576
    assert muse.max_output_tokens is None
    assert muse.parameter_count is None
    assert muse.supports_vision is True
    assert muse.supports_function_calling is True
    assert muse.supports_reasoning is True
    assert muse.supports_streaming is True
    assert muse.mandatory_reasoning_effort == "medium"
    assert muse.unsupported_parameters == (
        "frequency_penalty",
        "presence_penalty",
        "stop",
    )


def test_qwen_max_id_variants_route_to_openrouter_not_local_qwen():
    from agents.architectures.registry import get_all_models, provider_from_string

    assert provider_from_string("openrouter") == "openrouter"
    assert "openrouter" in get_all_models()

    for model_id in ("qwen3.8-max", "alibaba/qwen3.8-max", "Qwen/Qwen3.8-Max"):
        spec = infer_model_spec(model_id)
        assert spec.provider == "openrouter"
        assert spec.local is False


def test_google_gemini38_flash_metadata_is_explicit_and_server_backed():
    flash = get_model_spec("gemini-3.8-flash")

    assert flash.provider == "google"
    assert flash.backend == "openai_compatible"
    assert flash.local is False
    assert flash.context_window == 1048576
    assert flash.max_output_tokens == 65536
    assert flash.supports_vision is True
    assert flash.supports_function_calling is True
    assert flash.supports_reasoning is True
    assert flash.supports_streaming is True
    assert flash.recommended is True
    assert flash.mandatory_reasoning_effort is None
    assert flash.unsupported_parameters == ("n", "temperature", "top_p")
    assert flash.performance_note is not None
    assert "migration checklist by omitting n, temperature, and top_p" in (
        flash.performance_note
    )
    assert get_provider_endpoint(flash.provider) == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


def test_openai_gpt6_astra_metadata_is_explicit_and_rollout_aware():
    astra = get_model_spec("gpt-6-astra")

    assert astra.provider == "openai"
    assert astra.backend == "openai_compatible"
    assert astra.local is False
    assert astra.family == "gpt-6"
    assert astra.context_window == 1050000
    assert astra.max_output_tokens == 128000
    assert astra.supports_vision is True
    assert astra.supports_function_calling is True
    assert astra.supports_reasoning is True
    assert astra.supports_streaming is True
    assert astra.recommended is False
    assert astra.mandatory_reasoning_effort is None
    assert astra.unsupported_parameters == (
        "temperature",
        "top_p",
        "top_logprobs",
        "logprobs",
    )
    assert astra.performance_note is not None
    assert "not yet generally available" in astra.performance_note
    assert "Responses API" in astra.performance_note
    assert get_provider_endpoint(astra.provider) == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    "model_id",
    [
        "models/gemini-3.8-flash",
        "google/gemini-3.8-flash",
        "models/gemini-3.8-flash-latest",
        "google/gemini-3.8-flash-preview",
        "gemini-3.8-flash-lite",
    ],
)
def test_gemini38_id_variants_route_to_google(model_id):
    spec = infer_model_spec(model_id)

    assert spec.id == "gemini-3.8-flash"
    assert spec.provider == "google"
    assert spec.local is False


def test_gemini_request_aliases_share_the_canonical_request_contract():
    flash = get_model_spec("gemini-3.8-flash")

    assert flash.aliases
    assert all(resolve_request_spec(alias) is flash for alias in flash.aliases)


def test_catalog_aliases_are_unique_and_do_not_shadow_canonical_ids():
    canonical_ids = {spec.id.lower() for spec in MODEL_SPECS}
    aliases = [alias.lower() for spec in MODEL_SPECS for alias in spec.aliases]

    assert len(aliases) == len(set(aliases))
    assert canonical_ids.isdisjoint(aliases)


def test_openrouter_qwen38_flash_metadata_is_explicit_and_server_backed():
    flash = get_model_spec("qwen/qwen3.8-flash")

    assert flash.provider == "openrouter"
    assert flash.backend == "openai_compatible"
    assert flash.local is False
    assert flash.context_window == 1000000
    assert flash.max_output_tokens == 131072
    assert flash.parameter_count is None
    assert flash.activated_parameters is None
    assert flash.supports_vision is True
    assert flash.supports_function_calling is True
    assert flash.supports_reasoning is True
    assert flash.supports_streaming is True
    assert flash.mandatory_reasoning_effort is None
    assert flash.unsupported_parameters == ("n",)
    assert get_provider_endpoint(flash.provider) == "https://openrouter.ai/api/v1"


def test_openrouter_hy4_preview_metadata_is_explicit_and_server_backed():
    hy4 = get_model_spec("tencent/hy4-preview")

    assert hy4.provider == "openrouter"
    assert hy4.backend == "openai_compatible"
    assert hy4.local is False
    assert hy4.family == "hy"
    assert hy4.context_window == 1048576
    assert hy4.max_output_tokens == 64000
    assert hy4.parameter_count == "770B"
    assert hy4.activated_parameters == "49B"
    assert hy4.supports_vision is False
    assert hy4.supports_function_calling is True
    assert hy4.supports_reasoning is True
    assert hy4.supports_streaming is True
    assert hy4.mandatory_reasoning_effort is None
    assert hy4.unsupported_parameters == (
        "n",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    )
    assert get_provider_endpoint(hy4.provider) == "https://openrouter.ai/api/v1"


def test_openrouter_grok_46_metadata_is_explicit_and_server_backed():
    grok = get_model_spec("x-ai/grok-4.6")

    assert grok.provider == "openrouter"
    assert grok.backend == "openai_compatible"
    assert grok.local is False
    assert grok.context_window == 500000
    assert grok.max_output_tokens is None
    assert grok.parameter_count is None
    assert grok.supports_vision is True
    assert grok.supports_function_calling is True
    assert grok.supports_reasoning is True
    assert grok.supports_streaming is True
    assert grok.mandatory_reasoning_effort == "high"
    assert grok.unsupported_parameters == (
        "frequency_penalty",
        "presence_penalty",
        "stop",
    )
    assert get_provider_endpoint(grok.provider) == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    ("model_id", "recommended"),
    [
        ("muse-spark-1.1", False),
        ("muse-spark-1.2", False),
        ("muse-spark-1.3", True),
    ],
)
def test_meta_model_api_catalog_options(model_id, recommended):
    model = get_model_spec(model_id)

    assert model.provider == "meta"
    assert model.backend == "openai_compatible"
    assert model.local is False
    assert model.context_window == 1048576
    assert model.max_output_tokens is None
    assert model.parameter_count is None
    assert model.supports_vision is True
    assert model.supports_function_calling is True
    assert model.supports_reasoning is True
    assert model.supports_streaming is True
    assert model.unsupported_parameters == (
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "n",
    )
    assert model.recommended is recommended
    assert get_provider_endpoint(model.provider) == "https://api.meta.ai/v1"


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3.8-27B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-1b-it",
    "ibm-granite/granite-3.3-8b-instruct",
    "allenai/Olmo-3-7B-Instruct",
    "zai-org/glm-4-9b-chat-hf",
    "openai/gpt-oss-20b",
])
def test_local_catalog_models_use_managed_runner_on_linux(model_id):
    with patch("agents.factory.sys.platform", "linux"):
        assert AgentFactory._infer_runner_type(model_id) == "llama_server"


@pytest.mark.parametrize(
    "host_platform, machine, model_id, expected_runner",
    [
        ("linux", "x86_64", "Qwen/Qwen3-4B", "llama_server"),
        ("win32", "AMD64", "Qwen/Qwen3-4B", "llama_server"),
        ("darwin", "arm64", "Qwen/Qwen3-4B", "mlx_llama"),
        ("darwin", "x86_64", "Qwen/Qwen3-4B", "transformers"),
        ("darwin", "x86_64", "Qwen/Qwen3.8-27B", "mlx_llama"),
    ],
)
def test_platform_runner_selection(host_platform, machine, model_id, expected_runner):
    with (
        patch("agents.factory.sys.platform", host_platform),
        patch("agents.factory.platform.machine", return_value=machine),
    ):
        assert AgentFactory._infer_runner_type(model_id) == expected_runner


def test_unknown_huggingface_model_uses_generic_runner():
    expected = "llama_server" if sys.platform in {"win32", "linux"} else "transformers"
    assert AgentFactory._infer_runner_type("new-org/future-causal-lm") == expected
    assert AgentFactory._infer_runner_type("new-org/future-8B-instruct") == expected


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


def test_environment_runner_override_precedes_catalog_inference():
    context = MagicMock()
    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": "transformers"}),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            context,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )

    assert local_agent.call_args.kwargs["runner_type"] == "transformers"


def test_explicit_runner_argument_precedes_environment_override():
    context = MagicMock()
    with (
        patch.dict(os.environ, {"GEIST_LOCAL_RUNNER": "transformers"}),
        patch("agents.local_agent.LocalAgent") as local_agent,
    ):
        AgentFactory.create_agent(
            "local",
            context,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            runner_type="mlx_llama",
        )

    assert local_agent.call_args.kwargs["runner_type"] == "mlx_llama"


def test_existing_llama_id_preserves_optimized_runner():
    expected = "llama_server" if sys.platform in {"win32", "linux"} else "mlx_llama"
    assert AgentFactory._infer_runner_type(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ) == expected


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
    "x-ai/grok-4.6",
    "qwen/qwen3.8-max",
    "qwen3.8-max",
    "qwen/qwen3.8-flash",
    "tencent/hy4-preview",
    "z-ai/glm-5.3-flash",
    "meta/muse-spark-1.2-contributor",
    "muse-spark-1.1",
    "muse-spark-1.2",
    "muse-spark-1.3",
    "gemini-3.8-flash",
    "gpt-6-astra",
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


@pytest.mark.parametrize(
    "model_id",
    [
        "gemini-3.8-flash",
        "models/gemini-3.8-flash",
        "google/gemini-3.8-flash",
        "models/gemini-3.8-flash-latest",
        "google/gemini-3.8-flash-preview",
    ],
)
def test_google_gemini_model_infers_compatible_endpoint(model_id):
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model=model_id)
    assert online_agent.call_args.kwargs["base_url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "x-ai/grok-4.6",
        "qwen/qwen3.8-max",
        "qwen3.8-max",
        "qwen/qwen3.8-flash",
        "tencent/hy4-preview",
        "z-ai/glm-5.3-flash",
        "meta/muse-spark-1.2-contributor",
    ],
)
def test_openrouter_model_infers_openrouter_endpoint(model_id):
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model=model_id)
    assert online_agent.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    "model_id",
    ["muse-spark-1.1", "muse-spark-1.2", "muse-spark-1.3"],
)
def test_meta_model_infers_first_party_endpoint(model_id):
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model=model_id)
    assert online_agent.call_args.kwargs["base_url"] == "https://api.meta.ai/v1"


def test_openai_astra_infers_first_party_endpoint():
    context = MagicMock()
    with patch("agents.online_agent.OnlineAgent") as online_agent:
        AgentFactory.create_agent("online", context, model="gpt-6-astra")
    assert online_agent.call_args.kwargs["base_url"] == "https://api.openai.com/v1"


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
    assert isinstance(raw_values["llama_server"], tuple)


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
    assert "openrouter" in provider_ids
    assert "meta" in provider_ids
    assert "google" in provider_ids

    response = asyncio.run(get_available_models())
    assert "self-hosted" in response.providers
    assert "openrouter" in response.providers
    assert "meta" in response.providers
    assert "google" in response.providers
    assert any(model.id == "gpt-6-astra" for model in response.providers["openai"])
    assert any(model.id == "gemini-3.8-flash" for model in response.providers["google"])
    assert any(model.id == "x-ai/grok-4.6" for model in response.providers["openrouter"])
    assert any(
        model.id == "qwen/qwen3.8-flash" for model in response.providers["openrouter"]
    )
    assert any(
        model.id == "tencent/hy4-preview" for model in response.providers["openrouter"]
    )
    assert any(
        model.id == "z-ai/glm-5.3-flash" for model in response.providers["openrouter"]
    )
    assert any(
        model.id == "meta/muse-spark-1.2-contributor"
        for model in response.providers["openrouter"]
    )
    assert {
        "muse-spark-1.1",
        "muse-spark-1.2",
        "muse-spark-1.3",
    } <= {model.id for model in response.providers["meta"]}
    assert any(
        model.id == "openai/gpt-oss-120b"
        for model in response.providers["self-hosted"]
    )

    hosted_models = asyncio.run(get_models_by_provider("self-hosted"))
    assert all(model.provider == "self-hosted" for model in hosted_models)
