"""Capability and provider metadata for supported model families.

Adding a standard causal language model should normally require only a
``ModelSpec`` entry here.  Runner selection consumes backend metadata instead
of growing model-name conditionals throughout the application.
"""
from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass


MLX_DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.8-27B"
GGUF_DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-4B"


def default_local_model_id() -> str:
    """Keep the existing MLX default on Apple silicon; use curated GGUF elsewhere."""

    if platform.system().lower() == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }:
        return MLX_DEFAULT_LOCAL_MODEL
    return GGUF_DEFAULT_LOCAL_MODEL


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    base_url: str
    api_key_env: str
    openai_compatible: bool = True
    base_url_env: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    provider: str = "offline"
    backend: str = "transformers"
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_reasoning: bool = False
    supports_streaming: bool = False
    recommended: bool = False
    gated: bool = False
    requires_remote_code: bool = False
    min_transformers_version: str | None = None
    parameter_count: str | None = None
    activated_parameters: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    unsupported_parameters: tuple[str, ...] = ()
    mandatory_reasoning_effort: str | None = None
    local: bool = True
    performance_note: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "google": ProviderSpec(
        "google",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "groq": ProviderSpec("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "xai": ProviderSpec("xai", "xAI", "https://api.x.ai/v1", "GROK_API_KEY"),
    "moonshot": ProviderSpec(
        "moonshot", "Moonshot AI", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"
    ),
    "zai": ProviderSpec(
        "zai", "Z.AI", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
    ),
    "openrouter": ProviderSpec(
        "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
    "meta": ProviderSpec(
        "meta", "Meta Model API", "https://api.meta.ai/v1", "MODEL_API_KEY"
    ),
    "self-hosted": ProviderSpec(
        "self-hosted",
        "Self-hosted OpenAI-compatible",
        "",
        "API_KEY",
        base_url_env="OPENAI_COMPATIBLE_BASE_URL",
    ),
}


MODEL_SPECS: tuple[ModelSpec, ...] = (
    # Existing families and practical local reference checkpoints.
    ModelSpec(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", "Meta Llama 3.1 8B Instruct (Local)",
        "llama", backend="mlx_llama", context_window=131072,
        max_output_tokens=8192, recommended=True, gated=True,
        parameter_count="8B",
        performance_note="Optimized MLX path on Apple Silicon; use an explicit transformers override for HF-native loading.",
    ),
    ModelSpec(
        "Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B Instruct (Local)", "qwen",
        context_window=32768, max_output_tokens=8192, supports_function_calling=True,
        recommended=True, parameter_count="3B",
        performance_note="Good laptop baseline; use 4-bit weights when available.",
    ),
    ModelSpec(
        "Qwen/Qwen3-4B", "Qwen 3 4B (Local)", "qwen",
        context_window=32768, max_output_tokens=8192, supports_function_calling=True,
        supports_reasoning=True, recommended=True,
        min_transformers_version="4.51.0", parameter_count="4B",
        performance_note="Use non-thinking mode for lower latency when reasoning is unnecessary.",
    ),
    ModelSpec(
        "Qwen/Qwen3-8B", "Qwen 3 8B (Local)", "qwen",
        context_window=32768, max_output_tokens=8192, supports_function_calling=True,
        supports_reasoning=True, min_transformers_version="4.51.0", parameter_count="8B",
        performance_note="Use 4-bit weights and non-thinking mode for lower local latency.",
    ),
    ModelSpec(
        "Qwen/Qwen3.8-27B", "Qwen 3.8 27B (Local, MLX 4-bit)", "qwen",
        backend="mlx_llama", context_window=262144, max_output_tokens=8192,
        supports_function_calling=True, supports_reasoning=True,
        min_transformers_version="5.8.0", parameter_count="27B",
        performance_note=(
            "Text-only local inference on Apple Silicon using the curated "
            "mlx-community/Qwen3.8-27B-4bit snapshot."
        ),
    ),
    ModelSpec(
        "Qwen/Qwen3-1.7B", "Qwen 3 1.7B (Local)", "qwen",
        context_window=32768, max_output_tokens=8192, supports_reasoning=True,
        min_transformers_version="4.51.0", parameter_count="1.7B",
        performance_note="Small laptop-friendly reference model.",
    ),
    ModelSpec(
        "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3 (Local)",
        "mistral", context_window=32768, max_output_tokens=8192,
        supports_function_calling=True, recommended=True, parameter_count="7B",
        performance_note="Quantized MLX/GGUF weights are recommended on laptops.",
    ),
    ModelSpec(
        "microsoft/Phi-4-mini-instruct", "Phi 4 Mini Instruct (Local)", "phi",
        context_window=131072, max_output_tokens=8192, supports_function_calling=True,
        recommended=True, min_transformers_version="4.49.0", parameter_count="3.8B",
        performance_note="Strong small-model option; ONNX/INT4 variants improve CPU latency.",
    ),
    ModelSpec(
        "HuggingFaceTB/SmolLM3-3B", "SmolLM3 3B (Local)", "smollm",
        context_window=131072, max_output_tokens=8192, supports_function_calling=True,
        supports_reasoning=True,
        recommended=True, min_transformers_version="4.53.0", parameter_count="3B",
        performance_note="Lightweight reference model suitable for local smoke tests.",
    ),
    ModelSpec(
        "google/gemma-3-1b-it", "Gemma 3 1B IT (Local)", "gemma",
        context_window=32768, max_output_tokens=8192, recommended=True, gated=True,
        min_transformers_version="4.50.0", parameter_count="1B",
        performance_note="Very small footprint; Hugging Face license acceptance is required.",
    ),
    ModelSpec(
        "ibm-granite/granite-3.3-8b-instruct", "Granite 3.3 8B Instruct (Local)",
        "granite", context_window=131072, max_output_tokens=8192,
        supports_function_calling=True, supports_reasoning=True, parameter_count="8B",
        performance_note="Prefer quantized weights or a local OpenAI-compatible server.",
    ),
    ModelSpec(
        "allenai/Olmo-3-7B-Instruct", "OLMo 3 7B Instruct (Local)", "olmo",
        context_window=65536, max_output_tokens=32768, supports_function_calling=True,
        supports_reasoning=True,
        min_transformers_version="4.57.0", parameter_count="7B",
        performance_note="Prefer quantized weights for laptop deployment.",
    ),
    ModelSpec(
        "zai-org/glm-4-9b-chat-hf", "GLM 4 9B Chat HF (Local)", "glm",
        context_window=131072, max_output_tokens=8192,
        supports_function_calling=True, min_transformers_version="4.46.0",
        parameter_count="9B",
        performance_note="Transformers-native GLM baseline; prefer 4-bit weights on laptops.",
    ),
    ModelSpec(
        "openai/gpt-oss-20b", "gpt-oss 20B (Local/Server)", "gpt-oss",
        context_window=131072, max_output_tokens=32768, supports_function_calling=True,
        supports_reasoning=True, min_transformers_version="4.55.0",
        parameter_count="21B", activated_parameters="3.6B",
        optional_dependencies=("kernels",),
        performance_note="Native MXFP4 fits near 16 GB; preserve the tokenizer Harmony template.",
    ),
    ModelSpec(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill Qwen 7B (Local)",
        "deepseek-distill", context_window=131072, max_output_tokens=32768,
        supports_reasoning=True, parameter_count="7B",
        performance_note="Uses the underlying Qwen architecture and generic runner.",
    ),
    ModelSpec(
        "x-ai/grok-4.6", "Grok 4.6", "grok", provider="openrouter",
        backend="openai_compatible", context_window=500000,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True,
        unsupported_parameters=("frequency_penalty", "presence_penalty", "stop"),
        mandatory_reasoning_effort="high", local=False,
        performance_note=(
            "Hosted through OpenRouter; mandatory reasoning defaults to high. "
            "Enable OpenRouter ZDR routing before sending confidential workloads."
        ),
    ),
    # Heavyweight models are deliberately server-backed.
    ModelSpec(
        "zai-org/GLM-4.7-Flash", "GLM 4.7 Flash (Self-hosted)", "glm",
        provider="self-hosted", backend="openai_compatible",
        context_window=131072, max_output_tokens=131072,
        supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, parameter_count="31B",
        activated_parameters="3B", local=False,
        performance_note="Serve with vLLM/SGLang; sparse compute does not remove the 31B weight-residency cost.",
    ),
    ModelSpec(
        "meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B (Self-hosted)",
        "llama", provider="self-hosted", backend="openai_compatible",
        context_window=131072, max_output_tokens=32768,
        supports_function_calling=True, supports_streaming=True,
        parameter_count="70B", local=False,
        performance_note="Use a quantized runtime or a multi-GPU OpenAI-compatible server.",
    ),
    ModelSpec(
        "Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B (Self-hosted)",
        "qwen", provider="self-hosted", backend="openai_compatible",
        context_window=131072, max_output_tokens=32768,
        supports_function_calling=True, supports_streaming=True,
        parameter_count="72B", local=False,
        performance_note="Use a quantized runtime or a multi-GPU OpenAI-compatible server.",
    ),
    ModelSpec(
        "mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B (Self-hosted)",
        "mistral", provider="self-hosted", backend="openai_compatible",
        context_window=32768, max_output_tokens=8192,
        supports_function_calling=True, supports_streaming=True,
        parameter_count="46.7B", activated_parameters="12.9B", local=False,
        performance_note="MoE compute is sparse, but all expert weights still require residency.",
    ),
    ModelSpec(
        "openai/gpt-oss-120b", "gpt-oss 120B (Self-hosted)", "gpt-oss",
        provider="self-hosted", backend="openai_compatible",
        context_window=131072, max_output_tokens=32768,
        supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, min_transformers_version="4.55.0",
        parameter_count="117B", activated_parameters="5.1B", local=False,
        performance_note="MXFP4 still targets roughly 80 GB of accelerator memory.",
    ),
    ModelSpec(
        "gemini-3.8-flash", "Gemini 3.8 Flash", "gemini", provider="google",
        backend="openai_compatible", context_window=1048576, max_output_tokens=65536,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True, local=False,
        unsupported_parameters=("n", "temperature", "top_p"),
        performance_note=(
            "Google's stable Gemini 3.8 Flash model via the beta OpenAI-compatible "
            "endpoint. Unpaid-tier data may be used for product improvement and human "
            "review; use paid services for confidential workloads."
        ),
    ),
    ModelSpec(
        "kimi-k2.5", "Kimi K2.5", "kimi", provider="moonshot",
        backend="openai_compatible", context_window=262144, max_output_tokens=65536,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True, requires_remote_code=True,
        min_transformers_version="4.57.1", parameter_count="1T",
        activated_parameters="32B", local=False,
        performance_note="Server-only baseline; the official repository is hundreds of GB.",
    ),
    ModelSpec(
        "glm-5.2", "GLM 5.2", "glm", provider="zai", backend="openai_compatible",
        context_window=1000000, max_output_tokens=131072,
        supports_function_calling=True, supports_reasoning=True, supports_streaming=True,
        recommended=True, parameter_count="753B", local=False,
        performance_note="Use Z.AI or a multi-GPU OpenAI-compatible deployment.",
    ),
    ModelSpec(
        "glm-4.7-flash", "GLM 4.7 Flash (Hosted)", "glm", provider="zai",
        backend="openai_compatible", context_window=131072,
        max_output_tokens=131072, supports_function_calling=True,
        supports_reasoning=True, supports_streaming=True, local=False,
        parameter_count="31B", activated_parameters="3B",
        performance_note="Hosted alternative to running the 31B Flash checkpoint on your own server.",
    ),
    ModelSpec(
        "qwen/qwen3.8-max", "Qwen 3.8 Max", "qwen", provider="openrouter",
        backend="openai_compatible", context_window=1000000, max_output_tokens=131072,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True, parameter_count="2.4T",
        activated_parameters="95B", local=False,
        performance_note="Hosted-only flagship MoE; OpenRouter proxies Alibaba's endpoint until open weights ship.",
    ),
    ModelSpec(
        "qwen/qwen3.8-flash", "Qwen 3.8 Flash", "qwen", provider="openrouter",
        backend="openai_compatible", context_window=1000000, max_output_tokens=131072,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True, unsupported_parameters=("n",),
        local=False,
        performance_note=(
            "Low-cost Alibaba route with optional reasoning and structured outputs. "
            "The current endpoint is not OpenRouter ZDR; do not use it for "
            "confidential workloads."
        ),
    ),
    ModelSpec(
        "tencent/hy4-preview", "Tencent Hy4 Preview", "hy", provider="openrouter",
        backend="openai_compatible", context_window=1048576, max_output_tokens=64000,
        supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True, parameter_count="770B",
        activated_parameters="49B",
        unsupported_parameters=("n", "top_p", "frequency_penalty", "presence_penalty"),
        local=False,
        performance_note=(
            "Single Tencent FP8 preview route on OpenRouter with optional reasoning, "
            "native tools, and structured outputs. Enforce OpenRouter ZDR routing "
            "and review preview status before confidential workloads."
        ),
    ),
    ModelSpec(
        "z-ai/glm-5.3-flash", "GLM 5.3 Flash", "glm", provider="openrouter",
        backend="openai_compatible", context_window=1048576, max_output_tokens=131072,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True,
        parameter_count="320B", activated_parameters="18B",
        mandatory_reasoning_effort="max", local=False,
        performance_note=(
            "Hosted through OpenRouter; Geist applies Z.AI's recommended max reasoning "
            "effort. Enable OpenRouter ZDR routing for confidential workloads."
        ),
    ),
    ModelSpec(
        "meta/muse-spark-1.2-contributor", "Muse Spark 1.2 Contributor", "muse",
        provider="openrouter", backend="openai_compatible", context_window=1048576,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True,
        unsupported_parameters=("frequency_penalty", "presence_penalty", "stop"),
        mandatory_reasoning_effort="medium", local=False,
        performance_note=(
            "Low-cost Meta contributor tier with mandatory reasoning. Meta retains "
            "requests for 30 days; do not use for confidential workloads."
        ),
    ),
    ModelSpec(
        "muse-spark-1.1", "Muse Spark 1.1", "muse", provider="meta",
        backend="openai_compatible", context_window=1048576,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True,
        unsupported_parameters=("frequency_penalty", "presence_penalty", "stop", "n"),
        local=False,
        performance_note="Earlier Meta Model API release for agentic multimodal work.",
    ),
    ModelSpec(
        "muse-spark-1.2", "Muse Spark 1.2", "muse", provider="meta",
        backend="openai_compatible", context_window=1048576,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True,
        unsupported_parameters=("frequency_penalty", "presence_penalty", "stop", "n"),
        local=False,
        performance_note="Meta-hosted coding-focused Muse Spark release.",
    ),
    ModelSpec(
        "muse-spark-1.3", "Muse Spark 1.3", "muse", provider="meta",
        backend="openai_compatible", context_window=1048576,
        supports_vision=True, supports_function_calling=True, supports_reasoning=True,
        supports_streaming=True, recommended=True,
        unsupported_parameters=("frequency_penalty", "presence_penalty", "stop", "n"),
        local=False,
        performance_note=(
            "Hosted by Meta Model API for long-horizon agentic and coding work; "
            "open weights have not been released."
        ),
    ),
    ModelSpec(
        "deepseek-reasoner", "DeepSeek R1", "deepseek", provider="deepseek",
        backend="openai_compatible", context_window=131072, max_output_tokens=32768,
        supports_function_calling=True, supports_reasoning=True, supports_streaming=True,
        parameter_count="671B", activated_parameters="37B", local=False,
        performance_note="Use the hosted API or a specialized multi-GPU deployment.",
    ),
)


_MODEL_INDEX = {spec.id.lower(): spec for spec in MODEL_SPECS}


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Return exact catalog metadata using case-insensitive model IDs."""
    return _MODEL_INDEX.get(model_id.lower())


def infer_model_spec(model_id: str) -> ModelSpec | None:
    """Resolve known fine-tunes/quantizations to a family-level catalog entry."""
    exact = get_model_spec(model_id)
    if exact:
        return exact

    value = model_id.lower()
    if value in {"models/gemini-3.8-flash", "google/gemini-3.8-flash"}:
        return get_model_spec("gemini-3.8-flash")
    # Map heavyweight Hugging Face repository IDs to their server-backed
    # catalog entries so they can never fall through to an accidental local
    # trillion-parameter load.
    if "qwen3.8-max" in value:
        return get_model_spec("qwen/qwen3.8-max")
    if "moonshotai/kimi-k2" in value:
        return get_model_spec("kimi-k2.5")
    if "zai-org/glm-5" in value:
        return get_model_spec("glm-5.2")
    if "deepseek-ai/deepseek-r1" in value and "distill" not in value:
        return get_model_spec("deepseek-reasoner")
    if "gpt-oss-120b" in value:
        return get_model_spec("openai/gpt-oss-120b")
    heavyweight_parameters = _heavyweight_parameter_count(value)
    if heavyweight_parameters is not None:
        return ModelSpec(
            id=model_id,
            name=f"{model_id} (Self-hosted)",
            family="large-model",
            provider="self-hosted",
            backend="openai_compatible",
            parameter_count=f"{heavyweight_parameters:g}B",
            local=False,
            performance_note=(
                "Automatically routed to a server because the model ID indicates "
                "at least 32B total parameters."
            ),
        )
    prefix_map = (
        (("qwen/", "qwen", "deepseek-r1-distill-qwen"), "Qwen/Qwen3-4B"),
        (("mistralai/", "mistral", "mixtral"), "mistralai/Mistral-7B-Instruct-v0.3"),
        (("microsoft/phi", "phi-"), "microsoft/Phi-4-mini-instruct"),
        (("huggingfacetb/smollm", "smollm"), "HuggingFaceTB/SmolLM3-3B"),
        (("google/gemma", "gemma"), "google/gemma-3-1b-it"),
        (("ibm-granite/", "granite"), "ibm-granite/granite-3.3-8b-instruct"),
        (("allenai/olmo", "olmo"), "allenai/Olmo-3-7B-Instruct"),
        (("zai-org/glm-4", "thudm/glm-4", "glm-4"), "zai-org/glm-4-9b-chat-hf"),
        (("openai/gpt-oss", "gpt-oss"), "openai/gpt-oss-20b"),
    )
    for prefixes, reference_id in prefix_map:
        if any(prefix in value for prefix in prefixes):
            return get_model_spec(reference_id)
    return None


def _heavyweight_parameter_count(model_id: str) -> float | None:
    """Infer clearly large total parameter counts from conventional model IDs."""
    moe_match = re.search(r"(?:^|[-_/])(\d+)x(\d+(?:\.\d+)?)b(?:[-_/]|$)", model_id)
    if moe_match:
        total = int(moe_match.group(1)) * float(moe_match.group(2))
        return total if total >= 32 else None

    trillion_matches = [
        float(value) * 1000
        for value in re.findall(r"(?:^|[-_/])(\d+(?:\.\d+)?)t(?:[-_/]|$)", model_id)
    ]
    billion_matches = [
        float(value)
        for value in re.findall(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:[-_/]|$)", model_id)
    ]
    candidates = trillion_matches + billion_matches
    if not candidates:
        return None
    total = max(candidates)
    return total if total >= 32 else None


def get_provider_spec(provider_id: str) -> ProviderSpec | None:
    return PROVIDERS.get(provider_id.lower())


def get_provider_endpoint(provider_id: str) -> str | None:
    provider = get_provider_spec(provider_id)
    if not provider:
        return None
    if provider.base_url_env:
        configured = os.getenv(provider.base_url_env)
        if configured:
            return configured.rstrip("/")
    return provider.base_url or None
