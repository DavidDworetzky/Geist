"""
Speech-to-Text (STT) provider catalog and factory.

Mirrors the TTS provider pattern in app/services/tts.py: published metadata
for the frontend, a model-id allowlist (the model string is client-controlled
and reaches model loaders), and a string-keyed factory.
"""

import logging
from typing import Any, Protocol

from adapters.faster_whisper_adapter import (
    DEFAULT_FASTER_WHISPER_MODEL,
    FasterWhisperAdapter,
)
from adapters.mms_adapter import MMSAdapter
from adapters.parakeet_adapter import DEFAULT_PARAKEET_MODEL, ParakeetAdapter
from adapters.whisper_adapter import WhisperAdapter


logger = logging.getLogger(__name__)


class STTAdapter(Protocol):
    """Structural type for STT adapters: anything with a transcribe method."""

    def transcribe(self, audio_data: Any, language: str = "en") -> str: ...


SUPPORTED_STT_PROVIDERS: list[dict[str, Any]] = [
    {
        "provider": "mms",
        "display_name": "Meta MMS",
        "type": "local",
        "default_model": "facebook/mms-1b-all",
        "models": [
            {
                "id": "facebook/mms-1b-all",
                "display_name": "MMS 1B (all languages)",
                "sample_rate": 16000,
            }
        ],
    },
    {
        "provider": "whisper",
        "display_name": "OpenAI Whisper API",
        "type": "api",
        "default_model": "whisper-1",
        "models": [
            {
                "id": "whisper-1",
                "display_name": "Whisper (OpenAI API)",
                "sample_rate": 16000,
            }
        ],
    },
    {
        "provider": "faster_whisper",
        "display_name": "Whisper (local, faster-whisper)",
        "type": "local",
        "default_model": DEFAULT_FASTER_WHISPER_MODEL,
        "models": [
            {
                "id": "large-v3-turbo",
                "display_name": "Whisper large-v3-turbo",
                "sample_rate": 16000,
            },
            {
                "id": "large-v3",
                "display_name": "Whisper large-v3",
                "sample_rate": 16000,
            },
            {
                "id": "distil-large-v3",
                "display_name": "Distil-Whisper large-v3",
                "sample_rate": 16000,
            },
            {
                "id": "medium",
                "display_name": "Whisper medium",
                "sample_rate": 16000,
            },
            {
                "id": "small",
                "display_name": "Whisper small",
                "sample_rate": 16000,
            },
            {
                "id": "base",
                "display_name": "Whisper base",
                "sample_rate": 16000,
            },
            {
                "id": "tiny",
                "display_name": "Whisper tiny",
                "sample_rate": 16000,
            },
        ],
    },
    {
        "provider": "parakeet",
        "display_name": "NVIDIA Parakeet",
        "type": "local",
        "default_model": DEFAULT_PARAKEET_MODEL,
        "models": [
            {
                "id": DEFAULT_PARAKEET_MODEL,
                "display_name": "Parakeet TDT 0.6B v3",
                "sample_rate": 16000,
            },
            {
                "id": "nvidia/parakeet-tdt-0.6b-v2",
                "display_name": "Parakeet TDT 0.6B v2",
                "sample_rate": 16000,
            },
        ],
    },
]


def get_supported_stt_providers() -> list[dict[str, Any]]:
    """Return frontend-consumable metadata for supported STT providers."""
    return SUPPORTED_STT_PROVIDERS


def allowed_models_for_stt_provider(provider_type: str) -> list[str]:
    for entry in SUPPORTED_STT_PROVIDERS:
        if entry["provider"] == provider_type:
            return [model["id"] for model in entry.get("models", [])]
    return []


def _validate_stt_model(provider_type: str, model: str) -> str:
    """
    Reject model ids that are not in the provider's published list.

    The model string is client-controlled and reaches model loaders
    (e.g. from_pretrained), so it must never be an arbitrary repo id or
    filesystem path.
    """
    allowed = allowed_models_for_stt_provider(provider_type)
    if allowed and model not in allowed:
        raise ValueError(
            f"Model '{model}' is not a supported {provider_type} STT model. "
            f"Supported models: {allowed}"
        )
    return model


def create_stt_adapter(provider_type: str = "mms", **kwargs) -> STTAdapter:
    """
    Factory function to create an STT adapter.

    Args:
        provider_type: "mms", "whisper", "faster_whisper", or "parakeet"
        **kwargs: Provider-specific arguments (whisper_api_key, stt_model,
            stt_device)
    """
    provider = provider_type.lower().replace("-", "_")

    if provider == "mms":
        return MMSAdapter()
    if provider == "whisper":
        return WhisperAdapter(api_key=kwargs.get("whisper_api_key"))
    if provider == "faster_whisper":
        model = _validate_stt_model(
            "faster_whisper", kwargs.get("stt_model") or DEFAULT_FASTER_WHISPER_MODEL
        )
        return FasterWhisperAdapter(model=model, device=kwargs.get("stt_device"))
    if provider == "parakeet":
        model = _validate_stt_model(
            "parakeet", kwargs.get("stt_model") or DEFAULT_PARAKEET_MODEL
        )
        return ParakeetAdapter(model=model)

    raise ValueError(f"Unknown STT provider: {provider_type}")
