"""Voice model metadata without loading inference runtimes."""

import copy
from typing import Any


DEFAULT_KOKORO_TTS_MODEL = "hexgrad/Kokoro-82M"
KOKORO_MLX_ARTIFACT_ID = "kokoro-82m-bf16-mlx"
KOKORO_DEFAULT_VOICE = "af_heart"
DEFAULT_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN3_MLX_ARTIFACT_ID = "qwen3-tts-0.6b-customvoice-mlx-6bit"
MAGPIE_TTS_MODEL = "nvidia/magpie_tts_multilingual_357m"
MAGPIE_ARTIFACT_ID = "magpie-tts-multilingual-357m-nemo-speech"

SUPPORTED_TTS_PROVIDERS: list[dict[str, Any]] = [
    {
        "provider": "sesame",
        "display_name": "Sesame CSM",
        "type": "local",
        "default_model": "sesame/csm-1b",
        "models": [
            {
                "id": "sesame/csm-1b",
                "display_name": "Sesame CSM 1B",
                "sample_rate": 24000,
                "supports_streaming": False,
                "streaming_mode": "chunked_full_audio",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "0", "display_name": "Default Speaker"}],
                "languages": [{"code": "en", "display_name": "English"}],
            }
        ],
    },
    {
        "provider": "openai",
        "display_name": "OpenAI TTS",
        "type": "api",
        "default_model": "gpt-4o-mini-tts",
        "models": [
            {
                "id": "gpt-4o-mini-tts",
                "display_name": "GPT-4o mini TTS",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "provider_dependent",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "alloy", "display_name": "Alloy"},
                    {"id": "echo", "display_name": "Echo"},
                    {"id": "fable", "display_name": "Fable"},
                    {"id": "onyx", "display_name": "Onyx"},
                    {"id": "nova", "display_name": "Nova"},
                    {"id": "shimmer", "display_name": "Shimmer"},
                ],
                "languages": [{"code": "auto", "display_name": "Auto"}],
            },
            {
                "id": "tts-1",
                "display_name": "TTS 1",
                "sample_rate": 24000,
                "supports_streaming": False,
                "streaming_mode": "chunked_full_audio",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "alloy", "display_name": "Alloy"},
                    {"id": "echo", "display_name": "Echo"},
                    {"id": "fable", "display_name": "Fable"},
                    {"id": "onyx", "display_name": "Onyx"},
                    {"id": "nova", "display_name": "Nova"},
                    {"id": "shimmer", "display_name": "Shimmer"},
                ],
                "languages": [{"code": "auto", "display_name": "Auto"}],
            },
        ],
    },
    {
        "provider": "kokoro",
        "display_name": "Vera (Kokoro local MLX)",
        "type": "local",
        "default_model": DEFAULT_KOKORO_TTS_MODEL,
        "models": [
            {
                "id": DEFAULT_KOKORO_TTS_MODEL,
                "display_name": "Vera · Kokoro 82M BF16",
                "artifact_id": KOKORO_MLX_ARTIFACT_ID,
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "sentence_buffered_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": KOKORO_DEFAULT_VOICE, "display_name": "Vera"}],
                "languages": [{"code": "a", "display_name": "English (US)"}],
            },
        ],
    },
    {
        "provider": "qwen3",
        "display_name": "Qwen3 TTS (local MLX)",
        "type": "local",
        "default_model": DEFAULT_QWEN3_TTS_MODEL,
        "models": [
            {
                "id": DEFAULT_QWEN3_TTS_MODEL,
                "display_name": "Qwen3 TTS 0.6B Custom Voice",
                "artifact_id": QWEN3_MLX_ARTIFACT_ID,
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "Aiden", "display_name": "Aiden"}],
                "languages": [{"code": "English", "display_name": "English"}],
            },
        ],
    },
    {
        "provider": "magpie",
        "display_name": "NVIDIA Magpie TTS (local CUDA)",
        "type": "local",
        "default_model": MAGPIE_TTS_MODEL,
        "models": [
            {
                "id": MAGPIE_TTS_MODEL,
                "display_name": "Magpie TTS Multilingual 357M",
                "artifact_id": MAGPIE_ARTIFACT_ID,
                "sample_rate": 22050,
                "supports_streaming": True,
                "streaming_mode": "native_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "John", "display_name": "John"}],
                "languages": [{"code": "en-US", "display_name": "English"}],
            }
        ],
    },
]


def get_supported_tts_providers() -> list[dict[str, Any]]:
    """Return frontend-consumable metadata for supported TTS providers."""
    from app.services.local_models import get_local_model_manager

    manager = get_local_model_manager()
    providers = copy.deepcopy(SUPPORTED_TTS_PROVIDERS)
    available: list[dict[str, Any]] = []
    for provider in providers:
        models = []
        for model in provider.get("models", []):
            artifact_id = model.get("artifact_id")
            if not artifact_id:
                models.append(model)
                continue
            status = manager.status(artifact_id)
            if status.get("supported") is False:
                continue
            model["artifact"] = {
                key: status.get(key)
                for key in (
                    "id",
                    "status",
                    "supported",
                    "runtime_ready",
                    "runtime_detail",
                    "license",
                    "license_url",
                )
            }
            models.append(model)
        provider["models"] = models
        if models:
            available.append(provider)
    return available
