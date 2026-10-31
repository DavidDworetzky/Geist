"""Voice model metadata without loading inference runtimes."""

from typing import Any


DEFAULT_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_KOKORO_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_COSYVOICE2_MODEL = "FunAudioLLM/CosyVoice2-0.5B"

# Kokoro selects its G2P pipeline by single-letter language code.
KOKORO_LANG_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
}

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
        "provider": "qwen3",
        "display_name": "Qwen3 TTS",
        "type": "local",
        "default_model": DEFAULT_QWEN3_TTS_MODEL,
        "models": [
            {
                "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "display_name": "Qwen3 TTS 0.6B Custom Voice",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_or_chunked",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "Cherry", "display_name": "Cherry"},
                    {"id": "Chelsie", "display_name": "Chelsie"},
                    {"id": "Ethan", "display_name": "Ethan"},
                    {"id": "Serena", "display_name": "Serena"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                    {"code": "de", "display_name": "German"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "ru", "display_name": "Russian"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "it", "display_name": "Italian"},
                ],
            },
            {
                "id": DEFAULT_QWEN3_TTS_MODEL,
                "display_name": "Qwen3 TTS 1.7B Custom Voice",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_or_chunked",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "Cherry", "display_name": "Cherry"},
                    {"id": "Chelsie", "display_name": "Chelsie"},
                    {"id": "Ethan", "display_name": "Ethan"},
                    {"id": "Serena", "display_name": "Serena"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                    {"code": "de", "display_name": "German"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "ru", "display_name": "Russian"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "it", "display_name": "Italian"},
                ],
            },
        ],
    },
    {
        "provider": "kokoro",
        "display_name": "Kokoro",
        "type": "local",
        "default_model": DEFAULT_KOKORO_MODEL,
        "models": [
            {
                "id": DEFAULT_KOKORO_MODEL,
                "display_name": "Kokoro 82M",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "segment_streaming",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "af_heart", "display_name": "Heart (US female)"},
                    {"id": "af_bella", "display_name": "Bella (US female)"},
                    {"id": "af_nicole", "display_name": "Nicole (US female)"},
                    {"id": "af_sky", "display_name": "Sky (US female)"},
                    {"id": "am_adam", "display_name": "Adam (US male)"},
                    {"id": "am_michael", "display_name": "Michael (US male)"},
                    {"id": "bf_emma", "display_name": "Emma (UK female)"},
                    {"id": "bf_isabella", "display_name": "Isabella (UK female)"},
                    {"id": "bm_george", "display_name": "George (UK male)"},
                    {"id": "bm_lewis", "display_name": "Lewis (UK male)"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English (US)"},
                    {"code": "en-gb", "display_name": "English (UK)"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "hi", "display_name": "Hindi"},
                    {"code": "it", "display_name": "Italian"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "zh", "display_name": "Chinese"},
                ],
            }
        ],
    },
    {
        "provider": "cosyvoice2",
        "display_name": "CosyVoice2",
        "type": "local",
        "default_model": DEFAULT_COSYVOICE2_MODEL,
        "models": [
            {
                "id": DEFAULT_COSYVOICE2_MODEL,
                "display_name": "CosyVoice2 0.5B",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native",
                "supports_instruction_control": False,
                "supports_voice_cloning": True,
                "voices": [{"id": "default", "display_name": "Default"}],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                ],
            }
        ],
    },
]


def get_supported_tts_providers() -> list[dict[str, Any]]:
    """Return frontend-consumable metadata for supported TTS providers."""
    return SUPPORTED_TTS_PROVIDERS
