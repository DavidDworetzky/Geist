"""Unit tests for text-to-speech provider wiring."""

from pathlib import Path
from types import SimpleNamespace

import pytest


np = pytest.importorskip("numpy")

from app.services.tts import (
    DEFAULT_KOKORO_TTS_MODEL,
    DEFAULT_QWEN3_TTS_MODEL,
    KOKORO_DEFAULT_VOICE,
    MAGPIE_TTS_MODEL,
    KokoroMLXTTSProvider,
    MagpieTTSProvider,
    Qwen3MLXTTSProvider,
    create_tts_provider,
    get_supported_tts_providers,
)


class FakeModelManager:
    def status(self, artifact_id):
        return {
            "id": artifact_id,
            "status": "not_installed",
            "supported": True,
            "runtime_ready": False,
            "runtime_detail": "Runtime missing",
            "license": "test",
            "license_url": "https://example.com/license",
        }


def test_supported_tts_providers_include_curated_local_voice_models(monkeypatch):
    monkeypatch.setattr(
        "app.services.local_models.get_local_model_manager",
        lambda: FakeModelManager(),
    )

    providers = get_supported_tts_providers()
    kokoro = next(provider for provider in providers if provider["provider"] == "kokoro")
    qwen3 = next(provider for provider in providers if provider["provider"] == "qwen3")
    magpie = next(provider for provider in providers if provider["provider"] == "magpie")

    assert kokoro["default_model"] == DEFAULT_KOKORO_TTS_MODEL
    assert kokoro["models"][0]["voices"] == [{"id": KOKORO_DEFAULT_VOICE, "display_name": "Vera"}]
    assert kokoro["models"][0]["languages"] == [{"code": "a", "display_name": "English (US)"}]
    assert kokoro["models"][0]["streaming_mode"] == "sentence_buffered_pcm"

    assert qwen3["default_model"] == DEFAULT_QWEN3_TTS_MODEL
    assert qwen3["type"] == "local"
    assert len(qwen3["models"]) == 1
    assert qwen3["models"][0]["voices"] == [{"id": "Aiden", "display_name": "Aiden"}]
    assert qwen3["models"][0]["languages"] == [{"code": "English", "display_name": "English"}]
    assert qwen3["models"][0]["artifact"]["runtime_ready"] is False

    assert magpie["default_model"] == MAGPIE_TTS_MODEL
    assert len(magpie["models"]) == 1
    assert magpie["models"][0]["voices"] == [{"id": "John", "display_name": "John"}]


def test_create_tts_provider_builds_qwen_mlx_without_loading_model():
    provider = create_tts_provider("qwen3", voice="Aiden", language="English")

    assert isinstance(provider, Qwen3MLXTTSProvider)
    assert provider.model == DEFAULT_QWEN3_TTS_MODEL
    assert provider.voice == "Aiden"
    assert provider.language == "English"
    assert provider._engine is None


def test_create_tts_provider_builds_vera_kokoro_without_loading_model():
    provider = create_tts_provider("kokoro")

    assert isinstance(provider, KokoroMLXTTSProvider)
    assert provider.model == DEFAULT_KOKORO_TTS_MODEL
    assert provider.voice == KOKORO_DEFAULT_VOICE
    assert provider.language == "a"
    assert provider._engine is None


def test_kokoro_provider_streams_pcm_using_only_managed_voice_path():
    calls = []

    class FakeEngine:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        audio=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                        sample_rate=24_000,
                    )
                ]
            )

    provider = KokoroMLXTTSProvider(speed=1.1)
    provider._engine = FakeEngine()
    provider._voice_path = "/managed/voices/af_heart.safetensors"

    chunks = list(provider.synthesize_streaming("Hello David, this is Vera."))

    assert np.frombuffer(chunks[0], dtype=np.int16).tolist() == [-32767, 0, 32767]
    assert calls == [
        {
            "text": "Hello David, this is Vera.",
            "voice": "/managed/voices/af_heart.safetensors",
            "speed": 1.1,
            "lang_code": "a",
        }
    ]


def test_kokoro_provider_rejects_unmanaged_voice():
    with pytest.raises(ValueError, match="only supports the curated af_heart"):
        create_tts_provider("kokoro", voice="../other-voice")


def test_kokoro_provider_fails_before_misaki_can_download_missing_assets(monkeypatch):
    class FakeModelManager:
        def require_installed(self, artifact_id):
            assert artifact_id == "kokoro-82m-bf16-mlx"
            return SimpleNamespace(model_id=DEFAULT_KOKORO_TTS_MODEL), Path("/managed/kokoro")

    monkeypatch.setattr(
        "app.services.local_models.get_local_model_manager",
        lambda: FakeModelManager(),
    )
    monkeypatch.setattr(
        "app.services.tts.importlib.util.find_spec",
        lambda module: None if module == "en_core_web_sm" else object(),
    )
    provider = KokoroMLXTTSProvider()

    with pytest.raises(RuntimeError, match="runtime downloads are disabled"):
        provider._ensure_initialized()


def test_qwen_mlx_provider_streams_pcm_from_custom_voice_api():
    calls = []

    class FakeEngine:
        def generate_custom_voice(self, **kwargs):
            calls.append(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        audio=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                        sample_rate=24_000,
                    )
                ]
            )

    provider = Qwen3MLXTTSProvider(streaming_interval=0.32)
    provider._engine = FakeEngine()

    chunks = list(provider.synthesize_streaming("hello"))

    assert len(chunks) == 1
    assert np.frombuffer(chunks[0], dtype=np.int16).tolist() == [-32767, 0, 32767]
    assert calls == [
        {
            "text": "hello",
            "speaker": "Aiden",
            "language": "English",
            "stream": True,
            "streaming_interval": 0.32,
        }
    ]


def test_create_tts_provider_builds_magpie_without_starting_worker():
    provider = create_tts_provider("magpie")

    assert isinstance(provider, MagpieTTSProvider)
    assert provider.model == MAGPIE_TTS_MODEL
    assert provider.voice == "John"
    assert provider.language == "en-US"
    assert provider._process is None


def test_create_tts_provider_rejects_unlisted_local_model():
    with pytest.raises(ValueError, match="not a supported qwen3 TTS model"):
        create_tts_provider("qwen3", model="attacker/arbitrary-hf-repo")


def test_create_tts_provider_rejects_unlisted_openai_model():
    with pytest.raises(ValueError, match="not a supported openai TTS model"):
        create_tts_provider("openai", api_key="k", model="../local/path")
