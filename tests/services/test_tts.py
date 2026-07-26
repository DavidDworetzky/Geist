"""
Unit tests for text-to-speech provider wiring.
"""
import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from app.services.tts import (
    DEFAULT_QWEN3_TTS_MODEL,
    Qwen3TTSProvider,
    create_tts_provider,
    get_supported_tts_providers,
)


def test_supported_tts_providers_include_qwen3_metadata():
    providers = get_supported_tts_providers()
    qwen3 = next(provider for provider in providers if provider["provider"] == "qwen3")

    assert qwen3["default_model"] == DEFAULT_QWEN3_TTS_MODEL
    assert qwen3["type"] == "local"
    assert any(model["id"] == DEFAULT_QWEN3_TTS_MODEL for model in qwen3["models"])

    default_model = next(model for model in qwen3["models"] if model["id"] == DEFAULT_QWEN3_TTS_MODEL)
    assert default_model["supports_streaming"] is True
    assert default_model["supports_instruction_control"] is True
    assert {"code": "en", "display_name": "English"} in default_model["languages"]


def test_create_tts_provider_builds_qwen3_provider_without_loading_model():
    provider = create_tts_provider(
        "qwen3",
        model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        voice="Ethan",
        language="en",
        instruct="Speak calmly.",
        speed=0.9,
        device="cpu",
    )

    assert isinstance(provider, Qwen3TTSProvider)
    assert provider.model == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert provider.voice == "Ethan"
    assert provider.language == "en"
    assert provider.instruct == "Speak calmly."
    assert provider.speed == 0.9
    assert provider.device == "cpu"
    assert provider._engine is None


def test_qwen3_provider_chunks_full_audio_when_native_streaming_is_unavailable():
    class FakeQwenEngine:
        def generate(self, **kwargs):
            return torch.ones(250) * 0.5, 1000

    provider = Qwen3TTSProvider(device="cpu", sample_rate=1000)
    provider._engine = FakeQwenEngine()

    chunks = list(provider.synthesize_streaming("hello", chunk_size_ms=100))

    assert len(chunks) == 3
    assert all(isinstance(chunk, bytes) for chunk in chunks)
    assert [len(chunk) for chunk in chunks] == [200, 200, 100]


def test_qwen3_provider_uses_native_streaming_when_available():
    class FakeStreamingQwenEngine:
        def synthesize_streaming(self, **kwargs):
            return iter(
                [
                    np.ones(100, dtype=np.float32) * 0.25,
                    b"pcm",
                ]
            )

    provider = Qwen3TTSProvider(device="cpu", sample_rate=1000)
    provider._engine = FakeStreamingQwenEngine()

    chunks = list(provider.synthesize_streaming("hello", chunk_size_ms=100))

    assert len(chunks) == 2
    assert len(chunks[0]) == 200
    assert chunks[1] == b"pcm"


def test_create_tts_provider_rejects_unlisted_qwen3_model():
    import pytest

    with pytest.raises(ValueError, match="not a supported qwen3 TTS model"):
        create_tts_provider("qwen3", model="attacker/arbitrary-hf-repo")


def test_create_tts_provider_rejects_unlisted_openai_model():
    import pytest

    with pytest.raises(ValueError, match="not a supported openai TTS model"):
        create_tts_provider("openai", api_key="k", model="../local/path")


def test_qwen3_missing_package_raises_friendly_error(monkeypatch):
    import builtins

    import pytest

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "qwen_tts":
            raise ImportError("No module named 'qwen_tts'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    provider = Qwen3TTSProvider(model=DEFAULT_QWEN3_TTS_MODEL)
    with pytest.raises(RuntimeError, match="experimental"):
        provider._ensure_initialized()
