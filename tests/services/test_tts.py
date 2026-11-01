"""
Unit tests for text-to-speech provider wiring.
"""
import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from adapters.shared_model_cache import SharedModel, clear_shared_models
from app.services.tts import (
    DEFAULT_COSYVOICE2_MODEL,
    DEFAULT_KOKORO_TTS_MODEL,
    DEFAULT_QWEN3_TTS_MODEL,
    KOKORO_PYTORCH_PROVIDER,
    CosyVoice2TTSProvider,
    KokoroPyTorchTTSProvider,
    Qwen3TTSProvider,
    create_tts_provider,
    get_supported_tts_providers,
)


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    clear_shared_models()
    yield
    clear_shared_models()


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


def test_supported_tts_providers_include_kokoro_pytorch_and_cosyvoice2_metadata():
    providers = get_supported_tts_providers()
    names = {provider["provider"] for provider in providers}

    # The bare "kokoro" id is reserved for the managed MLX artifact.
    assert "kokoro" not in names
    kokoro = next(p for p in providers if p["provider"] == KOKORO_PYTORCH_PROVIDER)
    assert kokoro["type"] == "local"
    assert kokoro["default_model"] == DEFAULT_KOKORO_TTS_MODEL
    kokoro_model = kokoro["models"][0]
    assert kokoro_model["supports_streaming"] is True
    assert any(voice["id"] == "af_heart" for voice in kokoro_model["voices"])

    cosyvoice2 = next(
        provider for provider in providers if provider["provider"] == "cosyvoice2"
    )
    assert cosyvoice2["type"] == "local"
    assert cosyvoice2["default_model"] == DEFAULT_COSYVOICE2_MODEL
    assert cosyvoice2["models"][0]["streaming_mode"] == "native"


def test_create_tts_provider_builds_kokoro_provider_without_loading_model():
    provider = create_tts_provider(
        KOKORO_PYTORCH_PROVIDER, voice="am_adam", language="en-gb", speed=1.2, device="cpu"
    )

    assert isinstance(provider, KokoroPyTorchTTSProvider)
    assert provider.voice == "am_adam"
    assert provider.language == "en-gb"
    assert provider.speed == 1.2
    assert provider._shared is None
    assert provider.sample_rate == 24000


@pytest.mark.parametrize(
    "voice",
    [
        "/etc/x.pt",
        "../voices/af_heart.pt",
        "af_heart.pt",
        "af_heart,af_bella",
        "hexgrad/Kokoro-82M/voices/af_heart",
        "unknown_voice",
    ],
)
def test_kokoro_rejects_voices_outside_allowlist(voice):
    with pytest.raises(ValueError, match="not a supported Kokoro voice"):
        create_tts_provider(KOKORO_PYTORCH_PROVIDER, voice=voice)


def test_kokoro_rejects_unsupported_language_and_speed():
    with pytest.raises(ValueError, match="not supported by Kokoro"):
        create_tts_provider(KOKORO_PYTORCH_PROVIDER, language="xx")
    with pytest.raises(ValueError, match="speed"):
        create_tts_provider(KOKORO_PYTORCH_PROVIDER, speed=10.0)


def test_create_tts_provider_rejects_unlisted_kokoro_model():
    with pytest.raises(ValueError, match="not a supported kokoro_pytorch TTS model"):
        create_tts_provider(KOKORO_PYTORCH_PROVIDER, model="attacker/arbitrary-hf-repo")


def test_kokoro_missing_package_raises_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "kokoro":
            raise ImportError("No module named 'kokoro'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    provider = KokoroPyTorchTTSProvider()
    with pytest.raises(RuntimeError, match="kokoro"):
        provider._ensure_initialized()


class FakeKokoroResult:
    def __init__(self, audio):
        self.audio = audio


class FakeKokoroPipeline:
    def __init__(self):
        self.voices = []

    def __call__(self, text, voice, speed):
        self.voices.append(voice)
        yield FakeKokoroResult(torch.ones(250) * 0.5)
        yield (None, None, torch.ones(100) * 0.25)


def test_kokoro_streams_pcm_per_segment():
    provider = KokoroPyTorchTTSProvider()
    provider._shared = SharedModel(model=FakeKokoroPipeline())
    provider._sample_rate = 1000

    chunks = list(provider.synthesize_streaming("hello", chunk_size_ms=100))

    # First segment (250 samples) -> 100+100+50; second (100) -> one chunk
    assert [len(chunk) for chunk in chunks] == [200, 200, 100, 200]
    assert provider._shared.model.voices == ["af_heart"]


def test_kokoro_synthesize_concatenates_segments():
    provider = KokoroPyTorchTTSProvider()
    provider._shared = SharedModel(model=FakeKokoroPipeline())

    audio = provider.synthesize("hello")
    assert audio.shape[0] == 350


def test_kokoro_pipeline_is_shared_across_sessions(monkeypatch):
    loads = []

    def fake_load(self, lang_code):
        loads.append(lang_code)
        return FakeKokoroPipeline()

    monkeypatch.setattr(KokoroPyTorchTTSProvider, "_load_pipeline", fake_load)
    first = KokoroPyTorchTTSProvider(voice="af_heart")
    second = KokoroPyTorchTTSProvider(voice="am_adam")
    british = KokoroPyTorchTTSProvider(voice="bf_emma", language="en-gb")
    for provider in (first, second, british):
        provider.synthesize("hello")

    assert first._shared is second._shared
    assert loads == ["a", "b"]


def test_create_tts_provider_rejects_unlisted_cosyvoice2_model():
    import pytest

    with pytest.raises(ValueError, match="not a supported cosyvoice2 TTS model"):
        create_tts_provider("cosyvoice2", model="attacker/arbitrary-hf-repo")


def test_cosyvoice2_streams_native_chunks_with_sft_speaker():
    class FakeCosyVoiceEngine:
        sample_rate = 24000

        def list_available_spks(self):
            return ["default"]

        def inference_sft(self, text, spk_id, stream, speed):
            assert stream is True
            yield {"tts_speech": torch.ones(1, 240) * 0.5}
            yield {"tts_speech": torch.ones(1, 120) * 0.25}

    provider = CosyVoice2TTSProvider()
    provider._shared = SharedModel(model=FakeCosyVoiceEngine())

    chunks = list(provider.synthesize_streaming("hello"))

    assert [len(chunk) for chunk in chunks] == [480, 240]


def test_cosyvoice2_uses_zero_shot_when_prompt_voice_configured():
    class FakeCosyVoiceEngine:
        sample_rate = 24000

        def inference_zero_shot(self, text, prompt_text, prompt_speech, stream, speed):
            assert prompt_text == "reference transcript"
            yield {"tts_speech": torch.ones(1, 100) * 0.5}

    provider = CosyVoice2TTSProvider()
    provider._shared = SharedModel(model=FakeCosyVoiceEngine())
    provider._prompt_speech = torch.zeros(1, 16000)
    provider._prompt_text = "reference transcript"

    audio = provider.synthesize("hello")
    assert audio.shape[0] == 100


def test_cosyvoice2_without_speakers_or_prompt_raises_actionable_error():
    import pytest

    class FakeCosyVoiceEngine:
        def list_available_spks(self):
            return []

    provider = CosyVoice2TTSProvider()
    provider._shared = SharedModel(model=FakeCosyVoiceEngine())

    with pytest.raises(RuntimeError, match="COSYVOICE2_PROMPT_AUDIO"):
        provider.synthesize("hello")
