"""
Unit tests for speech-to-text provider catalog and factory.
"""
import numpy as np
import pytest

from adapters.faster_whisper_adapter import (
    DEFAULT_FASTER_WHISPER_MODEL,
    FasterWhisperAdapter,
)
from adapters.parakeet_adapter import DEFAULT_PARAKEET_MODEL, ParakeetAdapter
from adapters.whisper_adapter import WhisperAdapter
from app.services.stt import create_stt_adapter, get_supported_stt_providers


def test_supported_stt_providers_include_new_local_providers():
    providers = get_supported_stt_providers()
    names = {provider["provider"] for provider in providers}

    assert {"mms", "whisper", "faster_whisper", "parakeet"} <= names

    faster_whisper = next(p for p in providers if p["provider"] == "faster_whisper")
    assert faster_whisper["type"] == "local"
    assert faster_whisper["default_model"] == DEFAULT_FASTER_WHISPER_MODEL
    assert any(m["id"] == "large-v3-turbo" for m in faster_whisper["models"])

    parakeet = next(p for p in providers if p["provider"] == "parakeet")
    assert parakeet["type"] == "local"
    assert parakeet["default_model"] == DEFAULT_PARAKEET_MODEL


def test_create_faster_whisper_adapter_is_lazy():
    adapter = create_stt_adapter("faster_whisper", stt_model="small")

    assert isinstance(adapter, FasterWhisperAdapter)
    assert adapter.model_id == "small"
    assert adapter._model is None


def test_create_parakeet_adapter_is_lazy():
    adapter = create_stt_adapter("parakeet")

    assert isinstance(adapter, ParakeetAdapter)
    assert adapter.model_id == DEFAULT_PARAKEET_MODEL
    assert adapter._model is None


def test_create_whisper_adapter_passes_api_key():
    adapter = create_stt_adapter("whisper", whisper_api_key="key123")

    assert isinstance(adapter, WhisperAdapter)
    assert adapter.api_key == "key123"


def test_create_stt_adapter_rejects_unlisted_faster_whisper_model():
    with pytest.raises(ValueError, match="not a supported faster_whisper STT model"):
        create_stt_adapter("faster_whisper", stt_model="../local/path")


def test_create_stt_adapter_rejects_unlisted_parakeet_model():
    with pytest.raises(ValueError, match="not a supported parakeet STT model"):
        create_stt_adapter("parakeet", stt_model="attacker/arbitrary-hf-repo")


def test_create_stt_adapter_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown STT provider"):
        create_stt_adapter("nonexistent")


def test_faster_whisper_missing_package_raises_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    adapter = FasterWhisperAdapter()
    with pytest.raises(RuntimeError, match="faster-whisper"):
        adapter.transcribe(np.zeros(1600, dtype=np.float32))


def test_parakeet_missing_package_raises_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("nemo"):
            raise ImportError("No module named 'nemo'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    adapter = ParakeetAdapter()
    with pytest.raises(RuntimeError, match="nemo-toolkit"):
        adapter.transcribe(np.zeros(1600, dtype=np.float32))


def test_faster_whisper_joins_segment_texts():
    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeWhisperModel:
        def transcribe(self, audio, language=None, beam_size=5):
            return iter([FakeSegment(" Hello"), FakeSegment("world ")]), None

    adapter = FasterWhisperAdapter()
    adapter._model = FakeWhisperModel()

    assert adapter.transcribe(np.zeros(1600, dtype=np.float32)) == "Hello world"


class _FakeHypothesis:
    def __init__(self, text):
        self.text = text


@pytest.mark.parametrize(
    "output",
    [
        ["hello world"],
        [_FakeHypothesis("hello world")],
        ([_FakeHypothesis("hello world")], ["extra"]),
    ],
)
def test_parakeet_extracts_text_from_nemo_output_shapes(output):
    class FakeNemoModel:
        def transcribe(self, audio_list):
            return output

    adapter = ParakeetAdapter()
    adapter._model = FakeNemoModel()

    assert adapter.transcribe(np.zeros(1600, dtype=np.float32)) == "hello world"
