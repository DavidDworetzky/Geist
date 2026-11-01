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
from adapters.shared_model_cache import SharedModel, clear_shared_models
from adapters.whisper_adapter import WhisperAdapter
from app.services import stt as stt_service
from app.services.stt import create_stt_adapter, get_supported_stt_providers


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    clear_shared_models()
    yield
    clear_shared_models()


def _patch_find_spec(monkeypatch, overrides):
    real_find_spec = stt_service.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in overrides:
            return overrides[name]
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(stt_service.importlib.util, "find_spec", find_spec)


@pytest.fixture
def installed_runtimes(monkeypatch):
    """Pretend optional STT runtimes are installed and record model loads."""
    _patch_find_spec(monkeypatch, {"faster_whisper": object(), "nemo": object()})
    loads: list[str] = []

    def fake_load(adapter):
        loads.append(adapter.model_id)
        return object()

    monkeypatch.setattr(FasterWhisperAdapter, "_load_model", fake_load)
    monkeypatch.setattr(ParakeetAdapter, "_load_model", fake_load)
    return loads


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


def test_create_faster_whisper_adapter_loads_at_setup(installed_runtimes):
    adapter = create_stt_adapter("faster_whisper", stt_model="small", stt_device="cpu")

    assert isinstance(adapter, FasterWhisperAdapter)
    assert adapter.model_id == "small"
    assert installed_runtimes == ["small"]


def test_create_parakeet_adapter_loads_at_setup(installed_runtimes):
    adapter = create_stt_adapter("parakeet")

    assert isinstance(adapter, ParakeetAdapter)
    assert adapter.model_id == DEFAULT_PARAKEET_MODEL
    assert installed_runtimes == [DEFAULT_PARAKEET_MODEL]


def test_local_stt_models_are_shared_across_sessions(installed_runtimes):
    first = create_stt_adapter("faster_whisper", stt_model="small", stt_device="cpu")
    second = create_stt_adapter("faster_whisper", stt_model="small", stt_device="cpu")
    create_stt_adapter("faster_whisper", stt_model="base", stt_device="cpu")

    assert first._shared is second._shared
    assert installed_runtimes == ["small", "base"]


@pytest.mark.parametrize(
    ("provider", "package"),
    [("faster_whisper", "faster_whisper"), ("parakeet", "nemo")],
)
def test_create_stt_adapter_fails_at_setup_when_package_missing(monkeypatch, provider, package):
    _patch_find_spec(monkeypatch, {package: None})

    with pytest.raises(RuntimeError, match=f"{provider} STT provider requires"):
        create_stt_adapter(provider)


def test_voice_websocket_reports_missing_stt_package_instead_of_ready(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.endpoints import voice

    monkeypatch.setattr(voice, "get_default_agent_context", Mock())
    monkeypatch.setattr(voice, "get_agent_for_session", AsyncMock(return_value=Mock()))
    _patch_find_spec(monkeypatch, {"faster_whisper": None})

    app = FastAPI()
    app.include_router(voice.router)
    with TestClient(app) as client, client.websocket_connect(
        "/stream?session_id=1&stt_provider=faster_whisper&tts_provider=sesame"
    ) as socket:
        message = socket.receive_json()

    assert message["type"] == "error"
    assert "faster_whisper" in message["message"]


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
    adapter = FasterWhisperAdapter(device="cpu")
    with pytest.raises(RuntimeError, match="faster-whisper"):
        adapter.load()


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
        adapter.load()


def test_faster_whisper_joins_segment_texts():
    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeWhisperModel:
        def transcribe(self, audio, language=None, beam_size=5):
            return iter([FakeSegment(" Hello"), FakeSegment("world ")]), None

    adapter = FasterWhisperAdapter()
    adapter._shared = SharedModel(model=FakeWhisperModel())

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
    adapter._shared = SharedModel(model=FakeNemoModel())

    assert adapter.transcribe(np.zeros(1600, dtype=np.float32)) == "hello world"
