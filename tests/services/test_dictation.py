import io
import wave
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.services.dictation import transcribe_dictation


def recording(rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * 160)
    return output.getvalue()


def test_dictation_rejects_invalid_audio_before_loading_models():
    for data in [b"not audio", recording(24000)]:
        with pytest.raises(HTTPException) as error:
            transcribe_dictation(data, "mms")
        assert error.value.status_code == 400


def test_whisper_receives_wav_bytes_without_an_agent(monkeypatch):
    from adapters.whisper_adapter import WhisperAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transcribe = Mock(return_value="Dictated message")
    monkeypatch.setattr(WhisperAdapter, "transcribe", transcribe)
    audio = recording()
    assert transcribe_dictation(audio, "whisper") == "Dictated message"
    transcribe.assert_called_once_with(audio)
