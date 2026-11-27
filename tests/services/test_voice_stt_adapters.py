import io
import wave
from unittest.mock import Mock, patch

import numpy as np
import pytest

from adapters.whisper_adapter import WhisperAdapter


def test_whisper_encodes_float_samples_as_pcm_wav():
    captured = {}

    def post(url, **kwargs):
        captured["wav"] = kwargs["files"]["file"][1].read()
        return Mock(status_code=200, text="hello")

    with patch("adapters.whisper_adapter.httpx.post", side_effect=post):
        assert WhisperAdapter("test-key").transcribe(np.zeros(1600, dtype=np.float32)) == "hello"
    with wave.open(io.BytesIO(captured["wav"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
        assert wav.getnframes() == 1600


def test_mms_is_lazy_shared_and_offline(monkeypatch):
    from adapters import mms_adapter

    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mms_adapter, "_MODEL", None)
    monkeypatch.setattr(mms_adapter, "_PROCESSOR", None)
    processor, model = Mock(), Mock()
    processor.return_value = {}
    processor.decode.return_value = "hello"
    model.return_value.logits = torch.zeros((1, 2, 3))
    with (
        patch.object(
            transformers.AutoProcessor, "from_pretrained", return_value=processor
        ) as load_processor,
        patch.object(
            transformers.Wav2Vec2ForCTC, "from_pretrained", return_value=model
        ) as load_model,
    ):
        first, second = mms_adapter.MMSAdapter(), mms_adapter.MMSAdapter()
        load_model.assert_not_called()
        assert first.transcribe(np.zeros(1600)) == "hello"
        assert second.transcribe(np.zeros(1600)) == "hello"
        load_model.assert_called_once_with("facebook/mms-1b-all", local_files_only=True)
        load_processor.assert_called_once_with("facebook/mms-1b-all", local_files_only=True)
        model.load_adapter.assert_not_called()
