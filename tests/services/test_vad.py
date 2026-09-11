"""
Unit tests for voice activity detection providers.
"""
import numpy as np
import pytest

from app.services.vad import RMSVAD, SileroVAD, create_vad


def test_rms_vad_thresholding():
    vad = RMSVAD(threshold=0.01)

    assert not vad.is_speech(np.zeros(1600, dtype=np.float32))
    assert vad.is_speech(np.ones(1600, dtype=np.float32) * 0.05)


def test_create_vad_rms():
    vad = create_vad("rms", rms_threshold=0.02)

    assert isinstance(vad, RMSVAD)
    assert vad.threshold == 0.02


def test_create_vad_auto_falls_back_to_rms_when_silero_missing(monkeypatch):
    import importlib.util

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "silero_vad" else object()
    )

    vad = create_vad("auto")
    assert isinstance(vad, RMSVAD)


def test_create_vad_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown VAD provider"):
        create_vad("nonexistent")


def test_silero_vad_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError, match="sample rates"):
        SileroVAD(sample_rate=44100)


def test_silero_missing_package_raises_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "silero_vad":
            raise ImportError("No module named 'silero_vad'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    vad = SileroVAD()
    with pytest.raises(RuntimeError, match="silero-vad"):
        vad.is_speech(np.zeros(1600, dtype=np.float32))


class FakeSileroModel:
    """Callable stand-in returning a fixed probability per 512-sample window."""

    def __init__(self, prob):
        self.prob = prob
        self.calls = 0
        self.reset_calls = 0

    def __call__(self, window, sample_rate):
        import torch

        assert window.shape[-1] == 512
        assert sample_rate == 16000
        self.calls += 1
        return torch.tensor(self.prob)

    def reset_states(self):
        self.reset_calls += 1


def test_silero_vad_detects_speech_above_threshold():
    vad = SileroVAD(threshold=0.5)
    vad._model = FakeSileroModel(prob=0.9)

    # 1600 samples -> three full 512-sample windows
    assert vad.is_speech(np.ones(1600, dtype=np.float32) * 0.1)
    assert vad._model.calls == 3


def test_silero_vad_rejects_silence_below_threshold():
    vad = SileroVAD(threshold=0.5)
    vad._model = FakeSileroModel(prob=0.1)

    assert not vad.is_speech(np.zeros(1600, dtype=np.float32))


def test_silero_vad_pads_short_chunks():
    vad = SileroVAD(threshold=0.5)
    vad._model = FakeSileroModel(prob=0.9)

    assert vad.is_speech(np.ones(100, dtype=np.float32) * 0.1)
    assert vad._model.calls == 1


def test_silero_vad_degrades_to_rms_on_inference_failure():
    class BrokenModel:
        def __call__(self, window, sample_rate):
            raise ValueError("boom")

    vad = SileroVAD(threshold=0.5, rms_fallback_threshold=0.01)
    vad._model = BrokenModel()

    # First call fails and flips to RMS fallback; loud audio is still speech.
    assert vad.is_speech(np.ones(1600, dtype=np.float32) * 0.05)
    assert vad._failed
    assert not vad.is_speech(np.zeros(1600, dtype=np.float32))


def test_silero_vad_reset_propagates_to_model():
    vad = SileroVAD()
    vad._model = FakeSileroModel(prob=0.5)

    vad.reset()
    assert vad._model.reset_calls == 1
