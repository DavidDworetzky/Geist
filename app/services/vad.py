"""
Voice activity detection (VAD) providers for the voice pipeline.

Two implementations: a Silero VAD neural model (optional 'silero-vad'
package) and the original RMS energy threshold. The "auto" factory mode
prefers Silero and falls back to RMS when the package is not installed.
"""

import importlib.util
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

# Silero VAD consumes fixed-size windows: 512 samples at 16kHz, 256 at 8kHz.
SILERO_WINDOW_SIZES = {16000: 512, 8000: 256}


class VADProvider(ABC):
    """Abstract base class for voice activity detectors."""

    @abstractmethod
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Return True if the chunk (float32 mono, [-1, 1]) contains speech."""

    def reset(self) -> None:
        """Reset any streaming state between utterances/sessions."""
        return


class RMSVAD(VADProvider):
    """Energy-threshold VAD (the original behavior)."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return rms > self.threshold


class SileroVAD(VADProvider):
    """
    Neural VAD using the Silero VAD model (optional 'silero-vad' package).

    The model is stateful across calls; degrades permanently to an RMS
    fallback if inference fails at runtime so a broken VAD never takes
    down a live voice session.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        rms_fallback_threshold: float = 0.01,
    ):
        if sample_rate not in SILERO_WINDOW_SIZES:
            raise ValueError(
                f"Silero VAD supports sample rates {sorted(SILERO_WINDOW_SIZES)}, "
                f"got {sample_rate}"
            )
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size = SILERO_WINDOW_SIZES[sample_rate]
        self._model: Any = None
        self._fallback = RMSVAD(threshold=rms_fallback_threshold)
        self._failed = False

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from silero_vad import load_silero_vad
        except ImportError as e:
            raise RuntimeError(
                "The silero VAD provider requires the optional 'silero-vad' "
                "package (pip install silero-vad==6.2.1). Install it or "
                "select vad_provider=rms. "
                f"Import failed with: {e}"
            ) from e

        self._model = load_silero_vad()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        if self._failed:
            return self._fallback.is_speech(audio_chunk)

        try:
            import torch

            self._ensure_model()

            audio = np.asarray(audio_chunk, dtype=np.float32)
            if len(audio) < self.window_size:
                audio = np.pad(audio, (0, self.window_size - len(audio)))

            max_prob = 0.0
            for start in range(0, len(audio) - self.window_size + 1, self.window_size):
                window = torch.from_numpy(audio[start : start + self.window_size])
                prob = float(self._model(window, self.sample_rate).item())
                max_prob = max(max_prob, prob)

            return max_prob >= self.threshold
        except RuntimeError:
            # Missing package: surface the actionable error to the caller.
            raise
        except Exception as e:
            logger.error(f"Silero VAD inference failed, falling back to RMS: {e}")
            self._failed = True
            return self._fallback.is_speech(audio_chunk)

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception as e:
                logger.debug(f"Silero VAD reset failed: {e}")


def create_vad(
    provider: str = "auto",
    rms_threshold: float = 0.01,
    silero_threshold: float = 0.5,
    sample_rate: int = 16000,
) -> VADProvider:
    """
    Factory for VAD providers.

    Args:
        provider: "silero", "rms", or "auto" (silero when the package is
            installed, otherwise rms)
    """
    provider = provider.lower()

    if provider == "rms":
        return RMSVAD(threshold=rms_threshold)

    if provider == "silero":
        return SileroVAD(
            threshold=silero_threshold,
            sample_rate=sample_rate,
            rms_fallback_threshold=rms_threshold,
        )

    if provider == "auto":
        if importlib.util.find_spec("silero_vad") is not None:
            return SileroVAD(
                threshold=silero_threshold,
                sample_rate=sample_rate,
                rms_fallback_threshold=rms_threshold,
            )
        logger.info("silero-vad not installed; using RMS energy VAD")
        return RMSVAD(threshold=rms_threshold)

    raise ValueError(f"Unknown VAD provider: {provider}")
