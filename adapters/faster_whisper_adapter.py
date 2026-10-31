from typing import Any

import numpy as np

from adapters.inert_adapter import InertAdapter
from adapters.shared_model_cache import SharedModel, get_shared_model


DEFAULT_FASTER_WHISPER_MODEL = "large-v3-turbo"


class FasterWhisperAdapter(InertAdapter):
    '''
    FasterWhisperAdapter runs OpenAI Whisper models locally via the
    faster-whisper (CTranslate2) runtime. Requires the optional
    'faster-whisper' package (the `voice-sota` extra).
    '''

    def __init__(
        self,
        model: str = DEFAULT_FASTER_WHISPER_MODEL,
        device: str | None = None,
        compute_type: str = "auto",
        **kwargs,
    ):
        self.model_id = model
        self.device = device
        self.compute_type = compute_type
        self._shared: SharedModel | None = None

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def _load_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "The faster_whisper STT provider requires the optional "
                "'faster-whisper' package (install the voice-sota extra). "
                "Install it or select another stt_provider. "
                f"Import failed with: {e}"
            ) from e

        return WhisperModel(self.model_id, device=self.device, compute_type=self.compute_type)

    def load(self) -> None:
        """Load (or reuse) the shared model so failures surface at session setup."""
        if self._shared is not None:
            return

        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._shared = get_shared_model(
            ("faster_whisper", self.model_id, self.device, self.compute_type),
            self._load_model,
        )

    def transcribe(self, audio_data, language: str = "en") -> str:
        self.load()
        assert self._shared is not None

        audio = np.asarray(audio_data, dtype=np.float32)
        with self._shared.lock:
            segments, _info = self._shared.model.transcribe(
                audio, language=language or None, beam_size=5
            )
            # Segments are a lazy generator; decoding happens while iterating.
            return " ".join(segment.text.strip() for segment in segments).strip()
