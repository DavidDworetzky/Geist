from typing import Any

import numpy as np

from adapters.inert_adapter import InertAdapter


DEFAULT_FASTER_WHISPER_MODEL = "large-v3-turbo"


class FasterWhisperAdapter(InertAdapter):
    '''
    FasterWhisperAdapter runs OpenAI Whisper models locally via the
    faster-whisper (CTranslate2) runtime. Requires the optional
    'faster-whisper' package.
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
        self._model: Any = None

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "The faster_whisper STT provider requires the optional "
                "'faster-whisper' package (pip install faster-whisper==1.2.1). "
                "Install it or select another stt_provider. "
                f"Import failed with: {e}"
            ) from e

        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = WhisperModel(
            self.model_id, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, audio_data, language: str = "en") -> str:
        self._ensure_model()

        audio = np.asarray(audio_data, dtype=np.float32)
        segments, _info = self._model.transcribe(
            audio, language=language or None, beam_size=5
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
