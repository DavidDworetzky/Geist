from typing import Any

import numpy as np

from adapters.inert_adapter import InertAdapter


DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetAdapter(InertAdapter):
    '''
    ParakeetAdapter runs NVIDIA Parakeet ASR models locally via the NeMo
    toolkit. Requires the optional 'nemo-toolkit[asr]' package, which is
    intentionally not a declared dependency (heavy dependency tree); the
    adapter fails loudly if it is missing.
    '''

    def __init__(self, model: str = DEFAULT_PARAKEET_MODEL, **kwargs):
        self.model_id = model
        self._model: Any = None

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise RuntimeError(
                "The parakeet STT provider requires the optional "
                "'nemo-toolkit[asr]' package (pip install 'nemo-toolkit[asr]'). "
                "Install it or select another stt_provider. "
                f"Import failed with: {e}"
            ) from e

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        model.eval()
        self._model = model

    @staticmethod
    def _extract_text(result) -> str:
        # NeMo returns a list per input; entries are strings, Hypothesis
        # objects with .text, or (best, all) tuples depending on version.
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            if not result:
                return ""
            result = result[0]
        text = getattr(result, "text", result)
        return str(text).strip()

    def transcribe(self, audio_data, language: str = "en") -> str:
        self._ensure_model()

        audio = np.asarray(audio_data, dtype=np.float32)
        outputs = self._model.transcribe([audio])
        return self._extract_text(outputs)
