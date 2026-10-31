from typing import Any

import numpy as np

from adapters.inert_adapter import InertAdapter
from adapters.shared_model_cache import SharedModel, get_shared_model


DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetAdapter(InertAdapter):
    '''
    ParakeetAdapter runs NVIDIA Parakeet ASR models locally via the NeMo
    toolkit. Requires the optional 'nemo-toolkit[asr]' package, which is
    intentionally not a declared dependency (heavy, conflicting dependency
    tree); the adapter fails loudly at session setup if it is missing.
    '''

    def __init__(self, model: str = DEFAULT_PARAKEET_MODEL, **kwargs):
        self.model_id = model
        self._shared: SharedModel | None = None

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def _load_model(self) -> Any:
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
        return model

    def load(self) -> None:
        """Load (or reuse) the shared model so failures surface at session setup."""
        if self._shared is None:
            self._shared = get_shared_model(("parakeet", self.model_id), self._load_model)

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
        self.load()
        assert self._shared is not None

        audio = np.asarray(audio_data, dtype=np.float32)
        with self._shared.lock:
            outputs = self._shared.model.transcribe([audio])
        return self._extract_text(outputs)
