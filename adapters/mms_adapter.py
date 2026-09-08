import threading

from adapters.inert_adapter import InertAdapter


_MODEL_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None


class MMSAdapter(InertAdapter):
    """
    MMSAdapter is an adapter for Meta's MMS Speech to Text API.
    """

    def __init__(self, **kwargs):
        self.model_id = "facebook/mms-1b-all"

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def transcribe(self, audio_data, language: str = "en") -> str:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        global _MODEL, _PROCESSOR
        if language != "en":
            raise ValueError("The local voice pipeline currently supports English STT only")
        # MMS is expensive and mutable (language adapters). Share one English
        # instance, serialize inference, and never download on a recording path.
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
                    # The base checkpoint already contains English weights.
                    # Passing target_lang invokes an additional adapter loader
                    # that does not inherit from_pretrained's offline flag.
                    model = Wav2Vec2ForCTC.from_pretrained(self.model_id, local_files_only=True)
                except OSError as error:
                    raise RuntimeError(
                        "MMS speech recognition is not cached. Download it first with "
                        "uv run python scripts/download_models.py --voice_model mms-1b-all"
                    ) from error
                processor.tokenizer.set_target_lang("eng")
                model.eval()
                _PROCESSOR, _MODEL = processor, model
            inputs = _PROCESSOR(audio_data, sampling_rate=16_000, return_tensors="pt")
            with torch.inference_mode():
                outputs = _MODEL(**inputs).logits
            ids = torch.argmax(outputs, dim=-1)[0]
            return str(_PROCESSOR.decode(ids))
