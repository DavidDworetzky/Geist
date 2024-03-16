from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

class MMSAdapter(BaseAdapter):
    '''
    MMSAdapter is an adapter for Meta's MMS Speech to Text API.
    '''
    def __init__(self):
        self.model_id = "facebook/mms-1b-all"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

    def enumerate_actions(self) -> List[str]:
        return ["transcribe"]
    
    def transcribe(self, audio_data, language: str = "en"):
        inputs = self.processor(audio_data, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)
        return transcription
