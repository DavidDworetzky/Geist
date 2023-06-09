from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List

class WhisperAdapter(BaseAdapter):
    '''
    Whisper Adapter is an adapter for Open AI's speech to text model. 
    '''

    def __init__(self, api_key):
        self.api_key = api_key

    def enumerate_actions(self) -> List[str]:
        return ["transcribe"]
    
    def transcribe(self, audio_data):
        pass
