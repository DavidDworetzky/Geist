from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter

class MMSAdapter(BaseAdapter):
    '''
    MMSAdapter is an adapter for Meta's MMS Speech to Text API.
    '''

    def enumerate_actions(self):
        pass
    
    def transcribe(self, audio_data):
        pass
