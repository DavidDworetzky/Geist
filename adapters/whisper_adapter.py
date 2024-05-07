from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List
import requests
import json
import tempfile
import os

class WhisperAdapter(BaseAdapter):
    '''
    Whisper Adapter is an adapter for Open AI's speech to text model. 
    '''

    def __init__(self, api_key):
        self.api_key = api_key

    def enumerate_actions(self) -> List[str]:
        return ["transcribe"]
    
    def transcribe(self, audio_data):
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        # Write the audio data to the temporary file
        temp_file.write(audio_data)
        temp_file.close()

        # Update the audio_file variable to the path of the temporary file
        audio_file = temp_file.name
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        
        return transcript




