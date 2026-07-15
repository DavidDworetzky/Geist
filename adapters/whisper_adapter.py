import tempfile

import httpx

from adapters.inert_adapter import InertAdapter


TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"


class WhisperAdapter(InertAdapter):
    '''
    Whisper Adapter is an adapter for Open AI's speech to text model.
    '''

    def __init__(self, api_key, **kwargs):
        self.api_key = api_key

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def transcribe(self, audio_data, language: str | None = None) -> str:
        # Whisper needs a named file, so stage the audio bytes in a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            temp_file.seek(0)
            response = httpx.post(
                TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (temp_file.name, temp_file, "application/octet-stream")},
                data={"model": "whisper-1", "response_format": "text", **({"language": language} if language else {})},
                timeout=60.0,
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Transcription failed with status {response.status_code}: {response.text}")
        return response.text
