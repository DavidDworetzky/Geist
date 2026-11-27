import io
import wave

import httpx
import numpy as np

from adapters.inert_adapter import InertAdapter


TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"


class WhisperAdapter(InertAdapter):
    """
    Whisper Adapter is an adapter for Open AI's speech to text model.
    """

    def __init__(self, api_key, **kwargs):
        self.api_key = api_key

    def enumerate_actions(self) -> list[str]:
        return ["transcribe"]

    def transcribe(self, audio_data, language: str | None = None) -> str:
        # The voice service supplies normalized, mono 16 kHz samples.
        # Preserve encoded files for callers that already provide WAV bytes.
        if isinstance(audio_data, bytes):
            encoded_audio = audio_data
        else:
            samples = np.asarray(audio_data, dtype=np.float32)
            if samples.ndim != 1:
                raise ValueError("Whisper input must be mono audio")
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.tobytes())
            encoded_audio = output.getvalue()
        with io.BytesIO(encoded_audio) as audio_file:
            response = httpx.post(
                TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("speech.wav", audio_file, "audio/wav")},
                data={
                    "model": "whisper-1",
                    "response_format": "text",
                    **({"language": language} if language else {}),
                },
                timeout=60.0,
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Transcription failed with status {response.status_code}: {response.text}"
            )
        return str(response.text)
