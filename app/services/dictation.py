import io
import os
import wave

from fastapi import HTTPException


MAX_DICTATION_BYTES = 16000 * 2 * 120 + 44


def transcribe_dictation(audio: bytes, provider: str) -> str:
    try:
        with wave.open(io.BytesIO(audio), "rb") as recording:
            if (
                recording.getnchannels() != 1
                or recording.getsampwidth() != 2
                or recording.getframerate() != 16000
                or recording.getnframes() > 16000 * 120
            ):
                raise ValueError("Unsupported recording format")
            frames = recording.readframes(recording.getnframes())
            if not frames or len(frames) != recording.getnframes() * 2:
                raise ValueError("Empty recording")
    except (wave.Error, EOFError, ValueError) as error:
        raise HTTPException(
            400, "Dictation requires a mono 16 kHz WAV recording up to two minutes."
        ) from error

    if provider == "whisper":
        from adapters.whisper_adapter import WhisperAdapter

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(503, "Set OPENAI_API_KEY on the server to use Whisper dictation.")
        return WhisperAdapter(api_key=api_key).transcribe(audio)

    import numpy as np

    from adapters.mms_adapter import MMSAdapter

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return MMSAdapter().transcribe(samples, language="en")
