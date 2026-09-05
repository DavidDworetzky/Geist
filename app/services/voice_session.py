"""
Voice session service for handling real-time audio streaming, STT, LLM, and TTS.
"""

import asyncio
import functools
import logging
import re
from collections import deque
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from app.services.tts import TTSProvider, create_tts_provider


logger = logging.getLogger(__name__)
SEGMENT_BOUNDARY = re.compile(r"[.!?](?:[\"')\]]+)?\s+")
MAX_UNPUNCTUATED_SEGMENT = 180


def create_stt_adapter(provider_type: str, **kwargs) -> Any:
    """Create an STT adapter without importing optional audio runtimes at module load."""

    if provider_type.lower() == "mms":
        from adapters.mms_adapter import MMSAdapter

        return MMSAdapter()
    if provider_type.lower() == "whisper":
        from adapters.whisper_adapter import WhisperAdapter

        return WhisperAdapter(api_key=kwargs.get("whisper_api_key"))
    raise ValueError(f"Unknown STT provider: {provider_type}")


def _ready_tts_segments(text: str, *, final: bool = False) -> tuple[list[str], str]:
    """Split stable response text without speaking an unfinished phrase."""

    segments: list[str] = []
    remaining = text
    while remaining:
        boundary = SEGMENT_BOUNDARY.search(remaining)
        if boundary:
            segment = remaining[: boundary.end()].strip()
            remaining = remaining[boundary.end() :]
            if segment:
                segments.append(segment)
            continue
        if len(remaining) >= MAX_UNPUNCTUATED_SEGMENT:
            split_at = remaining.rfind(" ", 0, MAX_UNPUNCTUATED_SEGMENT)
            if split_at <= 0:
                split_at = MAX_UNPUNCTUATED_SEGMENT
            segment = remaining[:split_at].strip()
            remaining = remaining[split_at:].lstrip()
            if segment:
                segments.append(segment)
            continue
        break
    if final and remaining.strip():
        segments.append(remaining.strip())
        remaining = ""
    return segments, remaining


class VoiceSessionService:
    """
    Service for managing voice streaming sessions.

    Handles:
    - Audio buffering and segmentation
    - Speech-to-text transcription
    - LLM completion (streaming or non-streaming)
    - Text-to-speech synthesis
    """

    def __init__(
        self,
        agent: BaseAgent,
        stt_provider: str = "mms",
        tts_provider: str = "sesame",
        sample_rate: int = 16000,
        vad_threshold: float = 0.01,
        silence_duration_ms: int = 800,
        chunk_duration_ms: int = 100,
        **provider_kwargs,
    ):
        """
        Initialize voice session service.

        Args:
            agent: Agent to use for text completion
            stt_provider: STT provider ("mms" or "whisper")
            tts_provider: TTS provider ("sesame", "openai", "kokoro", "qwen3", or "magpie")
            sample_rate: Audio sample rate in Hz
            vad_threshold: Voice activity detection threshold (RMS)
            silence_duration_ms: Silence duration to trigger phrase boundary (ms)
            chunk_duration_ms: Audio chunk duration for processing (ms)
            **provider_kwargs: Additional provider-specific arguments
        """
        self.agent = agent
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.chunk_duration_ms = chunk_duration_ms

        # Initialize STT
        self.stt = create_stt_adapter(stt_provider, **provider_kwargs)

        # Initialize TTS
        self.tts: TTSProvider = create_tts_provider(tts_provider, **provider_kwargs)

        # Audio buffer
        self.audio_buffer: deque[np.ndarray] = deque()
        self.transcript_buffer = ""
        self.silence_frames = 0
        self.silence_threshold_frames = int(silence_duration_ms / chunk_duration_ms)
        self.silence_samples = 0
        self.buffered_samples = 0
        self.has_speech = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-session")
        self._closed = False

        self.logger = logging.getLogger(__name__)

    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS (root mean square) of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk**2)))

    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple VAD using RMS threshold."""
        rms = self._calculate_rms(audio_chunk)
        return rms > self.vad_threshold

    def add_audio_chunk(self, audio_chunk: bytes) -> str | None:
        """
        Buffer PCM without running inference on the capture/control thread.

        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit signed integer, mono)

        Returns:
            None. The endpoint schedules partial transcription on the worker.
        """
        if not audio_chunk or len(audio_chunk) % 2:
            raise ValueError("Audio must contain complete PCM16 samples")
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if self._detect_speech(audio_np):
            self.has_speech = True
            self.silence_samples = 0
            self.silence_frames = 0
        else:
            self.silence_samples += len(audio_np)
            self.silence_frames += 1
        if self.has_speech:
            self.audio_buffer.append(audio_np)
            self.buffered_samples += len(audio_np)
        return None

    def check_phrase_boundary(self) -> bool:
        """
        Check if we've reached a phrase boundary (silence detected).

        Returns:
            bool: True if phrase boundary detected
        """
        return self.has_speech and (
            self.silence_samples * 1000 >= self.sample_rate * self.silence_duration_ms
            or self.buffered_samples >= self.sample_rate * 30
        )

    def take_audio(self) -> np.ndarray:
        audio = (
            np.concatenate(list(self.audio_buffer))
            if self.audio_buffer
            else np.array([], dtype=np.float32)
        )
        self.reset()
        return audio

    async def transcribe(self, audio: np.ndarray) -> str:
        if not audio.size:
            return ""
        return str(await self._run(self.stt.transcribe, audio, language="en"))

    async def _run(self, function, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Voice session is closed")
        return await asyncio.get_running_loop().run_in_executor(
            self._worker, functools.partial(function, *args, **kwargs)
        )

    def get_final_transcript(self) -> str:
        """
        Get final transcript from buffered audio and clear buffer.

        Returns:
            str: Final transcript
        """
        if not self.audio_buffer:
            return ""

        try:
            # Combine all buffered audio
            combined_audio = self.take_audio()

            # Run final STT
            transcript = self.stt.transcribe(combined_audio, language="en")

            # Clear buffer
            self.audio_buffer.clear()
            self.silence_frames = 0

            return str(transcript)
        except Exception as e:
            self.logger.error(f"Final STT failed: {e}")
            self.audio_buffer.clear()
            self.silence_frames = 0
            return ""

    async def process_with_agent(
        self,
        transcript: str,
        chat_id: int | None = None,
        system_prompt: str | None = None,
        use_streaming: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Advance on one worker thread: bounded to a single event, with no GPU
        # work on the event loop. Closing is queued behind any in-flight step.
        iterator = self._process_with_agent(transcript, chat_id, system_prompt, use_streaming)
        sentinel = object()
        try:
            while True:
                event = await self._run(next, iterator, sentinel)
                if event is sentinel:
                    return
                yield event
        finally:
            self._worker.submit(iterator.close)

    def _process_with_agent(
        self,
        transcript: str,
        chat_id: int | None = None,
        system_prompt: str | None = None,
        use_streaming: bool = True,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Process transcript with agent and yield responses.

        Args:
            transcript: User transcript to process
            chat_id: Chat session ID
            system_prompt: Optional system prompt
            use_streaming: Whether to use streaming LLM

        Yields:
            Dict with 'type' and data:
            - {'type': 'text_start'}
            - {'type': 'text_chunk', 'text': str}
            - {'type': 'text_complete', 'text': str}
            - {'type': 'audio_chunk', 'audio': bytes}
            - {'type': 'audio_complete'}
        """
        try:
            # Try streaming if supported and requested
            if use_streaming:
                try:
                    yield {"type": "text_start"}

                    full_text = ""
                    pending_tts_text = ""
                    audio_started = False
                    for chunk in self.agent.stream_complete_text(
                        prompt=transcript, chat_id=chat_id, system_prompt=system_prompt
                    ):
                        full_text += chunk
                        pending_tts_text += chunk
                        yield {"type": "text_chunk", "text": chunk}

                        segments, pending_tts_text = _ready_tts_segments(pending_tts_text)
                        for segment in segments:
                            if not audio_started:
                                yield {
                                    "type": "audio_start",
                                    "encoding": "pcm_s16le",
                                    "sample_rate": self.tts.sample_rate,
                                    "channels": 1,
                                }
                                audio_started = True
                            for audio_chunk in self.tts.synthesize_streaming(segment):
                                yield {"type": "audio_chunk", "audio": audio_chunk}

                    yield {"type": "text_complete", "text": full_text}

                    tail_segments, _ = _ready_tts_segments(pending_tts_text, final=True)
                    for segment in tail_segments:
                        if not audio_started:
                            yield {
                                "type": "audio_start",
                                "encoding": "pcm_s16le",
                                "sample_rate": self.tts.sample_rate,
                                "channels": 1,
                            }
                            audio_started = True
                        for audio_chunk in self.tts.synthesize_streaming(segment):
                            yield {"type": "audio_chunk", "audio": audio_chunk}

                    yield {"type": "audio_complete"}

                except NotImplementedError:
                    # Fall back to non-streaming
                    self.logger.info("Streaming not supported, falling back to non-streaming")
                    use_streaming = False

            if not use_streaming:
                # Non-streaming path
                completion = self.agent.complete_text(
                    prompt=transcript, chat_id=chat_id, system_prompt=system_prompt
                )

                # Extract text from completion
                if hasattr(completion, "choices") and completion.choices:
                    response_text = completion.choices[0].message.content
                elif hasattr(completion, "messages"):
                    response_text = next(
                        (msg.content for msg in completion.messages if msg.role == "assistant"), ""
                    )
                else:
                    response_text = str(completion)

                yield {"type": "text_start"}
                yield {"type": "text_chunk", "text": response_text}
                yield {"type": "text_complete", "text": response_text}

                # Generate TTS
                yield {
                    "type": "audio_start",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.tts.sample_rate,
                    "channels": 1,
                }
                for audio_chunk in self.tts.synthesize_streaming(response_text):
                    yield {"type": "audio_chunk", "audio": audio_chunk}

                yield {"type": "audio_complete"}

        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}")
            yield {"type": "error", "message": str(e)}

    def reset(self):
        """Reset the session state."""
        self.audio_buffer.clear()
        self.transcript_buffer = ""
        self.silence_frames = 0
        self.silence_samples = 0
        self.buffered_samples = 0
        self.has_speech = False
        self.logger.info("Voice session reset")

    def close(self) -> None:
        """Release provider resources when the WebSocket session ends."""
        if self._closed:
            return
        self._closed = True
        self.reset()
        close = getattr(self.tts, "close", None)
        if callable(close):
            self._worker.submit(close)
        self._worker.shutdown(wait=False)
