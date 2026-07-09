"""
Voice session service for handling real-time audio streaming, STT, LLM, and TTS.
"""
import logging
import numpy as np
import torch
from typing import Optional, AsyncIterator, Dict, Any
from collections import deque
import asyncio
from adapters.mms_adapter import MMSAdapter
from adapters.whisper_adapter import WhisperAdapter
from app.services.tts import TTSProvider, create_tts_provider
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


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
        **provider_kwargs
    ):
        """
        Initialize voice session service.
        
        Args:
            agent: Agent to use for text completion
            stt_provider: STT provider ("mms" or "whisper")
            tts_provider: TTS provider ("sesame" or "openai")
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
        if stt_provider.lower() == "mms":
            self.stt = MMSAdapter()
        elif stt_provider.lower() == "whisper":
            api_key = provider_kwargs.get("whisper_api_key")
            self.stt = WhisperAdapter(api_key=api_key)
        else:
            raise ValueError(f"Unknown STT provider: {stt_provider}")
        
        # Initialize TTS
        self.tts: TTSProvider = create_tts_provider(tts_provider, **provider_kwargs)
        
        # Audio buffer
        self.audio_buffer = deque()
        self.transcript_buffer = ""
        self.silence_frames = 0
        self.silence_threshold_frames = int(silence_duration_ms / chunk_duration_ms)
        
        self.logger = logging.getLogger(__name__)
    
    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS (root mean square) of audio chunk."""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple VAD using RMS threshold."""
        rms = self._calculate_rms(audio_chunk)
        return rms > self.vad_threshold
    
    def add_audio_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """
        Add audio chunk to buffer and check for phrase boundaries.
        
        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit signed integer, mono)
            
        Returns:
            Optional[str]: Partial transcript if available, None otherwise
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.append(audio_np)
        
        # Check for speech
        if self._detect_speech(audio_np):
            self.silence_frames = 0
        else:
            self.silence_frames += 1
        
        # If we have enough audio, run STT for partial transcript
        if len(self.audio_buffer) >= 10:  # ~1 second at 100ms chunks
            try:
                combined_audio = np.concatenate(list(self.audio_buffer))
                # MMS expects 16kHz
                transcript = self.stt.transcribe(combined_audio, language="en")
                return transcript
            except Exception as e:
                self.logger.error(f"Partial STT failed: {e}")
                return None
        
        return None
    
    def check_phrase_boundary(self) -> bool:
        """
        Check if we've reached a phrase boundary (silence detected).
        
        Returns:
            bool: True if phrase boundary detected
        """
        return self.silence_frames >= self.silence_threshold_frames
    
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
            combined_audio = np.concatenate(list(self.audio_buffer))
            
            # Run final STT
            transcript = self.stt.transcribe(combined_audio, language="en")
            
            # Clear buffer
            self.audio_buffer.clear()
            self.silence_frames = 0
            
            return transcript
        except Exception as e:
            self.logger.error(f"Final STT failed: {e}")
            self.audio_buffer.clear()
            self.silence_frames = 0
            return ""
    
    async def process_with_agent(
        self,
        transcript: str,
        chat_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_streaming: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
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
                    for chunk in self.agent.stream_complete_text(
                        prompt=transcript,
                        chat_id=chat_id,
                        system_prompt=system_prompt
                    ):
                        full_text += chunk
                        yield {"type": "text_chunk", "text": chunk}
                    
                    yield {"type": "text_complete", "text": full_text}
                    
                    # Generate TTS from complete text
                    for audio_chunk in self.tts.synthesize_streaming(full_text):
                        yield {"type": "audio_chunk", "audio": audio_chunk}
                    
                    yield {"type": "audio_complete"}
                    
                except NotImplementedError:
                    # Fall back to non-streaming
                    self.logger.info("Streaming not supported, falling back to non-streaming")
                    use_streaming = False
            
            if not use_streaming:
                # Non-streaming path
                completion = self.agent.complete_text(
                    prompt=transcript,
                    chat_id=chat_id,
                    system_prompt=system_prompt
                )
                
                # Extract text from completion
                if hasattr(completion, 'choices') and completion.choices:
                    response_text = completion.choices[0].message.content
                elif hasattr(completion, 'messages'):
                    response_text = next(
                        (msg.content for msg in completion.messages if msg.role == 'assistant'),
                        ""
                    )
                else:
                    response_text = str(completion)
                
                yield {"type": "text_start"}
                yield {"type": "text_chunk", "text": response_text}
                yield {"type": "text_complete", "text": response_text}
                
                # Generate TTS
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
        self.logger.info("Voice session reset")

