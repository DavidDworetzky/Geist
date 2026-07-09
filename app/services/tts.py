"""
Text-to-Speech (TTS) service abstraction and implementations.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Iterator
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (if multi-speaker model)
            
        Returns:
            torch.Tensor: Audio tensor (1D, mono)
        """
        pass
    
    @abstractmethod
    def synthesize_streaming(self, text: str, speaker: int = 0, chunk_size_ms: int = 100) -> Iterator[bytes]:
        """
        Synthesize speech from text in streaming fashion.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            chunk_size_ms: Size of audio chunks in milliseconds
            
        Yields:
            bytes: Audio chunks as PCM bytes
        """
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the sample rate of the TTS model."""
        pass


class SesameTTSProvider(TTSProvider):
    """TTS provider using Sesame CSM model."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Sesame TTS provider.
        
        Args:
            device: Device to run model on ("cuda" or "cpu")
        """
        self.device = device
        self._generator = None
        self._sample_rate = 24000  # Sesame default
        self.logger = logging.getLogger(__name__)
    
    def _ensure_initialized(self):
        """Lazy initialization of the generator."""
        if self._generator is None:
            from agents.architectures.sesame.generator import load_csm_1b
            self.logger.info(f"Initializing Sesame CSM TTS on {self.device}")
            self._generator = load_csm_1b(device=self.device)
            self._sample_rate = self._generator.sample_rate
    
    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        """
        Synthesize speech from text using Sesame CSM.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (0 for default)
            
        Returns:
            torch.Tensor: Audio tensor (1D, mono, 24kHz)
        """
        self._ensure_initialized()
        
        try:
            audio = self._generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=10000,
                temperature=0.9,
                topk=50
            )
            return audio
        except Exception as e:
            self.logger.error(f"Sesame TTS synthesis failed: {e}")
            raise
    
    def synthesize_streaming(self, text: str, speaker: int = 0, chunk_size_ms: int = 100) -> Iterator[bytes]:
        """
        Synthesize speech and yield as audio chunks.
        
        Note: Sesame doesn't support true streaming generation, so we generate the full
        audio and chunk it for streaming playback.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            chunk_size_ms: Size of audio chunks in milliseconds
            
        Yields:
            bytes: PCM audio chunks (16-bit signed integers, mono)
        """
        # Generate full audio
        audio_tensor = self.synthesize(text, speaker)
        
        # Convert to numpy and then bytes
        audio_np = audio_tensor.cpu().numpy()
        
        # Convert float32 to int16 PCM
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Calculate chunk size in samples
        chunk_samples = int(self._sample_rate * chunk_size_ms / 1000)
        
        # Yield chunks
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            yield chunk.tobytes()
    
    @property
    def sample_rate(self) -> int:
        """Get the sample rate (24kHz for Sesame)."""
        return self._sample_rate


class OpenAITTSProvider(TTSProvider):
    """TTS provider using OpenAI's TTS API."""
    
    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "alloy"):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            api_key: OpenAI API key
            model: TTS model to use ("tts-1" or "tts-1-hd")
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self._sample_rate = 24000  # OpenAI TTS default
        self.logger = logging.getLogger(__name__)
        
        import httpx
        self.client = httpx.Client(timeout=30.0)
    
    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        """
        Synthesize speech using OpenAI TTS API.
        
        Args:
            text: Text to synthesize
            speaker: Ignored for OpenAI (use voice parameter instead)
            
        Returns:
            torch.Tensor: Audio tensor
        """
        import io
        
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav"
        }
        
        response = self.client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI TTS failed: {response.status_code} - {response.text}")
        
        # Load audio from response bytes
        audio_bytes = io.BytesIO(response.content)
        waveform, sample_rate = torchaudio.load(audio_bytes)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        return waveform
    
    def synthesize_streaming(self, text: str, speaker: int = 0, chunk_size_ms: int = 100) -> Iterator[bytes]:
        """
        Synthesize and stream audio chunks.
        
        Note: OpenAI TTS doesn't support streaming, so we generate and chunk.
        """
        audio_tensor = self.synthesize(text, speaker)
        audio_np = audio_tensor.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        chunk_samples = int(self._sample_rate * chunk_size_ms / 1000)
        
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            yield chunk.tobytes()
    
    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate


def create_tts_provider(provider_type: str = "sesame", **kwargs) -> TTSProvider:
    """
    Factory function to create TTS provider.
    
    Args:
        provider_type: Type of provider ("sesame" or "openai")
        **kwargs: Provider-specific arguments
        
    Returns:
        TTSProvider: Initialized TTS provider
    """
    if provider_type.lower() == "sesame":
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return SesameTTSProvider(device=device)
    elif provider_type.lower() == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key required for OpenAI TTS provider")
        model = kwargs.get("model", "tts-1")
        voice = kwargs.get("voice", "alloy")
        return OpenAITTSProvider(api_key=api_key, model=model, voice=voice)
    else:
        raise ValueError(f"Unknown TTS provider: {provider_type}")

