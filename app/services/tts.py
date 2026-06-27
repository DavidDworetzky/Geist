"""
Text-to-Speech (TTS) service abstraction and implementations.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

SUPPORTED_TTS_PROVIDERS: List[Dict[str, Any]] = [
    {
        "provider": "sesame",
        "display_name": "Sesame CSM",
        "type": "local",
        "default_model": "sesame/csm-1b",
        "models": [
            {
                "id": "sesame/csm-1b",
                "display_name": "Sesame CSM 1B",
                "sample_rate": 24000,
                "supports_streaming": False,
                "streaming_mode": "chunked_full_audio",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "0", "display_name": "Default Speaker"}],
                "languages": [{"code": "en", "display_name": "English"}],
            }
        ],
    },
    {
        "provider": "openai",
        "display_name": "OpenAI TTS",
        "type": "api",
        "default_model": "gpt-4o-mini-tts",
        "models": [
            {
                "id": "gpt-4o-mini-tts",
                "display_name": "GPT-4o mini TTS",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "provider_dependent",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "alloy", "display_name": "Alloy"},
                    {"id": "echo", "display_name": "Echo"},
                    {"id": "fable", "display_name": "Fable"},
                    {"id": "onyx", "display_name": "Onyx"},
                    {"id": "nova", "display_name": "Nova"},
                    {"id": "shimmer", "display_name": "Shimmer"},
                ],
                "languages": [{"code": "auto", "display_name": "Auto"}],
            },
            {
                "id": "tts-1",
                "display_name": "TTS 1",
                "sample_rate": 24000,
                "supports_streaming": False,
                "streaming_mode": "chunked_full_audio",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "alloy", "display_name": "Alloy"},
                    {"id": "echo", "display_name": "Echo"},
                    {"id": "fable", "display_name": "Fable"},
                    {"id": "onyx", "display_name": "Onyx"},
                    {"id": "nova", "display_name": "Nova"},
                    {"id": "shimmer", "display_name": "Shimmer"},
                ],
                "languages": [{"code": "auto", "display_name": "Auto"}],
            },
        ],
    },
    {
        "provider": "qwen3",
        "display_name": "Qwen3 TTS",
        "type": "local",
        "default_model": DEFAULT_QWEN3_TTS_MODEL,
        "models": [
            {
                "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "display_name": "Qwen3 TTS 0.6B Custom Voice",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_or_chunked",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "Cherry", "display_name": "Cherry"},
                    {"id": "Chelsie", "display_name": "Chelsie"},
                    {"id": "Ethan", "display_name": "Ethan"},
                    {"id": "Serena", "display_name": "Serena"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                    {"code": "de", "display_name": "German"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "ru", "display_name": "Russian"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "it", "display_name": "Italian"},
                ],
            },
            {
                "id": DEFAULT_QWEN3_TTS_MODEL,
                "display_name": "Qwen3 TTS 1.7B Custom Voice",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_or_chunked",
                "supports_instruction_control": True,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "Cherry", "display_name": "Cherry"},
                    {"id": "Chelsie", "display_name": "Chelsie"},
                    {"id": "Ethan", "display_name": "Ethan"},
                    {"id": "Serena", "display_name": "Serena"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                    {"code": "de", "display_name": "German"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "ru", "display_name": "Russian"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "it", "display_name": "Italian"},
                ],
            },
        ],
    },
]


def get_supported_tts_providers() -> List[Dict[str, Any]]:
    """Return frontend-consumable metadata for supported TTS providers."""
    return SUPPORTED_TTS_PROVIDERS


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
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            api_key: OpenAI API key
            model: TTS model to use
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


class Qwen3TTSProvider(TTSProvider):
    """TTS provider using Qwen3 TTS models."""

    def __init__(
        self,
        model: str = DEFAULT_QWEN3_TTS_MODEL,
        voice: str = "Cherry",
        language: str = "en",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        device: Optional[str] = None,
        sample_rate: int = 24000,
    ):
        self.model = model
        self.voice = voice
        self.language = language
        self.instruct = instruct
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sample_rate = sample_rate
        self._backend = None
        self._engine = None
        self._processor = None
        self._model_instance = None
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        """Lazy initialize Qwen3 TTS so normal app startup stays lightweight."""
        if self._engine is not None or self._model_instance is not None:
            return

        try:
            from qwen_tts import QwenTTS

            self.logger.info(f"Initializing Qwen3 TTS via qwen_tts: {self.model}")
            if hasattr(QwenTTS, "from_pretrained"):
                self._engine = QwenTTS.from_pretrained(self.model, device=self.device)
            else:
                self._engine = QwenTTS(model=self.model, device=self.device)
            self._backend = "qwen_tts"
            return
        except ImportError:
            self.logger.info("qwen_tts package not found; falling back to Transformers")

        try:
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForTextToWaveform as AutoTTSModel
            except ImportError:
                from transformers import AutoModel as AutoTTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 TTS requires either the qwen_tts package or Transformers."
            ) from exc

        self.logger.info(f"Initializing Qwen3 TTS via Transformers: {self.model}")
        self._processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        self._model_instance = AutoTTSModel.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self._model_instance = self._model_instance.to(self.device)
        self._backend = "transformers"

    def _build_kwargs(self, text: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "text": text,
            "voice": self.voice,
            "language": self.language,
            "speed": self.speed,
        }
        if self.instruct:
            kwargs["instruct"] = self.instruct
        return kwargs

    @staticmethod
    def _has_real_method(obj: Any, method_name: str) -> bool:
        if method_name in getattr(obj, "__dict__", {}):
            return True
        return any(method_name in getattr(cls, "__dict__", {}) for cls in type(obj).__mro__)

    def _generate_with_qwen_tts(self, text: str) -> Tuple[torch.Tensor, int]:
        kwargs = self._build_kwargs(text)
        engine = self._engine

        for method_name in ("synthesize", "generate", "infer"):
            method = getattr(engine, method_name, None)
            if method is None or not self._has_real_method(engine, method_name):
                continue
            result = method(**kwargs)
            return self._coerce_audio_result(result)

        raise RuntimeError("Loaded qwen_tts engine does not expose synthesize, generate, or infer.")

    def _generate_with_transformers(self, text: str) -> Tuple[torch.Tensor, int]:
        processor_kwargs = self._build_kwargs(text)
        processor_kwargs["return_tensors"] = "pt"

        try:
            inputs = self._processor(**processor_kwargs)
        except TypeError:
            processor_kwargs.pop("speed", None)
            inputs = self._processor(**processor_kwargs)

        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            result = self._model_instance.generate(**inputs)
        return self._coerce_audio_result(result)

    def _coerce_audio_result(self, result: Any) -> Tuple[torch.Tensor, int]:
        sample_rate = self._sample_rate
        audio = result

        if isinstance(result, tuple):
            audio = result[0]
            if len(result) > 1 and isinstance(result[1], int):
                sample_rate = result[1]
        elif isinstance(result, dict):
            sample_rate = result.get("sample_rate") or result.get("sampling_rate") or sample_rate
            audio = None
            for key in ("audio", "wav", "waveform", "speech"):
                if key in result and result[key] is not None:
                    audio = result[key]
                    break
        elif hasattr(result, "waveform"):
            audio = result.waveform
            sample_rate = getattr(result, "sample_rate", sample_rate)
        elif hasattr(result, "sequences"):
            audio = result.sequences
            sample_rate = getattr(result, "sample_rate", sample_rate)

        if audio is None:
            raise RuntimeError("Qwen3 TTS returned no audio data.")

        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio)
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio.detach().cpu()
        else:
            audio_tensor = torch.tensor(audio)

        audio_tensor = audio_tensor.float().squeeze()
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        self._sample_rate = int(sample_rate)
        return audio_tensor, self._sample_rate

    @staticmethod
    def _audio_tensor_to_pcm(audio_tensor: torch.Tensor) -> bytes:
        audio_np = audio_tensor.cpu().numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _stream_native(self, text: str) -> Optional[Iterator[bytes]]:
        self._ensure_initialized()
        engine = self._engine or self._model_instance
        if engine is None:
            return None

        for method_name in ("synthesize_streaming", "generate_stream", "stream"):
            method = getattr(engine, method_name, None)
            if method is None or not self._has_real_method(engine, method_name):
                continue

            def _iter_chunks():
                for chunk in method(**self._build_kwargs(text)):
                    if isinstance(chunk, bytes):
                        yield chunk
                        continue
                    audio_tensor, _ = self._coerce_audio_result(chunk)
                    yield self._audio_tensor_to_pcm(audio_tensor)

            return _iter_chunks()

        return None

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        self._ensure_initialized()

        if self._backend == "qwen_tts":
            audio, _ = self._generate_with_qwen_tts(text)
        else:
            audio, _ = self._generate_with_transformers(text)
        return audio

    def synthesize_streaming(
        self,
        text: str,
        speaker: int = 0,
        chunk_size_ms: int = 100,
    ) -> Iterator[bytes]:
        native_stream = self._stream_native(text)
        if native_stream is not None:
            yield from native_stream
            return

        audio_tensor = self.synthesize(text, speaker)
        audio_np = np.clip(audio_tensor.cpu().numpy(), -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        chunk_samples = int(self._sample_rate * chunk_size_ms / 1000)

        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            yield chunk.tobytes()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


def create_tts_provider(provider_type: str = "sesame", **kwargs) -> TTSProvider:
    """
    Factory function to create TTS provider.
    
    Args:
        provider_type: Type of provider ("sesame", "openai", or "qwen3")
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
        model = kwargs.get("model", "gpt-4o-mini-tts")
        voice = kwargs.get("voice", "alloy")
        return OpenAITTSProvider(api_key=api_key, model=model, voice=voice)
    elif provider_type.lower() in {"qwen", "qwen3"}:
        return Qwen3TTSProvider(
            model=kwargs.get("model", DEFAULT_QWEN3_TTS_MODEL),
            voice=kwargs.get("voice", "Cherry"),
            language=kwargs.get("language", "en"),
            instruct=kwargs.get("instruct"),
            speed=float(kwargs.get("speed", 1.0)),
            device=kwargs.get("device"),
            sample_rate=int(kwargs.get("sample_rate", 24000)),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider_type}")
