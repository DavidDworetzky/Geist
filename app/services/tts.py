"""
Text-to-Speech (TTS) service abstraction and implementations.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torchaudio


if TYPE_CHECKING:
    from agents.architectures.sesame.generator import Generator


logger = logging.getLogger(__name__)

DEFAULT_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_KOKORO_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_COSYVOICE2_MODEL = "FunAudioLLM/CosyVoice2-0.5B"

# Kokoro selects its G2P pipeline by single-letter language code.
KOKORO_LANG_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
}

SUPPORTED_TTS_PROVIDERS: list[dict[str, Any]] = [
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
    {
        "provider": "kokoro",
        "display_name": "Kokoro",
        "type": "local",
        "default_model": DEFAULT_KOKORO_MODEL,
        "models": [
            {
                "id": DEFAULT_KOKORO_MODEL,
                "display_name": "Kokoro 82M",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "segment_streaming",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [
                    {"id": "af_heart", "display_name": "Heart (US female)"},
                    {"id": "af_bella", "display_name": "Bella (US female)"},
                    {"id": "af_nicole", "display_name": "Nicole (US female)"},
                    {"id": "af_sky", "display_name": "Sky (US female)"},
                    {"id": "am_adam", "display_name": "Adam (US male)"},
                    {"id": "am_michael", "display_name": "Michael (US male)"},
                    {"id": "bf_emma", "display_name": "Emma (UK female)"},
                    {"id": "bf_isabella", "display_name": "Isabella (UK female)"},
                    {"id": "bm_george", "display_name": "George (UK male)"},
                    {"id": "bm_lewis", "display_name": "Lewis (UK male)"},
                ],
                "languages": [
                    {"code": "en", "display_name": "English (US)"},
                    {"code": "en-gb", "display_name": "English (UK)"},
                    {"code": "es", "display_name": "Spanish"},
                    {"code": "fr", "display_name": "French"},
                    {"code": "hi", "display_name": "Hindi"},
                    {"code": "it", "display_name": "Italian"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "pt", "display_name": "Portuguese"},
                    {"code": "zh", "display_name": "Chinese"},
                ],
            }
        ],
    },
    {
        "provider": "cosyvoice2",
        "display_name": "CosyVoice2",
        "type": "local",
        "default_model": DEFAULT_COSYVOICE2_MODEL,
        "models": [
            {
                "id": DEFAULT_COSYVOICE2_MODEL,
                "display_name": "CosyVoice2 0.5B",
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native",
                "supports_instruction_control": False,
                "supports_voice_cloning": True,
                "voices": [{"id": "default", "display_name": "Default"}],
                "languages": [
                    {"code": "en", "display_name": "English"},
                    {"code": "zh", "display_name": "Chinese"},
                    {"code": "ja", "display_name": "Japanese"},
                    {"code": "ko", "display_name": "Korean"},
                ],
            }
        ],
    },
]


def get_supported_tts_providers() -> list[dict[str, Any]]:
    """Return frontend-consumable metadata for supported TTS providers."""
    return SUPPORTED_TTS_PROVIDERS


def allowed_models_for_provider(provider_type: str) -> list[str]:
    """Published model ids for a provider, from SUPPORTED_TTS_PROVIDERS."""
    for entry in SUPPORTED_TTS_PROVIDERS:
        if entry["provider"] == provider_type:
            return [model["id"] for model in entry.get("models", [])]
    return []


def _validate_tts_model(provider_type: str, model: str) -> str:
    """
    Reject model ids that are not in the provider's published list.

    The model string is client-controlled and reaches model loaders
    (e.g. from_pretrained), so it must never be an arbitrary repo id or
    filesystem path.
    """
    allowed = allowed_models_for_provider(provider_type)
    if allowed and model not in allowed:
        raise ValueError(
            f"Model '{model}' is not a supported {provider_type} TTS model. "
            f"Supported models: {allowed}"
        )
    return model


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
    def synthesize_streaming(
        self, text: str, speaker: int = 0, chunk_size_ms: int = 100
    ) -> Iterator[bytes]:
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
        self._generator: Generator | None = None
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
        generator = self._generator
        if generator is None:
            raise RuntimeError("Sesame TTS generator failed to initialize")

        try:
            audio = generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=10000,
                temperature=0.9,
                topk=50,
            )
            return audio
        except Exception as e:
            self.logger.error(f"Sesame TTS synthesis failed: {e}")
            raise

    def synthesize_streaming(
        self, text: str, speaker: int = 0, chunk_size_ms: int = 100
    ) -> Iterator[bytes]:
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
            chunk = audio_int16[i : i + chunk_samples]
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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
        }

        response = self.client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"OpenAI TTS failed: {response.status_code} - {response.text}")

        # Load audio from response bytes
        audio_bytes = io.BytesIO(response.content)
        waveform, sample_rate = torchaudio.load(audio_bytes)

        # Convert to mono if stereo
        waveform = torch.mean(waveform, dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        return cast(torch.Tensor, waveform)

    def synthesize_streaming(
        self, text: str, speaker: int = 0, chunk_size_ms: int = 100
    ) -> Iterator[bytes]:
        """
        Synthesize and stream audio chunks.

        Note: OpenAI TTS doesn't support streaming, so we generate and chunk.
        """
        audio_tensor = self.synthesize(text, speaker)
        audio_np = audio_tensor.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        chunk_samples = int(self._sample_rate * chunk_size_ms / 1000)

        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
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
        instruct: str | None = None,
        speed: float = 1.0,
        device: str | None = None,
        sample_rate: int = 24000,
    ):
        self.model = model
        self.voice = voice
        self.language = language
        self.instruct = instruct
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sample_rate = sample_rate
        self._engine = None
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        """Lazy initialize Qwen3 TTS so normal app startup stays lightweight."""
        if self._engine is not None:
            return

        try:
            from qwen_tts import QwenTTS
        except ImportError as e:
            # Covers both a missing package and a mismatched API surface
            # (e.g. the class name changing) - this provider is experimental
            # and the qwen_tts integration has not been validated end-to-end.
            raise RuntimeError(
                "The qwen3 TTS provider requires the optional 'qwen_tts' package "
                "and is experimental. Install/verify qwen_tts or select another "
                f"tts_provider. Import failed with: {e}"
            ) from e

        self.logger.info(f"Initializing Qwen3 TTS via qwen_tts: {self.model}")
        if hasattr(QwenTTS, "from_pretrained"):
            self._engine = QwenTTS.from_pretrained(self.model, device=self.device)
        else:
            self._engine = QwenTTS(model=self.model, device=self.device)

    def _build_kwargs(self, text: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
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

    def _generate_with_qwen_tts(self, text: str) -> tuple[torch.Tensor, int]:
        kwargs = self._build_kwargs(text)
        engine = self._engine

        for method_name in ("synthesize", "generate", "infer"):
            method = getattr(engine, method_name, None)
            if method is None or not self._has_real_method(engine, method_name):
                continue
            result = method(**kwargs)
            return self._coerce_audio_result(result)

        raise RuntimeError("Loaded qwen_tts engine does not expose synthesize, generate, or infer.")

    def _coerce_audio_result(self, result: Any) -> tuple[torch.Tensor, int]:
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

    def _stream_native(self, text: str) -> Iterator[bytes] | None:
        self._ensure_initialized()
        engine = self._engine
        if engine is None:
            return None

        for method_name in ("synthesize_streaming", "generate_stream", "stream"):
            method = getattr(engine, method_name, None)
            if method is None or not self._has_real_method(engine, method_name):
                continue

            def _iter_chunks(stream_method=method):
                for chunk in stream_method(**self._build_kwargs(text)):
                    if isinstance(chunk, bytes):
                        yield chunk
                        continue
                    audio_tensor, _ = self._coerce_audio_result(chunk)
                    yield self._audio_tensor_to_pcm(audio_tensor)

            return _iter_chunks()

        return None

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        self._ensure_initialized()

        audio, _ = self._generate_with_qwen_tts(text)
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
            chunk = audio_int16[i : i + chunk_samples]
            yield chunk.tobytes()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class KokoroTTSProvider(TTSProvider):
    """
    TTS provider using the Kokoro-82M model (optional 'kokoro' package).

    Lightweight enough for CPU; streams audio per text segment as the
    pipeline produces it.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        language: str = "en",
        speed: float = 1.0,
        device: str | None = None,
    ):
        self.voice = voice
        self.language = language
        self.speed = speed
        self.device = device
        self._pipeline = None
        self._sample_rate = 24000
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        if self._pipeline is not None:
            return

        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "The kokoro TTS provider requires the optional 'kokoro' "
                "package (pip install kokoro==0.9.4). Install it or select "
                "another tts_provider. "
                f"Import failed with: {e}"
            ) from e

        lang_code = KOKORO_LANG_CODES.get(self.language.lower(), "a")
        self.logger.info(f"Initializing Kokoro TTS (lang_code={lang_code})")
        self._pipeline = KPipeline(
            lang_code=lang_code, repo_id=DEFAULT_KOKORO_MODEL, device=self.device
        )

    @staticmethod
    def _segment_audio(item: Any) -> torch.Tensor:
        audio = getattr(item, "audio", None)
        if audio is None and isinstance(item, tuple | list) and len(item) >= 3:
            audio = item[2]
        if audio is None:
            raise RuntimeError("Kokoro pipeline returned a segment with no audio")
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return cast(torch.Tensor, audio.detach().cpu().float().squeeze())

    def _segments(self, text: str) -> Iterator[torch.Tensor]:
        self._ensure_initialized()
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Kokoro TTS pipeline failed to initialize")

        for item in pipeline(text, voice=self.voice, speed=self.speed):
            yield self._segment_audio(item)

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        segments = list(self._segments(text))
        if not segments:
            return torch.zeros(0)
        return torch.cat(segments)

    def synthesize_streaming(
        self, text: str, speaker: int = 0, chunk_size_ms: int = 100
    ) -> Iterator[bytes]:
        chunk_samples = int(self._sample_rate * chunk_size_ms / 1000)

        for segment in self._segments(text):
            audio_np = np.clip(segment.numpy(), -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            for i in range(0, len(audio_int16), chunk_samples):
                yield audio_int16[i : i + chunk_samples].tobytes()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class CosyVoice2TTSProvider(TTSProvider):
    """
    TTS provider using CosyVoice2 with native low-latency streaming.

    Experimental: requires the 'cosyvoice' package from the FunAudioLLM
    CosyVoice repository, which is intentionally not a declared dependency.
    CosyVoice2-0.5B is a zero-shot model, so a reference voice is required:
    set COSYVOICE2_PROMPT_AUDIO (path to a short wav of the target voice)
    and COSYVOICE2_PROMPT_TEXT (its transcript) on the server. Falls back
    to pretrained speaker ids when the loaded model ships any.
    """

    def __init__(
        self,
        model: str = DEFAULT_COSYVOICE2_MODEL,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ):
        self.model = model
        self.voice = voice
        self.speed = speed
        self._sample_rate = sample_rate
        self._engine = None
        self._prompt_speech: torch.Tensor | None = None
        self._prompt_text: str | None = None
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        if self._engine is not None:
            return

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError as e:
            raise RuntimeError(
                "The cosyvoice2 TTS provider requires the optional 'cosyvoice' "
                "package from the FunAudioLLM/CosyVoice repository and is "
                "experimental. Install it or select another tts_provider. "
                f"Import failed with: {e}"
            ) from e

        from huggingface_hub import snapshot_download

        self.logger.info(f"Initializing CosyVoice2 TTS: {self.model}")
        model_dir = snapshot_download(self.model)
        self._engine = CosyVoice2(model_dir)
        self._sample_rate = int(getattr(self._engine, "sample_rate", self._sample_rate))
        self._load_prompt_voice()

    def _load_prompt_voice(self):
        """Load the operator-configured reference voice for zero-shot synthesis."""
        import os

        prompt_audio = os.getenv("COSYVOICE2_PROMPT_AUDIO")
        prompt_text = os.getenv("COSYVOICE2_PROMPT_TEXT")
        if not prompt_audio or not prompt_text:
            return

        waveform, source_rate = torchaudio.load(prompt_audio)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if source_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, source_rate, 16000)
        self._prompt_speech = waveform
        self._prompt_text = prompt_text

    def _stream_outputs(self, text: str, stream: bool) -> Iterator[torch.Tensor]:
        self._ensure_initialized()
        engine = self._engine
        if engine is None:
            raise RuntimeError("CosyVoice2 engine failed to initialize")

        if self._prompt_speech is not None and self._prompt_text is not None:
            outputs = engine.inference_zero_shot(
                text, self._prompt_text, self._prompt_speech, stream=stream, speed=self.speed
            )
        else:
            available = []
            if hasattr(engine, "list_available_spks"):
                available = list(engine.list_available_spks() or [])
            if not available:
                raise RuntimeError(
                    "CosyVoice2-0.5B has no pretrained speakers; configure "
                    "COSYVOICE2_PROMPT_AUDIO and COSYVOICE2_PROMPT_TEXT with a "
                    "reference voice sample for zero-shot synthesis."
                )
            spk_id = self.voice if self.voice in available else available[0]
            outputs = engine.inference_sft(text, spk_id, stream=stream, speed=self.speed)

        for output in outputs:
            speech = output["tts_speech"] if isinstance(output, dict) else output
            yield speech.detach().cpu().float().squeeze()

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        segments = list(self._stream_outputs(text, stream=False))
        if not segments:
            return torch.zeros(0)
        return torch.cat(segments)

    def synthesize_streaming(
        self, text: str, speaker: int = 0, chunk_size_ms: int = 100
    ) -> Iterator[bytes]:
        for segment in self._stream_outputs(text, stream=True):
            audio_np = np.clip(segment.numpy(), -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            yield audio_int16.tobytes()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


def create_tts_provider(provider_type: str = "sesame", **kwargs) -> TTSProvider:
    """
    Factory function to create TTS provider.

    Args:
        provider_type: Type of provider ("sesame", "openai", "qwen3",
            "kokoro", or "cosyvoice2")
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
        model = _validate_tts_model("openai", kwargs.get("model", "gpt-4o-mini-tts"))
        voice = kwargs.get("voice", "alloy")
        return OpenAITTSProvider(api_key=api_key, model=model, voice=voice)
    elif provider_type.lower() in {"qwen", "qwen3"}:
        return Qwen3TTSProvider(
            model=_validate_tts_model("qwen3", kwargs.get("model", DEFAULT_QWEN3_TTS_MODEL)),
            voice=kwargs.get("voice", "Cherry"),
            language=kwargs.get("language", "en"),
            instruct=kwargs.get("instruct"),
            speed=float(kwargs.get("speed", 1.0)),
            device=kwargs.get("device"),
            sample_rate=int(kwargs.get("sample_rate", 24000)),
        )
    elif provider_type.lower() == "kokoro":
        _validate_tts_model("kokoro", kwargs.get("model", DEFAULT_KOKORO_MODEL))
        return KokoroTTSProvider(
            voice=kwargs.get("voice", "af_heart"),
            language=kwargs.get("language", "en"),
            speed=float(kwargs.get("speed", 1.0)),
            device=kwargs.get("device"),
        )
    elif provider_type.lower() == "cosyvoice2":
        return CosyVoice2TTSProvider(
            model=_validate_tts_model(
                "cosyvoice2", kwargs.get("model", DEFAULT_COSYVOICE2_MODEL)
            ),
            voice=kwargs.get("voice", "default"),
            speed=float(kwargs.get("speed", 1.0)),
            sample_rate=int(kwargs.get("sample_rate", 24000)),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider_type}")
