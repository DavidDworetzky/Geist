"""
Text-to-Speech (TTS) service abstraction and implementations.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np


if TYPE_CHECKING:
    import torch

    from agents.architectures.sesame.generator import Generator


logger = logging.getLogger(__name__)

DEFAULT_KOKORO_TTS_MODEL = "hexgrad/Kokoro-82M"
KOKORO_MLX_ARTIFACT_ID = "kokoro-82m-bf16-mlx"
KOKORO_DEFAULT_VOICE = "af_heart"
DEFAULT_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN3_MLX_ARTIFACT_ID = "qwen3-tts-0.6b-customvoice-mlx-6bit"
MAGPIE_TTS_MODEL = "nvidia/magpie_tts_multilingual_357m"
MAGPIE_ARTIFACT_ID = "magpie-tts-multilingual-357m-nemo-speech"

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
        "provider": "kokoro",
        "display_name": "Vera (Kokoro local MLX)",
        "type": "local",
        "default_model": DEFAULT_KOKORO_TTS_MODEL,
        "models": [
            {
                "id": DEFAULT_KOKORO_TTS_MODEL,
                "display_name": "Vera · Kokoro 82M BF16",
                "artifact_id": KOKORO_MLX_ARTIFACT_ID,
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "sentence_buffered_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": KOKORO_DEFAULT_VOICE, "display_name": "Vera"}],
                "languages": [{"code": "a", "display_name": "English (US)"}],
            },
        ],
    },
    {
        "provider": "qwen3",
        "display_name": "Qwen3 TTS (local MLX)",
        "type": "local",
        "default_model": DEFAULT_QWEN3_TTS_MODEL,
        "models": [
            {
                "id": DEFAULT_QWEN3_TTS_MODEL,
                "display_name": "Qwen3 TTS 0.6B Custom Voice",
                "artifact_id": QWEN3_MLX_ARTIFACT_ID,
                "sample_rate": 24000,
                "supports_streaming": True,
                "streaming_mode": "native_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "Aiden", "display_name": "Aiden"}],
                "languages": [{"code": "English", "display_name": "English"}],
            },
        ],
    },
    {
        "provider": "magpie",
        "display_name": "NVIDIA Magpie TTS (local CUDA)",
        "type": "local",
        "default_model": MAGPIE_TTS_MODEL,
        "models": [
            {
                "id": MAGPIE_TTS_MODEL,
                "display_name": "Magpie TTS Multilingual 357M",
                "artifact_id": MAGPIE_ARTIFACT_ID,
                "sample_rate": 22050,
                "supports_streaming": True,
                "streaming_mode": "native_pcm",
                "supports_instruction_control": False,
                "supports_voice_cloning": False,
                "voices": [{"id": "John", "display_name": "John"}],
                "languages": [{"code": "en-US", "display_name": "English"}],
            }
        ],
    },
]


def get_supported_tts_providers() -> list[dict[str, Any]]:
    """Return frontend-consumable metadata for supported TTS providers."""
    from app.services.local_models import get_local_model_manager

    manager = get_local_model_manager()
    providers = copy.deepcopy(SUPPORTED_TTS_PROVIDERS)
    available: list[dict[str, Any]] = []
    for provider in providers:
        models = []
        for model in provider.get("models", []):
            artifact_id = model.get("artifact_id")
            if not artifact_id:
                models.append(model)
                continue
            status = manager.status(artifact_id)
            if status.get("supported") is False:
                continue
            model["artifact"] = {
                key: status.get(key)
                for key in (
                    "id",
                    "status",
                    "supported",
                    "runtime_ready",
                    "runtime_detail",
                    "license",
                    "license_url",
                )
            }
            models.append(model)
        provider["models"] = models
        if models:
            available.append(provider)
    return available


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

        import torch
        import torchaudio

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


class KokoroMLXTTSProvider(TTSProvider):
    """Pinned Kokoro provider with no synthesis-time downloads."""

    def __init__(
        self,
        model: str = DEFAULT_KOKORO_TTS_MODEL,
        voice: str = KOKORO_DEFAULT_VOICE,
        language: str = "a",
        speed: float = 1.0,
    ) -> None:
        if voice != KOKORO_DEFAULT_VOICE:
            raise ValueError(f"Kokoro only supports the curated {KOKORO_DEFAULT_VOICE} voice")
        if language != "a":
            raise ValueError("Kokoro only supports US English in the initial voice slice")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro speed must be between 0.5 and 2.0")
        self.model = model
        self.voice = voice
        self.language = language
        self.speed = speed
        self._sample_rate = 24_000
        self._engine: Any | None = None
        self._voice_path: str | None = None
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        if self._engine is not None:
            return
        from app.services.local_models import get_local_model_manager

        artifact, path = get_local_model_manager().require_installed(KOKORO_MLX_ARTIFACT_ID)
        if artifact.model_id != self.model:
            raise RuntimeError("Installed Kokoro voice artifact does not match the selected model")
        if importlib.util.find_spec("en_core_web_sm") is None:
            raise RuntimeError(
                "Kokoro's locked English speech assets are missing. Reinstall Geist's voice "
                "extra; runtime downloads are disabled."
            )
        voice_path = path / "voices" / f"{KOKORO_DEFAULT_VOICE}.safetensors"
        if not voice_path.is_file():
            raise RuntimeError("The managed Kokoro artifact is missing the Vera voice weights")
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as error:
            raise RuntimeError(
                "Kokoro on Apple Silicon requires the optional MLX Audio runtime."
            ) from error
        # Kokoro's upstream config predates MLX Audio's `model_type` field.
        # Supply the architecture explicitly because Geist stores every model
        # under a generic `snapshot` directory.
        self._engine = load_model(path, model_type="kokoro")
        self._voice_path = str(voice_path)

    @staticmethod
    def _result_to_pcm(result: Any) -> tuple[bytes, int]:
        audio = getattr(result, "audio", None)
        if audio is None:
            raise RuntimeError("MLX Audio returned a result without audio")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return pcm, int(getattr(result, "sample_rate", 24_000))

    def synthesize_streaming(
        self,
        text: str,
        speaker: int = 0,
        chunk_size_ms: int = 100,
    ) -> Iterator[bytes]:
        del speaker, chunk_size_ms
        with self._lock:
            self._ensure_initialized()
            if self._engine is None or self._voice_path is None:
                raise RuntimeError("Kokoro MLX TTS failed to initialize")
            for result in self._engine.generate(
                text=text,
                voice=self._voice_path,
                speed=self.speed,
                lang_code=self.language,
            ):
                pcm, sample_rate = self._result_to_pcm(result)
                self._sample_rate = sample_rate
                if pcm:
                    yield pcm

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        import torch

        pcm = b"".join(self.synthesize_streaming(text, speaker))
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(samples)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def close(self) -> None:
        with self._lock:
            self._engine = None
            self._voice_path = None


class Qwen3MLXTTSProvider(TTSProvider):
    """Pinned Qwen3 TTS provider using an explicitly installed MLX snapshot."""

    def __init__(
        self,
        model: str = DEFAULT_QWEN3_TTS_MODEL,
        voice: str = "Aiden",
        language: str = "English",
        streaming_interval: float = 0.32,
    ) -> None:
        self.model = model
        self.voice = voice
        self.language = language
        self.streaming_interval = streaming_interval
        self._sample_rate = 24_000
        self._engine: Any | None = None
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        if self._engine is not None:
            return
        from app.services.local_models import get_local_model_manager

        artifact, path = get_local_model_manager().require_installed(QWEN3_MLX_ARTIFACT_ID)
        if artifact.model_id != self.model:
            raise RuntimeError("Installed Qwen voice artifact does not match the selected model")
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as error:
            raise RuntimeError(
                "Qwen3 TTS on Apple Silicon requires the optional MLX Audio runtime."
            ) from error
        self._engine = load_model(str(path))

    @staticmethod
    def _result_to_pcm(result: Any) -> tuple[bytes, int]:
        audio = getattr(result, "audio", None)
        if audio is None:
            raise RuntimeError("MLX Audio returned a result without audio")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return pcm, int(getattr(result, "sample_rate", 24_000))

    def synthesize_streaming(
        self,
        text: str,
        speaker: int = 0,
        chunk_size_ms: int = 100,
    ) -> Iterator[bytes]:
        del speaker, chunk_size_ms
        with self._lock:
            self._ensure_initialized()
            if self._engine is None:
                raise RuntimeError("Qwen MLX TTS failed to initialize")
            results = self._engine.generate_custom_voice(
                text=text,
                speaker=self.voice,
                language=self.language,
                stream=True,
                streaming_interval=self.streaming_interval,
            )
            for result in results:
                pcm, sample_rate = self._result_to_pcm(result)
                self._sample_rate = sample_rate
                if pcm:
                    yield pcm

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        import torch

        pcm = b"".join(self.synthesize_streaming(text, speaker))
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(samples)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class MagpieTTSProvider(TTSProvider):
    """Pinned Magpie provider backed by an isolated NeMo-Speech.cpp worker."""

    def __init__(
        self,
        model: str = MAGPIE_TTS_MODEL,
        voice: str = "John",
        language: str = "en-US",
    ) -> None:
        self.model = model
        self.voice = voice
        self.language = language
        self._sample_rate = 22_050
        self._process: Any | None = None

    def _ensure_initialized(self) -> None:
        if self._process is not None:
            return
        from app.services.local_models import (
            get_local_model_manager,
            resolve_nemo_speech_library,
        )
        from app.services.local_tts_process import LocalTTSProcess

        artifact, path = get_local_model_manager().require_installed(MAGPIE_ARTIFACT_ID)
        if artifact.model_id != self.model:
            raise RuntimeError("Installed Magpie artifact does not match the selected model")
        library = resolve_nemo_speech_library()
        if not library:
            raise RuntimeError(
                "NeMo-Speech.cpp's TTS shared library is required for NVIDIA Magpie."
            )
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "The initial Magpie runtime is supported from the native Geist source "
                "distribution; packaged sidecar wiring is not available yet."
            )

        command = (
            sys.executable,
            "-m",
            "app.services.magpie_tts_worker",
            "--library",
            library,
            "--magpie-model",
            str(path / "magpie" / "magpie_tts_multilingual_357m.v2602.f16.gguf"),
            "--codec-model",
            str(path / "codec" / "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"),
            "--tokenizer-dir",
            str(path / "magpie" / "tokenizer"),
            "--language",
            self.language,
            "--voice",
            self.voice,
        )
        self._process = LocalTTSProcess(command)

    def synthesize_streaming(
        self,
        text: str,
        speaker: int = 0,
        chunk_size_ms: int = 100,
    ) -> Iterator[bytes]:
        del speaker, chunk_size_ms
        self._ensure_initialized()
        if self._process is None:
            raise RuntimeError("Magpie TTS failed to initialize")
        yield from self._process.synthesize(
            {"text": text, "voice": self.voice, "language": self.language}
        )
        if self._process.sample_rate:
            self._sample_rate = int(self._process.sample_rate)

    def synthesize(self, text: str, speaker: int = 0) -> torch.Tensor:
        import torch

        pcm = b"".join(self.synthesize_streaming(text, speaker))
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(samples)

    def close(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


def create_tts_provider(provider_type: str = "sesame", **kwargs) -> TTSProvider:
    """
    Factory function to create TTS provider.

    Args:
        provider_type: Type of provider ("sesame", "openai", "kokoro", "qwen3", or "magpie")
        **kwargs: Provider-specific arguments

    Returns:
        TTSProvider: Initialized TTS provider
    """
    if provider_type.lower() == "sesame":
        import torch

        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return SesameTTSProvider(device=device)
    elif provider_type.lower() == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key required for OpenAI TTS provider")
        model = _validate_tts_model("openai", kwargs.get("model", "gpt-4o-mini-tts"))
        voice = kwargs.get("voice", "alloy")
        return OpenAITTSProvider(api_key=api_key, model=model, voice=voice)
    elif provider_type.lower() == "kokoro":
        return KokoroMLXTTSProvider(
            model=_validate_tts_model("kokoro", kwargs.get("model", DEFAULT_KOKORO_TTS_MODEL)),
            voice=kwargs.get("voice", KOKORO_DEFAULT_VOICE),
            language=kwargs.get("language", "a"),
            speed=float(kwargs.get("speed", 1.0)),
        )
    elif provider_type.lower() in {"qwen", "qwen3"}:
        return Qwen3MLXTTSProvider(
            model=_validate_tts_model("qwen3", kwargs.get("model", DEFAULT_QWEN3_TTS_MODEL)),
            voice=kwargs.get("voice", "Aiden"),
            language=kwargs.get("language", "English"),
        )
    elif provider_type.lower() == "magpie":
        return MagpieTTSProvider(
            model=_validate_tts_model("magpie", kwargs.get("model", MAGPIE_TTS_MODEL)),
            voice=kwargs.get("voice", "John"),
            language=kwargs.get("language", "en-US"),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider_type}")
