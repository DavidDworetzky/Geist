import os
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    openai_key: str | None
    twilio_key: str | None
    twilio_sid: str | None
    twilio_source: str | None
    voice_stt_provider: str
    voice_tts_provider: str
    gemini_api_key: str | None


def load_environment() -> EnvironmentConfig:
    openai_key = os.getenv("OPENAI_API_KEY")
    twilio_key = os.getenv("TWILIO_TOKEN")
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_source = os.getenv("TWILIO_SOURCE")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Voice configuration
    voice_stt_provider = os.getenv("VOICE_STT_PROVIDER", "mms")  # Default to local MMS
    voice_tts_provider = os.getenv("VOICE_TTS_PROVIDER", "sesame")  # Default to local Sesame

    return EnvironmentConfig(
        openai_key=openai_key,
        twilio_key=twilio_key,
        twilio_sid=twilio_sid,
        twilio_source=twilio_source,
        voice_stt_provider=voice_stt_provider,
        voice_tts_provider=voice_tts_provider,
        gemini_api_key=gemini_api_key,
    )


def load_environment_dictionary() -> dict[str, str]:
    env_config = load_environment()
    entries = {
        "openai_key": env_config.openai_key,
        "twilio_key": env_config.twilio_key,
        "twilio_sid": env_config.twilio_sid,
        "twilio_source": env_config.twilio_source,
        "voice_stt_provider": env_config.voice_stt_provider,
        "voice_tts_provider": env_config.voice_tts_provider,
        "gemini_api_key": env_config.gemini_api_key,
    }
    return {key: value for key, value in entries.items() if value is not None}
