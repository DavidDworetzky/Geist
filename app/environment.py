from dataclasses import dataclass
import os
import dotenv
from typing import Dict

@dataclass
class EnvironmentConfig:
    openai_key: str
    twilio_key: str
    twilio_sid: str
    twilio_source: str
    voice_stt_provider: str
    voice_tts_provider: str

def LoadEnvironment() -> EnvironmentConfig:
    openai_key = os.getenv("OPENAI_TOKEN")
    twilio_key = os.getenv("TWILIO_TOKEN")
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_source = os.getenv("TWILIO_SOURCE")
    
    # Voice configuration
    voice_stt_provider = os.getenv("VOICE_STT_PROVIDER", "mms")  # Default to local MMS
    voice_tts_provider = os.getenv("VOICE_TTS_PROVIDER", "sesame")  # Default to local Sesame

    return EnvironmentConfig(
        openai_key = openai_key,
        twilio_key = twilio_key,
        twilio_sid = twilio_sid,
        twilio_source=twilio_source,
        voice_stt_provider=voice_stt_provider,
        voice_tts_provider=voice_tts_provider,
    )

def LoadEnvironmentDictionary() -> Dict[str, str]:
    env_config = LoadEnvironment()
    return {
        "openai_key": env_config.openai_key,
        "twilio_key": env_config.twilio_key,
        "twilio_sid": env_config.twilio_sid,
        "twilio_source": env_config.twilio_source,
        "voice_stt_provider": env_config.voice_stt_provider,
        "voice_tts_provider": env_config.voice_tts_provider,
    }
