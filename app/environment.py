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

def LoadEnvironment() -> EnvironmentConfig:
    openai_key = os.getenv("OPENAI_TOKEN")
    twilio_key = os.getenv("TWILIO_TOKEN")
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_source = os.getenv("TWILIO_SOURCE")

    return EnvironmentConfig(
        openai_key = openai_key,
        twilio_key = twilio_key,
        twilio_sid = twilio_sid,
        twilio_source=twilio_source,
    )

def LoadEnvironmentDictionary() -> Dict[str, str]:
    env_config = LoadEnvironment()
    return {
        "openai_key": env_config.openai_key,
        "twilio_key": env_config.twilio_key,
        "twilio_sid": env_config.twilio_sid,
        "twilio_source": env_config.twilio_source,
    }
