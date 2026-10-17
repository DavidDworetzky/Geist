from typing import Literal

from pydantic import BaseModel, Field


class LocalLiveVoiceConfig(BaseModel):
    protocol: Literal["geist-pcm-v1"] = "geist-pcm-v1"
    endpoint: Literal["/api/v1/voice/live/local"] = "/api/v1/voice/live/local"
    model: str
    sample_rate: Literal[16000, 24000, 48000] = 24000
    frame_samples: int = Field(default=1920, ge=128, le=4800)
