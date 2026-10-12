from typing import Literal

from pydantic import BaseModel, Field


class LiveSessionRequest(BaseModel):
    tools_enabled: bool = False
    sdp: str = Field(min_length=1, max_length=65536, pattern=r"^v=0\r?\n")
    model: Literal["gpt-live-1"] = "gpt-live-1"
    voice: Literal[
        "marin",
        "quartz",
        "ripple",
        "vesper",
        "willow",
        "stone",
        "gleam",
        "meridian",
        "bossa",
        "tempo",
        "beacon",
        "delta",
        "cinder",
    ] = "marin"


class LiveSessionResponse(BaseModel):
    session_id: str
    sdp: str
