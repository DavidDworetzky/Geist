import logging
import os

import httpx
from fastapi import HTTPException

from app.models.live_voice import LiveSessionRequest, LiveSessionResponse


logger = logging.getLogger(__name__)
LIVE_VOICES = [
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
]
LIVE_PROVIDER = {
    "provider": "openai_live",
    "display_name": "OpenAI GPT-Live (online)",
    "type": "api",
    "mode": "conversation",
    "default_model": "gpt-live-1",
    "models": [
        {
            "id": "gpt-live-1",
            "display_name": "GPT-Live 1",
            "sample_rate": 24000,
            "supports_streaming": True,
            "streaming_mode": "webrtc",
            "supports_instruction_control": False,
            "supports_voice_cloning": False,
            "voices": [{"id": voice, "display_name": voice.title()} for voice in LIVE_VOICES],
            "languages": [{"code": "auto", "display_name": "Auto"}],
        }
    ],
}


async def create_live_session(offer: LiveSessionRequest) -> LiveSessionResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "Set OPENAI_API_KEY on the server to use GPT-Live.")

    payload = {
        "session": {
            "model": offer.model,
            "audio": {"output": {"voice": offer.voice}},
            "instructions": (
                "You are Geist's live voice assistant. Have a natural, responsive spoken "
                "conversation. This is a voice-only call with no backend tools or text agent. "
                "Answer directly, keep replies concise, and listen to interruptions. "
                "Do not delegate requests or claim to perform external actions. "
                "If a task needs tools or detailed text work, suggest using the text chat."
            ),
            "delegation": {"type": "client"},
        },
        "transport": {"type": "webrtc", "sdp": offer.sdp},
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/live/sessions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        if response.is_error:
            logger.warning("GPT-Live session rejected with HTTP %s", response.status_code)
            detail = {
                401: "OpenAI rejected the server API key.",
                403: "The OpenAI project does not have GPT-Live access.",
                404: "GPT-Live is unavailable for this OpenAI project.",
                429: "OpenAI voice quota or rate limit reached. Try again later.",
            }.get(response.status_code, "OpenAI could not start the GPT-Live session.")
            raise HTTPException(502, detail)
        result = response.json()
        session_id = result["session"]["id"]
        sdp = result["transport"]["sdp"]
        if not isinstance(session_id, str) or not session_id or not isinstance(sdp, str) or not sdp:
            raise ValueError("Missing session ID or SDP")
        return LiveSessionResponse(session_id=session_id, sdp=sdp)
    except httpx.TimeoutException as error:
        raise HTTPException(504, "OpenAI timed out while starting GPT-Live.") from error
    except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(502, "Invalid or unavailable GPT-Live session response.") from error
