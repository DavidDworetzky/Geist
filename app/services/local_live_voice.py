import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import anyio
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect

from app.models.local_live_voice import LocalLiveVoiceConfig
from app.services.moshi_voice import MODEL_ID, MOSHI_PROVIDER, serve_moshi


logger = logging.getLogger(__name__)
READY_TIMEOUT = 120
IDLE_TIMEOUT = 20
CALL_TIMEOUT = 300


@dataclass(frozen=True)
class LocalVoiceBackend:
    backend: str
    transport: LocalLiveVoiceConfig
    url: str | None = None


def local_voice_backend() -> LocalVoiceBackend:
    backend = os.environ.get("GEIST_LOCAL_LIVE_VOICE_BACKEND", "moshi")
    if backend == "moshi":
        return LocalVoiceBackend(backend, LocalLiveVoiceConfig(model=MODEL_ID))
    try:
        if backend != "websocket":
            raise ValueError("Unsupported backend")
        url = os.environ.get("GEIST_LOCAL_LIVE_VOICE_URL", "")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("Invalid WebSocket URL")
        # Accessing port also validates malformed or out-of-range ports.
        _ = parsed.port
        transport = LocalLiveVoiceConfig.model_validate(
            {
                "model": os.environ.get("GEIST_LOCAL_LIVE_VOICE_MODEL", "local-voice"),
                "sample_rate": int(os.environ.get("GEIST_LOCAL_LIVE_VOICE_SAMPLE_RATE", "24000")),
                "frame_samples": int(
                    os.environ.get("GEIST_LOCAL_LIVE_VOICE_FRAME_SAMPLES", "1920")
                ),
            }
        )
        if (
            not transport.sample_rate // 100
            <= transport.frame_samples
            <= transport.sample_rate // 10
        ):
            raise ValueError("Frames must contain between 10 and 100 milliseconds of audio")
        return LocalVoiceBackend(backend, transport, url)
    except ValueError as error:
        raise HTTPException(
            503, "Check the server's GEIST_LOCAL_LIVE_VOICE configuration."
        ) from error


def local_voice_provider() -> dict:
    try:
        selected = local_voice_backend()
    except HTTPException:
        selected = LocalVoiceBackend("unavailable", LocalLiveVoiceConfig(model="local-voice"))
    config = selected.transport
    if selected.backend == "moshi":
        model: dict[str, Any] = {**MOSHI_PROVIDER["models"][0]}
        description = (
            "Moshi on Apple Silicon. English; five-minute calls. Captions show Moshi's speech."
        )
    else:
        model = {
            "id": config.model,
            "display_name": config.model,
            "sample_rate": config.sample_rate,
            "supports_streaming": True,
            "streaming_mode": "websocket",
            "voices": [],
            "languages": [],
            "supports_instruction_control": False,
            "supports_voice_cloning": False,
        }
        description = (
            "Uses the configured local voice server. Model and voice are managed by that server."
        )
    if selected.backend == "unavailable":
        description = "Local voice configuration is invalid. Check the server's GEIST_LOCAL_LIVE_VOICE settings."
    return {
        "provider": "local_live",
        "display_name": "Local live voice",
        "description": description,
        "type": "local",
        "mode": "conversation",
        "default_model": config.model,
        "models": [model],
    }


async def relay_local_voice(websocket: WebSocket, selected: LocalVoiceBackend) -> None:
    await websocket.accept()
    frame_bytes = selected.transport.frame_samples * 4
    ready = False
    upstream = None
    tasks: list[asyncio.Task] = []

    async def receive() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            audio = message.get("bytes")
            if not ready or upstream is None or audio is None or len(audio) != frame_bytes:
                raise ValueError("Invalid local voice input frame")
            await asyncio.wait_for(upstream.send(audio), timeout=1)

    async def send() -> None:
        nonlocal ready, upstream
        assert selected.url is not None
        async with connect(
            selected.url,
            proxy=None,
            compression=None,
            open_timeout=10,
            close_timeout=2,
            max_size=max(frame_bytes, 24576),
            max_queue=4,
            write_limit=frame_bytes * 4,
        ) as connection:
            upstream = connection
            while True:
                message = await asyncio.wait_for(
                    connection.recv(), timeout=IDLE_TIMEOUT if ready else READY_TIMEOUT
                )
                if isinstance(message, bytes):
                    if not ready or len(message) != frame_bytes:
                        raise ValueError("Invalid local voice output frame")
                    await asyncio.wait_for(websocket.send_bytes(message), timeout=1)
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "ready" and not ready:
                    if (
                        event.get("protocol") != selected.transport.protocol
                        or event.get("sample_rate") != selected.transport.sample_rate
                        or event.get("frame_samples") != selected.transport.frame_samples
                    ):
                        raise ValueError(
                            "Local voice server audio format does not match configuration"
                        )
                    ready = True
                    await websocket.send_json({"type": "ready"})
                elif kind == "transcript" and ready:
                    role, text = event.get("role", "assistant"), event.get("text")
                    if (
                        role not in {"user", "assistant"}
                        or not isinstance(text, str)
                        or len(text) > 4000
                    ):
                        raise ValueError("Invalid local voice transcript")
                    await websocket.send_json({"type": "transcript", "role": role, "text": text})
                else:
                    raise ValueError("Local voice server error or invalid event")

    try:
        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        done, _ = await asyncio.wait(
            tasks, timeout=CALL_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            await websocket.send_json(
                {"type": "error", "message": "Voice call reached five minutes. Start another call."}
            )
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Local live voice relay failed")
        with contextlib.suppress(RuntimeError):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Local voice server stopped responding. Check the server and reconnect.",
                }
            )
    finally:
        with anyio.CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(RuntimeError):
                await websocket.close()


async def serve_local_voice(websocket: WebSocket) -> None:
    try:
        selected = local_voice_backend()
    except HTTPException as error:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": error.detail})
        await websocket.close(code=1011)
        return
    if selected.backend == "moshi":
        await serve_moshi(websocket)
    else:
        await relay_local_voice(websocket, selected)
