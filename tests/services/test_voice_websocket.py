import asyncio
import threading
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from starlette.websockets import WebSocketState

from app.api.v1.endpoints import voice


class Socket:
    client_state = WebSocketState.CONNECTED

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, data):
        self.outgoing.put_nowait(data)

    async def send_bytes(self, data):
        self.outgoing.put_nowait({"type": "audio", "audio": data})

    async def close(self):
        self.client_state = WebSocketState.DISCONNECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "reset"])
async def test_control_is_acknowledged_during_blocking_inference(control):
    entered, release = threading.Event(), threading.Event()

    def synthesize(text):
        entered.set()
        release.wait(5)
        yield b"stale audio"

    agent = Mock()
    agent.stream_complete_text.return_value = iter(["Hello."])
    tts = Mock(sample_rate=24000)
    tts.synthesize_streaming.side_effect = synthesize
    socket = Socket()
    with (
        patch(
            "app.services.voice_session.create_stt_adapter",
            return_value=Mock(transcribe=Mock(return_value="hello")),
        ),
        patch("app.services.voice_session.create_tts_provider", return_value=tts),
        patch.object(voice, "get_agent_for_session", new=AsyncMock(return_value=agent)),
        patch.object(voice, "get_default_agent_context", return_value=Mock()),
    ):
        task = asyncio.create_task(
            voice.voice_stream_websocket(
                socket,
                session_id=1,
                agent_type="local",
                stt_provider="mms",
                tts_provider="kokoro",
                tts_model=None,
                tts_voice=None,
                tts_language=None,
                tts_instruct=None,
                tts_speed=None,
            )
        )
        try:
            assert (await asyncio.wait_for(socket.outgoing.get(), 2))["type"] == "ready"
            socket.incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "bytes": np.full(1600, 4000, dtype=np.int16).tobytes(),
                }
            )
            socket.incoming.put_nowait(
                {"type": "websocket.receive", "bytes": np.zeros(12800, dtype=np.int16).tobytes()}
            )
            assert await asyncio.to_thread(entered.wait, 2)
            socket.incoming.put_nowait(
                {"type": "websocket.receive", "text": '{"type":"' + control + '"}'}
            )

            async def acknowledgment():
                while True:
                    message = await socket.outgoing.get()
                    assert message["type"] != "audio"
                    if message["type"] in {"stopped", "reset_complete"}:
                        return message["type"]

            assert await asyncio.wait_for(acknowledgment(), 0.5) == (
                "stopped" if control == "stop" else "reset_complete"
            )
        finally:
            release.set()
            socket.incoming.put_nowait({"type": "websocket.disconnect"})
            await asyncio.wait_for(task, 2)
