import asyncio
import contextlib
import logging
import os
import platform
import struct
from pathlib import Path
from typing import Any

import anyio
from fastapi import WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)

MODEL_ID = "kyutai/moshiko-mlx-q4"
MOSHI_PROVIDER: dict[str, Any] = {
    "provider": "moshi",
    "display_name": "Moshi (local · MLX)",
    "type": "local",
    "mode": "conversation",
    "default_model": MODEL_ID,
    "models": [
        {
            "id": MODEL_ID,
            "display_name": "Moshiko 7B · 4-bit",
            "sample_rate": 24000,
            "supports_streaming": True,
            "streaming_mode": "websocket",
            "supports_instruction_control": False,
            "supports_voice_cloning": False,
            "voices": [{"id": "moshiko", "display_name": "Moshiko"}],
            "languages": [{"code": "en", "display_name": "English"}],
        }
    ],
}
_active = False


def worker_python() -> Path:
    root = Path(__file__).resolve().parents[2]
    return Path(
        os.environ.get("GEIST_MOSHI_PYTHON", str(root / "scripts/runtimes/moshi/.venv/bin/python"))
    )


async def serve_moshi(websocket: WebSocket) -> None:
    global _active
    await websocket.accept()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        await websocket.send_json(
            {"type": "error", "message": "Moshi MLX requires the native Apple Silicon backend."}
        )
        await websocket.close(code=1008)
        return
    if _active:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Moshi is already in a voice call. End it before starting another.",
            }
        )
        await websocket.close(code=1013)
        return
    executable = worker_python()
    if not executable.is_file():
        await websocket.send_json(
            {
                "type": "error",
                "message": "Moshi needs its local runtime. Follow scripts/runtimes/moshi/README.md to set it up.",
            }
        )
        await websocket.close(code=1011)
        return
    _active = True
    process = None
    tasks: list[asyncio.Task] = []
    stderr_task: asyncio.Task | None = None
    try:
        script = Path(__file__).resolve().parents[2] / "scripts/runtimes/moshi/worker.py"
        process = await asyncio.create_subprocess_exec(
            str(executable),
            str(script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None
        assert process.stderr is not None
        stderr = process.stderr

        async def log_stderr() -> None:
            # Bounded reads also drain diagnostics without newlines or very long lines.
            while chunk := await stderr.read(4096):
                logger.warning("Moshi worker: %s", chunk.decode("utf-8", errors="replace").rstrip())

        stderr_task = asyncio.create_task(log_stderr())
        writer, reader = process.stdin, process.stdout
        pending = 0
        ready = False

        async def receive() -> None:
            nonlocal pending
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is None:
                    return
                if len(data) != 1920 * 4:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid Moshi audio frame."}
                    )
                    return
                if not ready or pending >= 10:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Moshi cannot keep up with live audio on this device. Reconnect to try again.",
                        }
                    )
                    return
                pending += 1
                writer.write(data)
                await writer.drain()

        async def send() -> None:
            nonlocal ready, pending
            while True:
                header = await asyncio.wait_for(
                    reader.readexactly(5), timeout=120 if not ready else 20
                )
                kind, size = struct.unpack("<BI", header)
                if size > 192000:
                    raise ValueError("Invalid worker frame")
                data = await reader.readexactly(size)
                if kind == 0:
                    ready = True
                    await websocket.send_json({"type": "ready"})
                elif kind == 1:
                    await websocket.send_bytes(data)
                elif kind == 2:
                    await websocket.send_json({"type": "transcript", "text": data.decode("utf-8")})
                elif kind == 3:
                    await websocket.send_json({"type": "error", "message": data.decode("utf-8")})
                    return
                elif kind == 4:
                    pending = max(0, pending - 1)
                else:
                    raise ValueError("Invalid worker event")

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Moshi voice session failed")
        with contextlib.suppress(RuntimeError):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Moshi stopped responding. End the call and reconnect.",
                }
            )
    finally:
        with anyio.CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if process is not None and process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            if stderr_task is not None:
                try:
                    await asyncio.wait_for(stderr_task, timeout=1)
                except TimeoutError:
                    logger.warning("Moshi worker stderr did not close after shutdown")
            _active = False
            with contextlib.suppress(RuntimeError):
                await websocket.close()
