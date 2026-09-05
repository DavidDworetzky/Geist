"""Small framed IPC client for isolated local TTS workers."""

from __future__ import annotations

import atexit
import json
import select
import struct
import subprocess  # nosec B404 - private local worker, argv only, never a shell
import threading
from collections.abc import Iterator, Sequence
from typing import IO, Any


FRAME_HEADER = struct.Struct(">cI")
MAX_CONTROL_FRAME = 1024 * 1024
MAX_AUDIO_FRAME = 16 * 1024 * 1024


class LocalTTSProcess:
    """Own one private worker and stream its PCM frames synchronously."""

    def __init__(self, command: Sequence[str], *, startup_timeout: float = 120.0) -> None:
        self.command = tuple(command)
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.sample_rate: int | None = None
        atexit.register(self.close)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        self._process = subprocess.Popen(  # nosec B603 - command is constructed by the provider, not request text
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        try:
            frame_type, payload = self._read_frame(timeout=self.startup_timeout)
        except Exception:
            self.close()
            raise
        if frame_type == b"E":
            self.close()
            raise RuntimeError(_error_message(payload))
        if frame_type != b"R":
            self.close()
            raise RuntimeError("Local TTS worker did not return a ready frame")
        metadata = _decode_control(payload)
        self.sample_rate = int(metadata["sample_rate"])

    def synthesize(self, request: dict[str, Any]) -> Iterator[bytes]:
        with self._lock:
            self._ensure_started()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Local TTS worker is unavailable")
            completed = False
            try:
                process.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
                process.stdin.flush()
                while True:
                    frame_type, payload = self._read_frame(timeout=120.0)
                    if frame_type == b"A":
                        if payload:
                            yield payload
                        continue
                    if frame_type == b"D":
                        completed = True
                        return
                    if frame_type == b"E":
                        raise RuntimeError(_error_message(payload))
                    raise RuntimeError("Local TTS worker returned an unknown frame")
            finally:
                if not completed:
                    # An interrupted exchange cannot be reused: unread frames
                    # belong to the old utterance, not the next request.
                    if process.poll() is None:
                        process.terminate()
                    self.close()

    def _read_frame(self, *, timeout: float) -> tuple[bytes, bytes]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Local TTS worker is not running")
        header = _read_exact(process.stdout, FRAME_HEADER.size, timeout=timeout)
        frame_type, length = FRAME_HEADER.unpack(header)
        if frame_type != b"A" and length > MAX_CONTROL_FRAME:
            raise RuntimeError("Local TTS worker returned an oversized control frame")
        if frame_type == b"A" and length > MAX_AUDIO_FRAME:
            raise RuntimeError("Local TTS worker returned an oversized audio frame")
        return frame_type, _read_exact(process.stdout, length, timeout=timeout)

    def close(self) -> None:
        process = self._process
        self._process = None
        self.sample_rate = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b'{"action":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


def write_frame(stream: IO[bytes], frame_type: bytes, payload: bytes = b"") -> None:
    """Write one worker protocol frame."""

    stream.write(FRAME_HEADER.pack(frame_type, len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: IO[bytes], length: int, *, timeout: float) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        readable, _, _ = select.select([stream], [], [], timeout)
        if not readable:
            raise TimeoutError("Timed out waiting for the local TTS worker")
        chunk = stream.read(length - len(payload))
        if not chunk:
            raise RuntimeError("Local TTS worker exited unexpectedly")
        payload.extend(chunk)
    return bytes(payload)


def _decode_control(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise RuntimeError("Local TTS worker returned invalid metadata") from error
    if not isinstance(value, dict):
        raise RuntimeError("Local TTS worker metadata must be an object")
    return value


def _error_message(payload: bytes) -> str:
    try:
        return str(_decode_control(payload).get("message") or "Local TTS worker failed")
    except RuntimeError:
        return "Local TTS worker failed"
