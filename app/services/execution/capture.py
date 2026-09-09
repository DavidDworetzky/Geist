"""Bound host memory while draining subprocess output."""

from __future__ import annotations

import os
import select
import subprocess  # nosec B404 - capture only; callers own approved execution
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import IO

from app.services.execution.base import ExecutionResult, truncate_output


MAX_CAPTURE_BYTES = 64 * 1024


def capture_process(
    process: subprocess.Popen[bytes], timeout: float, terminate: Callable[[], None]
) -> ExecutionResult:
    """Retain at most 64 KiB per stream; drain excess until completion or deadline.

    POSIX readers poll pipes so escaped descendants cannot strand reader threads.
    On Windows, a daemon reader owns its pipe until the descendant closes it.
    """
    started = time.monotonic()
    limit_reached = threading.Event()
    stopping = threading.Event()
    buffers = [bytearray(), bytearray()]
    lock = threading.Lock()
    finished = [threading.Event(), threading.Event()]

    def drain(index: int, stream: IO[bytes]) -> None:
        try:
            while not stopping.is_set():
                if os.name == "posix" and not select.select([stream], [], [], 0.05)[0]:
                    continue
                chunk = os.read(stream.fileno(), 8192)
                if not chunk or stopping.is_set():
                    break
                with lock:
                    remaining = MAX_CAPTURE_BYTES - len(buffers[index])
                    buffers[index].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    limit_reached.set()
        except OSError:
            pass
        finally:
            # Only the reader closes its fd, never another thread between reads.
            stream.close()
            finished[index].set()

    streams = (process.stdout, process.stderr)
    readers = []
    for index, stream in enumerate(streams):
        if stream is None:
            raise ValueError("Bounded capture requires stdout and stderr pipes")
        reader = threading.Thread(target=drain, args=(index, stream), daemon=True)
        reader.start()
        readers.append(reader)

    timed_out = False
    try:
        while True:
            if process.poll() is not None and all(event.is_set() for event in finished):
                break
            remaining_time = timeout - (time.monotonic() - started)
            if remaining_time <= 0:
                timed_out = True
                terminate()
                stopping.set()
                break
            stopping.wait(min(0.02, remaining_time))
        # Reap the direct child even if a detached descendant retains its pipes.
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        for reader in readers:
            reader.join(timeout=0.2)
    finally:
        stopping.set()

    with lock:
        stdout, stdout_truncated = truncate_output(buffers[0].decode(errors="replace"))
        stderr, stderr_truncated = truncate_output(buffers[1].decode(errors="replace"))
    if timed_out:
        stderr = stderr or f"Command timed out after {timeout:g} seconds"
    return ExecutionResult(
        exit_code=124 if timed_out else process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=time.monotonic() - started,
        timed_out=timed_out,
        truncated=limit_reached.is_set() or stdout_truncated or stderr_truncated,
    )
