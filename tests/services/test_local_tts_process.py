"""Tests for the framed local TTS worker process contract."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from app.services.local_tts_process import LocalTTSProcess


WORKER = r"""
import json
import struct
import sys

header = struct.Struct('>cI')

def frame(kind, payload=b''):
    sys.stdout.buffer.write(header.pack(kind, len(payload)) + payload)
    sys.stdout.buffer.flush()

frame(b'R', json.dumps({'sample_rate': 22050}).encode())
for line in sys.stdin.buffer:
    request = json.loads(line)
    if request.get('action') == 'shutdown':
        break
    frame(b'A', b'pcm-one')
    frame(b'A', b'pcm-two')
    frame(b'D', json.dumps({'sample_rate': 22050}).encode())
"""


def test_local_tts_process_streams_binary_frames_and_reuses_worker():
    process = LocalTTSProcess((sys.executable, "-c", WORKER), startup_timeout=5)
    try:
        assert list(process.synthesize({"text": "hello"})) == [b"pcm-one", b"pcm-two"]
        assert process.sample_rate == 22_050
        first_pid = process._process.pid

        assert list(process.synthesize({"text": "again"})) == [b"pcm-one", b"pcm-two"]
        assert process._process.pid == first_pid
    finally:
        process.close()


def test_local_tts_process_surfaces_worker_error():
    worker = r"""
import json
import struct
import sys
payload = json.dumps({'message': 'model load failed'}).encode()
sys.stdout.buffer.write(struct.Struct('>cI').pack(b'E', len(payload)) + payload)
sys.stdout.buffer.flush()
"""
    process = LocalTTSProcess((sys.executable, "-c", worker), startup_timeout=5)

    with pytest.raises(RuntimeError, match="model load failed"):
        list(process.synthesize({"text": "hello"}))

    process.close()


def test_abandoned_utterance_restarts_worker_without_stale_audio():
    process = LocalTTSProcess((sys.executable, "-c", WORKER), startup_timeout=5)
    try:
        stream = process.synthesize({"text": "old"})
        assert next(stream) == b"pcm-one"
        old_process = process._process
        stream.close()
        assert old_process.poll() is not None
        assert list(process.synthesize({"text": "new"})) == [b"pcm-one", b"pcm-two"]
        assert process._process.pid != old_process.pid
    finally:
        process.close()


def test_timeout_invalidates_exchange_before_next_request():
    process = LocalTTSProcess((sys.executable, "-c", WORKER), startup_timeout=5)
    try:
        process._ensure_started()
        old_process = process._process
        with (
            patch.object(process, "_read_frame", side_effect=TimeoutError("test timeout")),
            pytest.raises(TimeoutError),
        ):
            list(process.synthesize({"text": "old"}))
        assert old_process.poll() is not None
        assert list(process.synthesize({"text": "new"})) == [b"pcm-one", b"pcm-two"]
    finally:
        process.close()
