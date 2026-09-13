"""Moshi MLX worker. stdin: PCM float32 frames; stdout: framed PCM/text events."""

import contextlib
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path


MODEL_ID = "kyutai/moshiko-mlx-q4"
REVISION = "18e4df760a34d5977a34517d7d1580e07acbb2f1"
FRAME_SAMPLES = 1920
MAX_STEPS = 3750
OUTPUT = sys.stdout.buffer
OUTPUT_LOCK = threading.Lock()


def emit(kind: int, payload: bytes = b"") -> None:
    with OUTPUT_LOCK:
        OUTPUT.write(struct.pack("<BI", kind, len(payload)) + payload)
        OUTPUT.flush()


def weight_path(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parents[3]
    local = root / "app" / "model_weights" / MODEL_ID.replace("/", "_") / filename
    if local.is_file():
        return str(local)
    return hf_hub_download(MODEL_ID, filename, revision=REVISION, local_files_only=True)


def wait_codec(read):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = read()
        if value is not None:
            return value
        time.sleep(0.001)
    raise TimeoutError("Mimi audio codec timed out")


def run() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import rustymimi
    import sentencepiece
    from moshi_mlx import models, utils

    tokenizer = sentencepiece.SentencePieceProcessor(weight_path("tokenizer_spm_32k_3.model"))
    model = models.Lm(models.config_v0_1())
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32)
    model.load_weights(weight_path("model.q4.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.warmup(None)
    generator = models.LmGen(
        model=model,
        max_steps=MAX_STEPS + 5,
        text_sampler=utils.Sampler(),
        audio_sampler=utils.Sampler(),
        check=False,
    )
    codec = rustymimi.StreamTokenizer(
        weight_path("tokenizer-e351c8d8-checkpoint125.safetensors"),
        num_codebooks=8,
    )
    # Mimi decoding overlaps the next model step instead of delaying its input.
    audio_queue: queue.Queue[bool] = queue.Queue(maxsize=10)
    decoder_errors: list[Exception] = []

    def write_audio() -> None:
        try:
            while audio_queue.get():
                audio = wait_codec(codec.get_decoded)
                emit(1, np.asarray(audio, dtype="<f4").tobytes())
        except Exception as error:
            decoder_errors.append(error)

    decoder = threading.Thread(target=write_audio, daemon=True)
    decoder.start()
    emit(0)
    for _ in range(MAX_STEPS):
        data = sys.stdin.buffer.read(FRAME_SAMPLES * 4)
        if len(data) != FRAME_SAMPLES * 4:
            break
        if decoder_errors:
            raise decoder_errors[0]
        pcm = np.frombuffer(data, dtype="<f4").copy()
        np.nan_to_num(pcm, copy=False, nan=0, posinf=1, neginf=-1)
        np.clip(pcm, -1, 1, out=pcm)
        codec.encode(pcm)
        encoded = wait_codec(codec.get_encoded)
        codes = mx.array(encoded).transpose(1, 0)[:, : generator.main_codebooks]
        token = generator.step(codes, ct=None)[0].item()
        audio_tokens = generator.last_audio_tokens()
        if token not in (0, 3):
            emit(2, tokenizer.id_to_piece(token).replace("▁", " ").encode("utf-8"))
        if audio_tokens is not None:
            codec.decode(np.array(audio_tokens).astype(np.uint32))
            audio_queue.put(True, timeout=10)
        emit(4)
    else:
        emit(3, b"Moshi's five-minute session limit was reached. Start another call.")
    audio_queue.put(False, timeout=10)
    decoder.join(timeout=12)
    if decoder_errors:
        raise decoder_errors[0]
    if decoder.is_alive():
        raise TimeoutError("Mimi decoding did not finish")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            run()
    except Exception as error:
        # Never put arbitrary provider errors or local configuration on the wire.
        print(f"Moshi worker failed: {type(error).__name__}", file=sys.stderr)
        emit(3, b"Moshi could not run. Verify the isolated runtime and downloaded model weights.")
        raise SystemExit(1) from error
