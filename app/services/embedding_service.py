"""Small deterministic local embedder used by the first memory index."""

import hashlib
import math
import re
import struct


EMBEDDING_MODEL_ID = "geist-feature-hash-v1"
EMBEDDING_DIMENSIONS = 256
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")


def embed_text(text: str) -> list[float]:
    """Return a normalized feature-hashed token and character-ngram vector."""
    normalized = text.lower()
    tokens = _TOKEN_RE.findall(normalized)
    features = tokens + [
        f"#{token[index:index + 3]}"
        for token in tokens
        for index in range(max(0, len(token) - 2))
    ]
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % EMBEDDING_DIMENSIONS
        vector[index] += -1.0 if value & (1 << 63) else 1.0
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude:
        vector = [value / magnitude for value in vector]
    return vector


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(value: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"<{dimensions}f", value))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))
