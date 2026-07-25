"""Unpredictable but verifiable audit sampling.

Indices are drawn from an HMAC-SHA256 stream keyed by a randomness-beacon
nonce that is only revealed *after* the provider commits to its results, so
the provider cannot predict which requests will be audited. Any party holding
the nonce and block id can recompute the exact same sample and verify the
auditor did not choose indices adversarially.
"""

import hashlib
import hmac


def _hmac_stream(key: bytes, message: bytes):
    counter = 0
    while True:
        block = hmac.new(key, message + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        yield int.from_bytes(block[:8], "big")
        counter += 1


def sample_indices(beacon_nonce: str, block_id: str, population: int, k: int) -> tuple[int, ...]:
    """Deterministically draw k distinct indices from range(population)."""
    if population <= 0:
        raise ValueError("population must be positive")
    k = min(k, population)
    chosen: list[int] = []
    seen: set[int] = set()
    stream = _hmac_stream(beacon_nonce.encode("utf-8"), block_id.encode("utf-8"))
    for draw in stream:
        index = draw % population
        if index not in seen:
            seen.add(index)
            chosen.append(index)
            if len(chosen) == k:
                return tuple(chosen)
