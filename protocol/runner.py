"""Inference runner interface plus a deterministic stub for tests and audits."""

import hashlib
from typing import Protocol

from protocol.models import InferenceRecord, InferenceRequest


class InferenceRunner(Protocol):
    """Executes one inference request deterministically for a given model.

    Determinism is the protocol's core requirement: with temperature 0 (or a
    fixed seed on a deterministic backend), the auditor's re-execution must
    reproduce the provider's output token-for-token.
    """

    def run(self, model_id: str, request: InferenceRequest) -> InferenceRecord:
        ...


class DeterministicStubRunner:
    """Reference runner that derives output purely from the request content.

    Stands in for a real model in tests and protocol simulations: identical
    requests always produce identical records, so honest providers and
    auditors agree byte-for-byte.
    """

    def run(self, model_id: str, request: InferenceRequest) -> InferenceRecord:
        digest = hashlib.sha256(
            f"{model_id}|{request.prompt}|{request.seed}|{request.temperature}".encode()
        ).hexdigest()
        token_count = min(request.max_tokens, 8)
        logprobs = tuple(
            -float(int(digest[i * 2 : i * 2 + 2], 16)) / 256.0 for i in range(token_count)
        )
        return InferenceRecord(
            request_id=request.request_id,
            output_text=f"stub:{digest[:16]}",
            token_logprobs=logprobs,
        )


class TamperingRunner:
    """Wraps a runner and corrupts a chosen set of requests.

    Simulates a lazy or malicious provider (skipping work, serving a cheaper
    model, fabricating output) for audit tests and detection-rate simulations.
    """

    def __init__(self, inner: InferenceRunner, corrupt_request_ids: set[str]):
        self._inner = inner
        self._corrupt = corrupt_request_ids

    def run(self, model_id: str, request: InferenceRequest) -> InferenceRecord:
        record = self._inner.run(model_id, request)
        if request.request_id in self._corrupt:
            return InferenceRecord(
                request_id=record.request_id,
                output_text=record.output_text + ":forged",
                token_logprobs=record.token_logprobs,
            )
        return record
