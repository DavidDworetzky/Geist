"""Data model for compute blocks, results, providers, and audits."""

from dataclasses import dataclass, field
from enum import StrEnum

from protocol.hashing import canonical_hash


class BlockStatus(StrEnum):
    SUBMITTED = "submitted"
    ASSIGNED = "assigned"
    COMMITTED = "committed"
    PASSED = "passed"
    FAILED = "failed"
    SETTLED = "settled"


class AuditVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class InferenceRequest:
    """One inference call inside a block. Decoding must be deterministic."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ComputeBlock:
    """An ordered batch of inference requests against one model."""

    model_id: str
    requests: tuple[InferenceRequest, ...]
    client_id: str
    price: int

    def __post_init__(self):
        if not self.requests:
            raise ValueError("a compute block needs at least one request")
        if self.price <= 0:
            raise ValueError("block price must be positive")

    @property
    def block_id(self) -> str:
        return canonical_hash(
            {
                "model_id": self.model_id,
                "client_id": self.client_id,
                "price": self.price,
                "requests": [request.to_dict() for request in self.requests],
            }
        )


@dataclass(frozen=True)
class InferenceRecord:
    """A provider's claimed output for one request."""

    request_id: str
    output_text: str
    token_logprobs: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "output_text": self.output_text,
            "token_logprobs": list(self.token_logprobs),
        }

    @property
    def record_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class BlockResult:
    """Provider submission: full records plus the Merkle commitment over them."""

    block_id: str
    provider_id: str
    merkle_root: str
    records: tuple[InferenceRecord, ...]


@dataclass(frozen=True)
class SampledCheck:
    """Outcome of re-executing one sampled request."""

    request_id: str
    leaf_index: int
    inclusion_valid: bool
    output_matches: bool
    claimed_output: str
    reference_output: str


@dataclass(frozen=True)
class AuditReport:
    block_id: str
    provider_id: str
    beacon_nonce: str
    sampled_indices: tuple[int, ...]
    checks: tuple[SampledCheck, ...]
    verdict: AuditVerdict


@dataclass
class Provider:
    provider_id: str
    reputation: float = 1.0
    blocks_completed: int = 0
    audits_failed: int = 0
    supported_models: set[str] = field(default_factory=set)
