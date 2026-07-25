"""Geist compute protocol: verifiable blocks of LLM inference with sampling audits.

Individuals submit blocks of inference work, providers execute them and commit
to the results with a Merkle root, and auditors re-execute an unpredictable
random sample of requests to catch cheating providers. Payment is escrowed per
block and released on a clean audit; a failed audit slashes the provider's
stake and refunds the client.

See docs/compute_protocol.md for the full specification.
"""

from protocol.auditor import Auditor, AuditPolicy, detection_probability
from protocol.coordinator import Coordinator, ProtocolConfig, ProtocolError
from protocol.hashing import MerkleTree, canonical_hash
from protocol.ledger import Ledger, LedgerError
from protocol.models import (
    AuditReport,
    AuditVerdict,
    BlockResult,
    BlockStatus,
    ComputeBlock,
    InferenceRecord,
    InferenceRequest,
    Provider,
)
from protocol.runner import DeterministicStubRunner, InferenceRunner
from protocol.sampling import sample_indices


__all__ = [
    "AuditPolicy",
    "AuditReport",
    "AuditVerdict",
    "Auditor",
    "BlockResult",
    "BlockStatus",
    "ComputeBlock",
    "Coordinator",
    "DeterministicStubRunner",
    "InferenceRecord",
    "InferenceRequest",
    "InferenceRunner",
    "Ledger",
    "LedgerError",
    "MerkleTree",
    "Provider",
    "ProtocolConfig",
    "ProtocolError",
    "canonical_hash",
    "detection_probability",
    "sample_indices",
]
