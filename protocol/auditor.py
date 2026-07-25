"""Sampling audits: re-execute an unpredictable subset and compare outputs."""

import math
from dataclasses import dataclass

from protocol.hashing import MerkleTree
from protocol.models import (
    AuditReport,
    AuditVerdict,
    BlockResult,
    ComputeBlock,
    SampledCheck,
)
from protocol.runner import InferenceRunner
from protocol.sampling import sample_indices


@dataclass(frozen=True)
class AuditPolicy:
    """How many requests to sample and how strictly to compare outputs.

    ``logprob_tolerance`` absorbs benign nondeterminism (kernel scheduling,
    hardware differences) when comparing per-token logprobs; output text must
    always match exactly for temperature-0/seeded decoding.
    """

    sample_size: int = 8
    logprob_tolerance: float = 1e-3

    def records_equivalent(self, claimed, reference) -> bool:
        if claimed.output_text != reference.output_text:
            return False
        if len(claimed.token_logprobs) != len(reference.token_logprobs):
            return False
        return all(
            abs(a - b) <= self.logprob_tolerance
            for a, b in zip(claimed.token_logprobs, reference.token_logprobs, strict=True)
        )


def detection_probability(cheat_fraction: float, sample_size: int) -> float:
    """Probability that at least one corrupted record lands in the sample."""
    if not 0.0 <= cheat_fraction <= 1.0:
        raise ValueError("cheat_fraction must be in [0, 1]")
    return 1.0 - math.pow(1.0 - cheat_fraction, sample_size)


class Auditor:
    """Verifies a committed block result against a trusted reference runner."""

    def __init__(self, reference_runner: InferenceRunner, policy: AuditPolicy | None = None):
        self._runner = reference_runner
        self._policy = policy or AuditPolicy()

    def audit(self, block: ComputeBlock, result: BlockResult, beacon_nonce: str) -> AuditReport:
        tree = MerkleTree([record.record_hash for record in result.records])
        indices = sample_indices(
            beacon_nonce, block.block_id, len(block.requests), self._policy.sample_size
        )
        checks = tuple(self._check_index(block, result, tree, index) for index in indices)
        verdict = (
            AuditVerdict.PASS
            if tree.root == result.merkle_root
            and all(check.inclusion_valid and check.output_matches for check in checks)
            else AuditVerdict.FAIL
        )
        return AuditReport(
            block_id=block.block_id,
            provider_id=result.provider_id,
            beacon_nonce=beacon_nonce,
            sampled_indices=indices,
            checks=checks,
            verdict=verdict,
        )

    def _check_index(
        self, block: ComputeBlock, result: BlockResult, tree: MerkleTree, index: int
    ) -> SampledCheck:
        request = block.requests[index]
        claimed = result.records[index]
        proof = tree.proof(index)
        inclusion_valid = (
            claimed.request_id == request.request_id
            and MerkleTree.verify(claimed.record_hash, proof, result.merkle_root)
        )
        reference = self._runner.run(block.model_id, request)
        return SampledCheck(
            request_id=request.request_id,
            leaf_index=index,
            inclusion_valid=inclusion_valid,
            output_matches=self._policy.records_equivalent(claimed, reference),
            claimed_output=claimed.output_text,
            reference_output=reference.output_text,
        )
