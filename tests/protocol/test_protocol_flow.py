import pytest

from protocol import (
    Auditor,
    AuditPolicy,
    AuditVerdict,
    BlockResult,
    BlockStatus,
    ComputeBlock,
    Coordinator,
    DeterministicStubRunner,
    InferenceRequest,
    Ledger,
    ProtocolConfig,
    ProtocolError,
    detection_probability,
    sample_indices,
)
from protocol.hashing import MerkleTree
from protocol.runner import TamperingRunner


MODEL = "llama-3.1-8b"
BEACON = "beacon-nonce-42"


def make_block(n_requests=20, price=500, client_id="client"):
    requests = tuple(
        InferenceRequest(request_id=f"req-{i}", prompt=f"prompt {i}", max_tokens=32)
        for i in range(n_requests)
    )
    return ComputeBlock(model_id=MODEL, requests=requests, client_id=client_id, price=price)


def make_coordinator(sample_size=8):
    ledger = Ledger()
    auditor = Auditor(DeterministicStubRunner(), AuditPolicy(sample_size=sample_size))
    coordinator = Coordinator(
        ledger=ledger, auditor=auditor, config=ProtocolConfig(min_stake=1_000)
    )
    ledger.deposit("client", 10_000)
    ledger.deposit("provider", 5_000)
    coordinator.register_provider("provider", stake=2_000, supported_models={MODEL})
    return coordinator, ledger


def execute_block(block, runner):
    records = tuple(runner.run(block.model_id, request) for request in block.requests)
    root = MerkleTree([record.record_hash for record in records]).root
    return BlockResult(
        block_id=block.block_id, provider_id="provider", merkle_root=root, records=records
    )


def test_honest_provider_passes_audit_and_gets_paid():
    coordinator, ledger = make_coordinator()
    block = make_block(price=500)

    block_id = coordinator.submit_block(block)
    assert ledger.account("client").balance == 9_500

    coordinator.assign_block(block_id, "provider")
    coordinator.submit_result(execute_block(block, DeterministicStubRunner()))

    report = coordinator.run_audit(block_id, BEACON)
    assert report.verdict == AuditVerdict.PASS

    coordinator.settle(block_id)
    fee = int(500 * coordinator.config.protocol_fee_fraction)
    assert ledger.account("provider").balance == 3_000 + 500 - fee
    assert ledger.account(coordinator.config.fee_account).balance == fee
    assert coordinator.providers["provider"].blocks_completed == 1
    assert coordinator.blocks[block_id].status == BlockStatus.SETTLED


def test_cheating_provider_is_caught_slashed_and_client_refunded():
    coordinator, ledger = make_coordinator()
    block = make_block(price=500)
    block_id = coordinator.submit_block(block)
    coordinator.assign_block(block_id, "provider")

    # Corrupt exactly the requests the beacon will sample, so detection is certain.
    sampled = sample_indices(BEACON, block_id, len(block.requests), 8)
    corrupt_ids = {block.requests[i].request_id for i in sampled[:1]}
    cheater = TamperingRunner(DeterministicStubRunner(), corrupt_ids)
    coordinator.submit_result(execute_block(block, cheater))

    report = coordinator.run_audit(block_id, BEACON)
    assert report.verdict == AuditVerdict.FAIL
    assert any(not check.output_matches for check in report.checks)

    coordinator.settle(block_id)
    # Client got the escrow back plus half the provider's 2,000 stake.
    assert ledger.account("client").balance == 10_000 + 1_000
    assert ledger.account("provider").stake == 1_000
    assert coordinator.providers["provider"].audits_failed == 1


def test_cheating_outside_the_sample_evades_this_audit():
    """Documents the residual risk: sampling audits are probabilistic."""
    coordinator, _ = make_coordinator(sample_size=4)
    block = make_block(n_requests=50)
    block_id = coordinator.submit_block(block)
    coordinator.assign_block(block_id, "provider")

    sampled = set(sample_indices(BEACON, block_id, len(block.requests), 4))
    unsampled = next(i for i in range(len(block.requests)) if i not in sampled)
    cheater = TamperingRunner(
        DeterministicStubRunner(), {block.requests[unsampled].request_id}
    )
    coordinator.submit_result(execute_block(block, cheater))

    assert coordinator.run_audit(block_id, BEACON).verdict == AuditVerdict.PASS


def test_result_with_mismatched_merkle_root_is_rejected():
    coordinator, _ = make_coordinator()
    block = make_block()
    block_id = coordinator.submit_block(block)
    coordinator.assign_block(block_id, "provider")

    result = execute_block(block, DeterministicStubRunner())
    forged = BlockResult(
        block_id=result.block_id,
        provider_id=result.provider_id,
        merkle_root="00" * 32,
        records=result.records,
    )
    with pytest.raises(ProtocolError):
        coordinator.submit_result(forged)


def test_unassigned_provider_cannot_submit_result():
    coordinator, ledger = make_coordinator()
    ledger.deposit("intruder", 5_000)
    coordinator.register_provider("intruder", stake=2_000, supported_models={MODEL})
    block = make_block()
    block_id = coordinator.submit_block(block)
    coordinator.assign_block(block_id, "provider")

    result = execute_block(block, DeterministicStubRunner())
    forged = BlockResult(
        block_id=result.block_id,
        provider_id="intruder",
        merkle_root=result.merkle_root,
        records=result.records,
    )
    with pytest.raises(ProtocolError):
        coordinator.submit_result(forged)


def test_provider_must_meet_min_stake_and_serve_model():
    coordinator, ledger = make_coordinator()
    ledger.deposit("small", 500)
    with pytest.raises(ProtocolError):
        coordinator.register_provider("small", stake=500, supported_models={MODEL})

    block = ComputeBlock(
        model_id="other-model",
        requests=(InferenceRequest(request_id="r", prompt="p"),),
        client_id="client",
        price=10,
    )
    block_id = coordinator.submit_block(block)
    with pytest.raises(ProtocolError):
        coordinator.assign_block(block_id, "provider")


def test_detection_probability_math():
    assert detection_probability(0.0, 8) == 0.0
    assert detection_probability(1.0, 1) == 1.0
    # Cheating on 10% of requests with 16 samples is caught ~81% of the time.
    assert detection_probability(0.10, 16) == pytest.approx(0.8147, abs=1e-3)


def test_settle_requires_completed_audit():
    coordinator, _ = make_coordinator()
    block_id = coordinator.submit_block(make_block())
    with pytest.raises(ProtocolError):
        coordinator.settle(block_id)
