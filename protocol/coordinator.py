"""Protocol coordinator: block lifecycle, assignment, audits, and settlement.

Lifecycle: SUBMITTED -> ASSIGNED -> COMMITTED -> PASSED/FAILED -> SETTLED.

On submission the client's payment is locked in escrow. A registered, staked
provider is assigned and executes the block, committing to its outputs with a
Merkle root. After commitment, a beacon nonce is revealed and the auditor
re-executes the sampled requests. A passing audit releases escrow to the
provider; a failing audit slashes the provider's stake and refunds the client.
"""

from dataclasses import dataclass, field

from protocol.auditor import Auditor
from protocol.hashing import MerkleTree
from protocol.ledger import Ledger
from protocol.models import (
    AuditReport,
    AuditVerdict,
    BlockResult,
    BlockStatus,
    ComputeBlock,
    Provider,
)


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class ProtocolConfig:
    min_stake: int = 1_000
    slash_fraction: float = 0.5
    protocol_fee_fraction: float = 0.02
    fee_account: str = "protocol_treasury"


@dataclass
class BlockState:
    block: ComputeBlock
    status: BlockStatus
    provider_id: str | None = None
    result: BlockResult | None = None
    report: AuditReport | None = None


@dataclass
class Coordinator:
    ledger: Ledger
    auditor: Auditor
    config: ProtocolConfig = field(default_factory=ProtocolConfig)
    providers: dict[str, Provider] = field(default_factory=dict)
    blocks: dict[str, BlockState] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Registration and submission
    # ------------------------------------------------------------------

    def register_provider(self, provider_id: str, stake: int, supported_models: set[str]) -> Provider:
        if stake < self.config.min_stake:
            raise ProtocolError(
                f"stake {stake} below protocol minimum {self.config.min_stake}"
            )
        self.ledger.add_stake(provider_id, stake)
        provider = Provider(provider_id=provider_id, supported_models=set(supported_models))
        self.providers[provider_id] = provider
        return provider

    def submit_block(self, block: ComputeBlock) -> str:
        block_id = block.block_id
        if block_id in self.blocks:
            raise ProtocolError(f"block {block_id} already submitted")
        self.ledger.lock_escrow(block_id, block.client_id, block.price)
        self.blocks[block_id] = BlockState(block=block, status=BlockStatus.SUBMITTED)
        return block_id

    def assign_block(self, block_id: str, provider_id: str) -> None:
        state = self._state(block_id, expected=BlockStatus.SUBMITTED)
        provider = self.providers.get(provider_id)
        if provider is None:
            raise ProtocolError(f"unknown provider {provider_id}")
        if state.block.model_id not in provider.supported_models:
            raise ProtocolError(
                f"provider {provider_id} does not serve model {state.block.model_id}"
            )
        if self.ledger.account(provider_id).stake < self.config.min_stake:
            raise ProtocolError(f"provider {provider_id} stake fell below minimum")
        state.provider_id = provider_id
        state.status = BlockStatus.ASSIGNED

    # ------------------------------------------------------------------
    # Commitment and audit
    # ------------------------------------------------------------------

    def submit_result(self, result: BlockResult) -> None:
        state = self._state(result.block_id, expected=BlockStatus.ASSIGNED)
        if result.provider_id != state.provider_id:
            raise ProtocolError("result submitted by a provider not assigned to this block")
        if len(result.records) != len(state.block.requests):
            raise ProtocolError("result must contain one record per request")
        recomputed = MerkleTree([record.record_hash for record in result.records]).root
        if recomputed != result.merkle_root:
            raise ProtocolError("merkle root does not match submitted records")
        state.result = result
        state.status = BlockStatus.COMMITTED

    def run_audit(self, block_id: str, beacon_nonce: str) -> AuditReport:
        state = self._state(block_id, expected=BlockStatus.COMMITTED)
        report = self.auditor.audit(state.block, state.result, beacon_nonce)
        state.report = report
        state.status = (
            BlockStatus.PASSED if report.verdict == AuditVerdict.PASS else BlockStatus.FAILED
        )
        return report

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, block_id: str) -> None:
        state = self.blocks.get(block_id)
        if state is None:
            raise ProtocolError(f"unknown block {block_id}")
        if state.status not in (BlockStatus.PASSED, BlockStatus.FAILED):
            raise ProtocolError(f"block {block_id} is not audited yet ({state.status.value})")
        provider = self.providers[state.provider_id]
        if state.status == BlockStatus.PASSED:
            fee = int(state.block.price * self.config.protocol_fee_fraction)
            self.ledger.release_escrow_to(block_id, state.provider_id)
            if fee > 0:
                self.ledger.account(state.provider_id).balance -= fee
                self.ledger.account(self.config.fee_account).balance += fee
            provider.blocks_completed += 1
            provider.reputation = min(1.0, provider.reputation + 0.01)
        else:
            self.ledger.refund_escrow(block_id)
            self.ledger.slash(
                state.provider_id, self.config.slash_fraction, state.block.client_id
            )
            provider.audits_failed += 1
            provider.reputation = max(0.0, provider.reputation - 0.25)
        state.status = BlockStatus.SETTLED

    def _state(self, block_id: str, expected: BlockStatus) -> BlockState:
        state = self.blocks.get(block_id)
        if state is None:
            raise ProtocolError(f"unknown block {block_id}")
        if state.status != expected:
            raise ProtocolError(
                f"block {block_id} is {state.status.value}, expected {expected.value}"
            )
        return state
