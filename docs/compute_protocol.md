# Geist Compute Protocol: Verifiable LLM Inference with Sampling Audits

The `protocol/` package implements a marketplace-style protocol where clients
submit **blocks of LLM inference compute**, providers execute them, and
**sampling audits** keep providers honest without re-running every request.
It is a self-contained, dependency-free reference implementation: the ledger
is in-memory and the coordinator is a single process, but the message flow and
cryptographic commitments are the same ones a decentralized deployment would
put on a chain or settlement service.

## Roles

| Role        | Responsibility |
|-------------|----------------|
| Client      | Submits a `ComputeBlock` (ordered batch of `InferenceRequest`s against one model) and escrows payment. |
| Provider    | Stakes collateral, executes assigned blocks, submits an `InferenceRecord` per request plus a Merkle-root commitment. |
| Auditor     | Re-executes an unpredictable random sample of requests on a trusted reference runner and issues a verdict. |
| Coordinator | Tracks block lifecycle, enforces stake/assignment rules, and settles escrow based on audit verdicts. |

## Protocol flow

```mermaid
sequenceDiagram
    participant C as Client
    participant K as Coordinator
    participant P as Provider
    participant A as Auditor
    C->>K: submit_block(block) — payment locked in escrow
    K->>P: assign_block (staked, serves model)
    P->>P: run every request deterministically
    P->>K: submit_result(records, merkle_root)
    Note over K,A: beacon nonce revealed AFTER commitment
    A->>A: sample k indices = HMAC(nonce, block_id)
    A->>A: re-execute sampled requests, verify Merkle inclusion + outputs
    A->>K: AuditReport(PASS | FAIL)
    K->>P: PASS → escrow released (minus protocol fee), reputation up
    K->>C: FAIL → refund + slashed stake, provider reputation down
```

1. **Submission.** `Coordinator.submit_block` locks `block.price` from the
   client in escrow. The block id is the canonical hash of the model id,
   client, price, and every request — the work is content-addressed.
2. **Assignment.** Only registered providers with stake ≥ `min_stake` that
   serve the block's model can be assigned.
3. **Commitment.** The provider returns one record per request and a Merkle
   root over the record hashes. The coordinator recomputes the root before
   accepting, so the provider is bound to exactly these outputs.
4. **Audit.** A beacon nonce — revealed only after commitment — seeds an
   HMAC-SHA256 draw (`protocol/sampling.py`) that picks `k` request indices.
   Because the nonce is unknown at execution time, the provider cannot know
   which requests will be checked; because the draw is deterministic, anyone
   can verify the auditor sampled fairly. For each sampled index the auditor
   checks Merkle inclusion and re-executes the request on a reference runner,
   requiring exact output-text equality and per-token logprobs within
   `logprob_tolerance`.
5. **Settlement.** PASS releases escrow to the provider minus a protocol fee;
   FAIL refunds the client and slashes `slash_fraction` of the provider's
   stake to them.

## Why sampling works (and its limits)

A provider that corrupts a fraction *f* of a block's requests evades one audit
of sample size *k* with probability (1 − *f*)^*k*:

| cheat fraction | k = 8 | k = 16 | k = 32 |
|---------------:|------:|-------:|-------:|
| 5%             | 34%   | 56%    | 81%    |
| 10%            | 57%   | 81%    | 97%    |
| 25%            | 90%   | 99%    | ~100%  |

(`detection_probability` in `protocol/auditor.py`.) Combined with slashing,
cheating is negative-expected-value whenever
`slash × P(detect) > price × f`, so the economics — stake size, sample size,
and slash fraction — are what make small-scale cheating irrational, not the
audit alone. `test_cheating_outside_the_sample_evades_this_audit` documents
the residual per-audit risk explicitly.

## Determinism

Sampling audits only work if re-execution reproduces the original output.
The protocol therefore requires temperature-0 (or fixed-seed) decoding, and
`AuditPolicy.logprob_tolerance` absorbs benign numeric noise across hardware.
Real GPU inference is not always bit-exact across devices or batch sizes;
production deployments handle this by pinning runtime/hardware classes per
model, comparing logprobs within a tolerance instead of exact token ids, or
using activation commitments (see TOPLOC below). `GeistAgentRunner`
(`protocol/geist_runner.py`) adapts any Geist `BaseAgent` as a runner, so
audits can run against the local MLX/vLLM stack or an online provider.

## Does this already exist?

Yes — this design space is active, and several systems combine "rent out
blocks of compute" with probabilistic verification for ML/LLM workloads:

- **Gensyn** — decentralized ML compute protocol whose verification
  (probabilistic proof-of-learning, graph-based pinpointing, Truebit-style
  incentive games) is the closest analogue for training workloads.
- **Hyperbolic's Proof of Sampling (PoSP, 2024)** — spot-checks a fraction of
  inference results with validators and slashes discrepancies; essentially
  the same sampling-audit economics implemented here.
- **Prime Intellect's TOPLOC (2025)** — verifiable inference via
  locality-sensitive hashes of intermediate activations, letting a verifier
  cheaply confirm which model actually produced an output — a stronger
  commitment than output hashing that tolerates GPU nondeterminism.
- **opML / ORA** — optimistic ML: results are accepted unless challenged
  within a dispute window, with fraud proofs bisecting the computation.
- **zkML (EZKL, Modulus Labs)** — full cryptographic proofs of inference;
  strongest guarantees, but orders of magnitude too slow for LLM-scale
  workloads today, which is exactly why sampling/optimistic schemes exist.
- **Bittensor** — incentivized inference network where validators score miner
  outputs (quality-weighted rather than correctness-audited).
- **Akash, Golem, io.net, Petals** — compute/GPU marketplaces and volunteer
  inference swarms; they rent the compute but largely trust the operator,
  i.e., the market without the audit layer.

What this package contributes to Geist is a small, testable reference of the
audit-marketplace core — content-addressed blocks, post-commitment beacon
sampling, Merkle-bound results, stake/escrow/slash settlement — that plugs
into Geist's own agents as runners.

## Usage

```python
from protocol import (
    AuditPolicy, Auditor, ComputeBlock, Coordinator, DeterministicStubRunner,
    InferenceRequest, Ledger, ProtocolConfig,
)

ledger = Ledger()
coordinator = Coordinator(
    ledger=ledger,
    auditor=Auditor(DeterministicStubRunner(), AuditPolicy(sample_size=16)),
    config=ProtocolConfig(min_stake=1_000, slash_fraction=0.5),
)

ledger.deposit("alice", 10_000)
ledger.deposit("gpu-farm", 5_000)
coordinator.register_provider("gpu-farm", stake=2_000, supported_models={"llama-3.1-8b"})

block = ComputeBlock(
    model_id="llama-3.1-8b",
    client_id="alice",
    price=500,
    requests=tuple(
        InferenceRequest(request_id=f"req-{i}", prompt=p, temperature=0.0)
        for i, p in enumerate(["Summarize ...", "Translate ..."])
    ),
)
block_id = coordinator.submit_block(block)
coordinator.assign_block(block_id, "gpu-farm")
# provider executes, builds records + Merkle root, then:
# coordinator.submit_result(result)
# coordinator.run_audit(block_id, beacon_nonce)
# coordinator.settle(block_id)
```

Run the test suite with `pytest tests/protocol`.
