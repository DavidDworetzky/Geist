# Plan: Local generation metrics

## Goal

Expose consistent, per-request token and timing metrics for Geist's local MLX,
Transformers (including the legacy `vllm`/Qwen pipeline shim), and managed
llama.cpp inference paths without changing online provider behavior or
persisting prompt contents.

## Contract

- Add a shared `GenerationStats` value containing backend/model identity,
  prompt and completion token counts, prompt/decode durations and throughput,
  end-to-end generation duration, time to first token, and peak memory when the
  backend can provide them.
- Attach metrics to the completed model turn so concurrent requests cannot
  overwrite one another through mutable runner-level state.
- Aggregate local model rounds in the chat orchestrator and expose them on the
  final completion response. Tool-using chats may contain more than one model
  round, so the response retains a list rather than pretending all rounds share
  one timing sample.
- Keep all fields optional except identifying/count fields that a backend can
  establish reliably. Do not log or return prompts as part of instrumentation.

## Implementation

1. Define `GenerationStats` in the shared model contract and add optional
   metrics to `ModelTurn` and `AgentCompletion`.
2. Adapt both MLX implementations' existing `last_stats` dictionaries into the
   shared contract at turn completion.
3. Instrument Transformers generation with exact input/output token counts,
   monotonic wall timing, and CUDA/MPS synchronization around asynchronous
   accelerator work. Label this as end-to-end generation throughput because
   `transformers.generate()` combines prefill and decode.
4. Parse llama.cpp's native `usage` and `timings` response objects for
   non-streaming and streaming requests. Prefer its native prompt and predicted
   throughput over client-side estimates.
5. Instrument the backward-compatible `vllm`/Qwen Transformers pipeline shim
   with tokenized prompt/output counts and synchronized end-to-end throughput.
6. Carry per-round metrics through the chat orchestrator's final SSE and JSON
   response, update client response types, and preserve compatibility for
   online and uninstrumented runners by defaulting to an empty list.
7. Add focused tests for MLX adaptation, Transformers timing/counting,
   llama.cpp native timing parsing, and chat-response propagation.

## Validation

- Run focused agent, runner, route, and orchestrator tests.
- Run lint/type checks for changed Python and TypeScript sources where the
  repository tooling is available.
- Run the repository Docker startup/log/curl smoke loop.
- Attempt the native `make run MLX_BACKEND=1` smoke loop because the shared
  local-agent and MLX turn path changes; report missing model/runtime resources
  as an explicit blocker.

## Acceptance criteria

- MLX, Transformers, and llama.cpp local turns expose generation metrics in the
  final completion response.
- llama.cpp uses server-native prompt/decode timings.
- Transformers reports exact token counts and explicitly end-to-end throughput,
  without claiming separate decode throughput.
- Concurrent model turns do not depend on a mutable global or runner-level
  `last_stats` lookup after completion.
- Existing online completions and clients that ignore metrics remain compatible.
