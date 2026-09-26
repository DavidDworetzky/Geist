# Preserve OpenRouter reasoning across tool calls

The Opus 5.5 catalog addition uses the shared OpenAI-compatible streaming
backend. That backend currently discards `delta.reasoning_details`, leaving
tool continuations without the provider state required by mandatory thinking.

## Contract change

- Add optional reasoning details and the originating model ID and endpoint to `ModelTurn`
  and `ChatMessage`. Existing backends and historical transcripts default to
  no reasoning state.
- Collect the provider's reasoning chunks in their original order, including
  empty text, signatures, encrypted data, and metadata. Do not render them as
  text deltas or modify their content.
- Carry them through orchestration, saved transcripts, and goal checkpoints.
  Replay them only to the originating model and endpoint on the OpenAI-compatible path.
  Ordinary history loading omits prior-turn reasoning while retaining visible
  answers and tool results; active goal resume uses the intact checkpoint.
- Keep existing tool invocation IDs and approval isolation unchanged.
- Count this state against existing context limits and discard whole protocol
  blocks rather than partially truncating reasoning details.
  Keep a pending signed tool turn intact even if it exceeds the soft retention
  budget so checkpointing and resume cannot lose executed results. The provider
  still enforces its token limit. For completed plain answers, discard oversized
  reasoning and retain bounded visible text. Completed tool blocks can shed
  reasoning under budget pressure while retaining their call/result pairs.

## Validation

Exercise a streamed Opus tool request, execution, continuation, persisted
history reload, and checkpoint round trip. Cover reasoning-only chunks,
multiple chunks at the same index, encrypted and signed blocks, model switches,
legacy messages, context budgets, retries, and the absence of reasoning in
visible output. Run focused agent, contract, and orchestrator tests plus Ruff
and mypy. Attempt the repository's Docker smoke checks when available; report
environment limits without installing packages or making paid calls implicitly.

Provider reference:
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
