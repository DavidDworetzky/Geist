# Local Streaming Review Fixes

## Goal

Make the canonical local streaming contract behave consistently across llama.cpp, manual MLX,
Transformers, and the Qwen/vLLM compatibility runner, including cancellation and structured
conversation history.

## Implementation

1. Separate llama-server's raw provider event stream from its normalized public model-turn stream
   so response-edge whitespace and stop sequences are applied once without a delegation cycle.
2. Replace the manual MLX streamer's word-boundary buffering with Unicode-safe token-granular
   decoding, and finalize generation telemetry even when the consumer closes the stream early.
3. Preserve structured chat history when adapting `ChatMessage` values into runner messages, and
   give `BaseAgent` the protocol's default native-tool capability flag.
4. Bound Transformers-compatible streamer waits, remove the dead VLLM pipeline state, and document
   the required agent stream contract plus the chat-template semantics of `BaseRunner.generate()`.
5. Add focused regression coverage for llama-server normalization, multilingual MLX deltas,
   cancellation telemetry, structured history, and bounded streamer waits.

## Validation

- Run the focused runner, agent-contract, route, and local-model tests.
- Run formatting, lint, and typing checks for changed Python files.
- Exercise native MLX chat with the installed Qwen snapshot when feasible, then rely on the PR's
  cross-platform CI lanes for llama-server and Transformers platform coverage.
