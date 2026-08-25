# Local Chat Streaming

## Goal

Make local-model responses appear incrementally in the existing chat SSE flow instead of
arriving as one buffered delta, while preserving non-streaming compatibility and chat-history
persistence.

## Implementation

1. Add an optional text-streaming contract to `BaseRunner` whose default implementation emits
   one complete response for runtimes without native streaming.
2. Implement native incremental generation for the MLX runner:
   - forward `mlx-lm` text chunks directly;
   - expose token-by-token decoding from the manual MLX backend;
   - apply configured stop sequences without leaking delimiters into chat.
3. Implement Transformers streaming with `TextIteratorStreamer` and a background generation
   thread, propagating model-generation failures to the request stream.
4. Update `LocalAgent.stream_model_turn()` and the legacy voice-facing
   `stream_complete_text()` path to forward runner chunks and persist only the finalized turn.
5. Add focused runner, agent, API-stream, and frontend regression coverage proving multiple
   local deltas are preserved through the existing SSE and React chat pipeline.

## Validation

- Run focused backend inference/route tests and frontend chat-hook/component tests.
- Run formatting, lint, and typing checks for changed files.
- Run Docker startup/log/curl and browser chat smoke checks.
- Run native `make run MLX_BACKEND=1` and exercise local chat when model weights are available;
  otherwise report the exact environmental blocker.
