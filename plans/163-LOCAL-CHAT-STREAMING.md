# Local Chat Streaming

## Goal

Make local-model responses appear incrementally in the existing chat SSE flow instead of
arriving as one buffered delta, while preserving non-streaming compatibility and chat-history
persistence.

## Implementation

1. Make raw incremental generation the required `BaseRunner` contract. The public streaming
   adapter applies shared stop/response-edge semantics, and `generate()`, `complete()`, and
   `complete_messages()` collect that same stream for buffered callers.
2. Add the same stream-first model-turn interface to `BaseAgent` and the structural chat backend
   protocol. `complete_model_turn()` is the default buffered collector, while both streaming and
   non-streaming HTTP chat endpoints execute the orchestrator's canonical stream.
3. Implement native incremental generation for every registered local runner:
   - forward `mlx-lm` text chunks directly;
   - expose Unicode-safe incremental decoding from the manual MLX backend;
   - consume llama-server's OpenAI-compatible SSE stream;
   - use `TextIteratorStreamer` for the Transformers and VLLM/Qwen compatibility runners;
   - apply configured stop sequences without leaking delimiters into chat.
4. Update `LocalAgent.stream_model_turn()` and the voice-facing `stream_complete_text()` path to
   forward runner chunks and persist only the finalized turn. Voice's buffered response mode
   collects the same model stream instead of selecting a second inference implementation.
5. Remove legacy route fallbacks that could silently execute a buffered generator. Unsupported
   runner implementations now fail at construction because the stream method is abstract.
6. Add focused runner, agent, API-stream, and frontend regression coverage proving multiple
   local deltas are preserved through the existing SSE and React chat pipeline.

## Validation

- Run focused backend inference/route tests and frontend chat-hook/component tests.
- Run formatting, lint, and typing checks for changed files.
- Run Docker startup/log/curl and browser chat smoke checks.
- Run the repository QA workflow against the installed `Qwen/Qwen3.8-27B` model with
  `GEIST_LOCAL_RUNNER=mlx_llama make run MLX_BACKEND=1`, recording UI and runtime evidence;
  otherwise report the exact environmental blocker.
