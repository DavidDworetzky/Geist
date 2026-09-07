# Local Tool Protocol Plan

## Base

- Base branch: `claude/agent-plugin-standard-geist-hhd03b` (PR #312)
- Head branch: `codex/local-tool-protocol`

## Goal

Add fail-closed native tool turns to the local Transformers and MLX runtimes,
and reuse the same protocol for the legacy VLLM/Qwen runner aliases.

## Implementation

1. Add a shared chat-template tool protocol for safe tool-name mapping,
   OpenAI-shaped schema/message serialization, tokenizer capability probing,
   and Qwen-style tool-call parsing.
2. Implement structured `stream_model_turn` generation in
   `TransformersRunner` and the `mlx_lm` implementation of `MLXLlamaRunner`.
   Keep the manual MLX implementation explicitly tool-disabled.
3. Integrate the Transformers-backed `VLLMRunner` and inherited `Qwen3Runner`
   with the same protocol rather than creating another parser.
4. Advertise native capability only when both model metadata and the loaded
   tokenizer template accept and render tool definitions.
5. Add focused protocol, runner, and LocalAgent contract tests covering tool
   catalog injection, dotted names, tool-result follow-up messages, malformed
   output, and compatibility aliases.

## Verification

1. Run focused protocol/runner/orchestrator tests, formatting, lint, and type
   checks for changed Python files.
2. Run the Docker build/start/log/curl smoke loop.
3. Run native `make run MLX_BACKEND=1` and exercise a local tool call when
   compatible weights are available; otherwise record the exact blocker.
4. Review the diff, commit, push, and open a stacked PR against PR #312.
