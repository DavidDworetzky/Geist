# Top-Level Tool Routing

## Goal

Prevent unrelated tools from biasing local model turns by classifying each user turn once, then exposing only the relevant subset of the enabled tool catalog.

## Routes

- `Answer`: direct answers plus all read-only retrieval tools, including public web and local workspace retrieval.
- `Sensitive_Answer`: local-only retrieval for documented information, past memories, or legal, personal, and medical topics.
- `Action`: code generation, computer use, state changes, and other tasks; exposes enabled tools except image generation.
- `Image_Generation`: explicit image creation or editing; exposes only image-generation tools.

Malformed classifier output and classifier failures fall back to `Action`. Dynamic MCP and reflected adapter tools default to `Action` because their semantics are not trustworthy enough to classify as retrieval.

## Implementation

1. Add semantic retrieval/action/image tags to tool definitions and tag the built-in catalog.
2. Add a small four-label classifier that uses the active chat backend without tool schemas. The same pass returns one `needs_retrieval` boolean so direct writing and reasoning do not enter tokenizer tool mode.
3. Filter the enabled, available tool list before the assistant generation pass.
4. Add deterministic tests for parsing, tool filtering, and orchestration, plus a native live-model routing smoke pass.
5. Address the related bare-JSON tool-call false positive by requiring a registered tool name on the heuristic unwrapped path.

## Validation

- Focused parser, registry, router, and orchestrator tests.
- Docker backend startup, logs, and HTTP smoke test.
- Native MLX startup and real Qwen routing smoke tests for all four routes.
- Browser chat regression: a moon haiku must answer directly without exposing image generation.
