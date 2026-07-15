# Native Chat Tool Orchestration Plan

## Goal

Add a small Python-native model/tool contract and bounded chat orchestration loop inspired by Pi's core semantics, then expose live tool activity through the existing chat SSE and React interfaces. Add deterministic routing coverage before enabling a default tool catalog.

## Scope

- Replace the regex-only chat tool shortcut with model-selected tool calls.
- Keep model invocation provider-specific while making tool definitions, execution, persistence, events, and policy common to online and local agents.
- Register safe default tools for current web search, user-scoped uploaded-file search, and image generation.
- Audit every existing adapter and expose only explicit, production-appropriate wrappers by default.
- Preserve compatibility with existing `AgentCompletion`, chat history, SSE routes, and React chat rendering.
- Add a frontend contract for live tool lifecycle events and artifacts.
- Add deterministic pytest routing coverage without executing real side effects.

## Design

1. Add normalized model/tool contracts.
   - `ToolDefinition` contains a stable public name, description, Pydantic argument model, execution policy, and handler.
   - `ToolCall`, `ToolResult`, `ModelTurn`, and `ModelEvent` normalize provider output.
   - Tool implementations remain server-side; the model sees only authorized names, descriptions, and JSON schemas.

2. Add an explicit tool registry and execution boundary.
   - Build the registry from reviewed wrappers rather than reflecting every adapter method.
   - Validate arguments before dispatch.
   - Carry trusted context such as `user_id` separately from model-provided arguments.
   - Enforce timeouts, result-size limits, side-effect metadata, and approval requirements.

3. Add a bounded chat orchestrator.
   - Load the ordered conversation.
   - Call the selected model backend with messages and allowed tool schemas.
   - Execute validated tool calls, append results, and continue until a final response or configured limit.
   - Persist the completed turn once and emit normalized text/tool/artifact/final/error events.

4. Integrate model backends.
   - Online models use provider-native tool definitions and normalize both non-streaming and streamed tool calls.
   - Local runners expose a capability-aware model-step interface.
   - Native local tool templates may be added when supported; unsupported local models must fail closed rather than using unrestricted reflection.
   - Existing legacy task/tick execution remains separate from chat.

5. Define the default tool catalog.
   - `web.search`: structured, bounded, current public web search.
   - `documents.search`: user-scoped uploaded-file metadata/content search.
   - `image.generate`: existing OpenAI-compatible image generation adapter.
   - Review all other adapters individually. Read-only, well-bounded adapters may receive explicit wrappers; communications, filesystem writes, logs, transcription-only adapters, and stubs are excluded from the default chat registry unless their safety and product contract are complete.

6. Update chat persistence and HTTP/SSE contracts.
   - Preserve stable tool call IDs, arguments, status, summaries, errors, and artifact references.
   - Emit tool proposed/started/completed/failed events plus text deltas and a final compatibility envelope.
   - Fix existing cache, new-chat streaming, and duplicate-completion correctness issues encountered at the integration seam.

7. Update the React chat contract.
   - Track live tool calls by ID.
   - Render queued/running/succeeded/failed tool states and artifacts while the assistant is streaming.
   - Reconcile live state with the final envelope and persisted history.
   - Keep the contract ready for later approval/cancel events without adding irreversible controls in this slice.

8. Add evaluation coverage.
   - Fast pytest tests use fake model backends and fake tool handlers to verify loop, validation, persistence, events, failures, and limits.
   - A framework-neutral routing dataset covers expected tools, forbidden tools, and argument predicates for document search, current news, image generation, and no-tool contrasts.
   - Exact recorded-call checks cover tool names and arguments while project assertions reject forbidden calls.
   - Live-model evaluations are explicitly marked and do not execute real web, file mutation, messaging, or other side effects.

## Adapter Review Policy

Each adapter is classified as one of:

- **Default:** safe, useful, bounded, and sufficiently implemented for ordinary chat.
- **Opt-in:** implemented but side-effecting, credential-sensitive, or context-specific; requires explicit policy/approval before exposure.
- **Excluded:** stub, duplicate transport, unsafe generic primitive, or not meaningful as a chat tool.

The final PR documents the classification and only registers default tools automatically.

## Verification

1. Run targeted Python compilation, Ruff, type checks where practical, and focused pytest suites.
2. Run the deterministic routing tests without an external judge or evaluation dependency.
3. Run frontend unit tests and production build.
4. Start `db`, `backend`, and `frontend` with Docker; inspect logs and verify the frontend with `curl`.
5. Exercise chat and live tool rendering in the browser using stubbed or locally available safe tools.
6. Run native `make run MLX_BACKEND=1` because the LocalAgent contract changes, or report exact environment/model blockers.
7. Inspect the final git diff, stage only feature files, commit, push, and open a PR with the validation evidence.
