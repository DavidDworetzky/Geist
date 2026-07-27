# Multiplayer Prompting

## Goal

Let multiple human users participate in one chat session: every message is
attributed to a real user, all participants see who is in the room and who is
typing in real time over WebSockets, and everything degrades gracefully to
HTTP polling when a WebSocket cannot be established. Delivered as five stacked
PRs so each layer lands reviewable and independently testable.

## Background and current constraints

- History is stored as a JSON string of turn pairs on
  `chat_session.chat_history` (`app/models/database/chat_session.py`). Each
  entry is `{"user": str, "ai": str, "tool_calls": ..., "artifacts": ...,
  "run_id": ..., "status": ...}` — one anonymous user string per turn, no
  author identity. There is no per-message table.
- Identity is a stub: every request resolves to `get_default_user()`
  (`app/models/database/geist_user.py:37`), a hardcoded user. There are no
  login endpoints, tokens, or sessions. `geist_user` has `user_id, username,
  name, email, password` but nothing consumes `password`.
- The LLM context is built by `ChatOrchestrator._entry_messages` /
  `_history_messages` (`app/services/chat_orchestrator.py`), which map each
  entry to `ChatMessage(role="user"|"assistant")`. `ChatMessage`
  (`agents/models/tool_calling.py`) already carries an optional `name` field
  we can use for speaker identity.
- Exactly one WebSocket exists (`/api/v1/voice/stream`,
  `app/api/v1/endpoints/voice.py:118`) and it is strictly per-connection.
  There is no connection registry or broadcast primitive. Chat token
  streaming uses SSE (`/agent/complete_text_stream/...`), parsed manually in
  `client/geist/src/Hooks/useCompleteText.tsx`.
- `app/loopback_security.py` already validates `scope["type"] == "websocket"`
  and closes violations with code 1008, so a new chat WebSocket inherits the
  same origin/host protections.

## Design

### 1. Representing users in a multi-turn conversation

**Membership** becomes explicit via a new `chat_participant` table:

| column            | type      | notes                                    |
|-------------------|-----------|------------------------------------------|
| chat_session_id   | int FK    | composite PK with `user_id`              |
| user_id           | int FK    | → `geist_user.user_id`                   |
| role              | str       | `owner` \| `member`                      |
| joined_at         | datetime  |                                          |
| last_seen_at      | datetime  | updated by presence traffic              |

`chat_session.user_id` remains the owner; creating a session inserts the
owner as an `owner` participant. A backfill migration inserts owner rows for
existing sessions.

**Attribution** stays inside the existing JSON turn-pair format (no message
table rewrite) by adding an `author` object to each entry:

```json
{
  "user": "can you compare these two approaches?",
  "author": {"user_id": 2, "username": "alice", "display_name": "Alice"},
  "ai": "...",
  "run_id": "...",
  "status": "completed"
}
```

Entries without `author` are legacy and render as the session owner. The
snapshot (`user_id`, `username`, `display_name`) is denormalized on purpose:
history stays render-complete even if a user is later renamed, and readers
never join against `geist_user`.

**Exposure to the model.** The orchestrator sets
`ChatMessage(role="user", name=<username>)` for every user turn. Backends
whose chat template supports `name` use it natively; for all others the
orchestrator falls back to prefixing the content with `<<username>> ` — but
only when the session actually has more than one distinct author, so
single-player prompts are byte-identical to today and existing orchestrator
tests keep passing.

### 2. Exposing user identities

Full auth is out of scope; multiplayer needs *identity*, not *security*
(the deployment remains loopback/LAN-trusted, guarded by
`loopback_security`). The identity layer is deliberately swappable for real
auth later:

- `resolve_request_user(request) -> UserModel`: reads a signed `geist_uid`
  cookie (itsdangerous-style HMAC with a server secret, so a casual client
  cannot impersonate by editing an integer); falls back to
  `get_default_user()` when absent, preserving all current behavior. All
  `get_default_user()` call sites in `app/main.py` move to this resolver.
- `GET /api/v1/users/me`, `GET /api/v1/users`, `POST /api/v1/users`
  (username + display name), `POST /api/v1/users/select` (sets the cookie).
  `password` is never returned; a `UserPublic` schema (`user_id`, `username`,
  `display_name`) is the only shape that crosses the API.
- Frontend: a lightweight user picker (create/choose profile) stored via the
  cookie; the active user's identity is rendered in the header and on every
  authored message (name + deterministic color/initials avatar derived from
  `user_id`).
- Participant management: `POST /api/v1/chat/{chat_id}/participants` and
  `GET  /api/v1/chat/{chat_id}/participants`. Sending a message to a session
  auto-joins the sender as a `member`.

### 3. Realtime channel: WebSocket presence and typing

New endpoint `WS /api/v1/chat/{chat_id}/ws?user_id=...` (cookie preferred,
query param for parity with the voice socket). A new
`app/services/chat_realtime.py` owns a `ChatConnectionManager`:

- in-process registry `dict[chat_id, set[Connection]]` guarded by an
  `asyncio.Lock`; `broadcast(chat_id, event, exclude=sender)` fans out with
  per-socket error isolation (a dead socket is evicted, never breaks the
  loop). In-process is sufficient because Geist runs as a single uvicorn
  process; the manager is the one seam to swap for Redis pub/sub if that
  ever changes.

**Protocol** — JSON text frames, `{"type": ..., "v": 1, ...}`:

| direction | type             | payload                                        |
|-----------|------------------|------------------------------------------------|
| C → S     | `typing_start`   | none (server stamps user + chat)               |
| C → S     | `typing_stop`    | none                                           |
| C → S     | `ping`           | none (keepalive)                               |
| S → C     | `hello`          | `participants`, `active_typers`, `event_cursor`|
| S → C     | `presence`       | `{user, status: "joined"|"left"}`              |
| S → C     | `typing`         | `{user, is_typing: bool}`                      |
| S → C     | `message_committed` | `{entry_index, author, run_id}`             |
| S → C     | `turn_started` / `turn_completed` | `{author, run_id, status}`    |
| S → C     | `pong`           | none                                           |

Semantics:

- Typing state is server-authoritative with a TTL: `typing_start` arms a
  ~5 s expiry that the client refreshes every ~3 s while keystrokes continue;
  expiry or disconnect broadcasts `typing: false`. Clients therefore never
  see a stuck indicator after a crash.
- Token streaming **stays on the existing SSE endpoints**. The socket carries
  coordination events only; `message_committed` tells non-authoring clients
  to fetch the new turn via the existing paginated history endpoint. This
  keeps the streaming pipeline untouched and makes the fallback story simple.
- The completion routes call into the manager (via FastAPI dependency) to
  emit `turn_started` / `message_committed` / `turn_completed`, so realtime
  events flow even for participants who submitted nothing.

**Concurrent prompting.** A per-chat `asyncio.Lock` in the orchestrator path
serializes runs. While a run is active, a second `complete_text*` request
returns `409 {"active_run_id", "author"}`; clients disable submit and show
"Alice is prompting…" from `turn_started`. Queueing (rather than rejecting)
is explicitly deferred — 409 + UI lockout is predictable and matches the
existing single-run-per-chat assumption in run cancellation.

### 4. Graceful fallback when WebSockets are unavailable

Server side, every broadcast also appends to a per-chat in-memory ring
buffer (last ~200 events, monotonically increasing `event_cursor`), exposed
as:

- `GET /api/v1/chat/{chat_id}/events?cursor=N&wait=25` — long-poll: returns
  immediately with any events `> N`, otherwise parks up to `wait` seconds on
  an `asyncio.Event` and returns `{events: [], cursor: N}` on timeout.
- `POST /api/v1/chat/{chat_id}/typing {"is_typing": bool}` — heartbeat
  equivalent of `typing_start`/`stop`, feeding the same TTL logic, so socket
  and polling clients see each other's typing symmetrically.

Client side, `useChatPresence(chatId)` wraps a transport abstraction:

1. **WebSocketTransport** (preferred): connects with `wss:`/`ws:` selection
   and reconnect via exponential backoff + jitter (1 s → 30 s cap),
   mirroring `useVoiceChat.tsx`. Presence state is rebuilt from `hello` on
   each reconnect, and the `event_cursor` in `hello` lets the hook fetch
   anything missed during the gap from the events endpoint.
2. **PollingTransport** (fallback): activated when the socket fails to open
   twice consecutively or the environment lacks WebSocket support; drives the
   long-poll loop and typing heartbeats. It periodically retries the socket
   (every ~60 s) and promotes back when one succeeds.
3. **Degraded mode**: if even long-poll errors persist, the hook drops to
   silent history refetch on a 10 s interval — no presence/typing, but
   messages from other users still appear. The UI shows a subtle
   "realtime unavailable" state rather than failing.

The consuming UI (`Chat.tsx`) is transport-agnostic: it sees one event
stream plus a `connectionQuality: "realtime" | "polling" | "degraded"`.

## Stacked PR plan

Each PR builds on the previous branch and is releasable on its own; nothing
later is required for earlier layers to function.

1. **PR 1 — Identity and participants (schema + resolver).**
   Alembic migration for `chat_participant` + owner backfill; `author`
   accepted and persisted by `update_chat_history`; `resolve_request_user`
   with signed cookie and default-user fallback; `UserPublic` schema and
   `/api/v1/users*` + participant endpoints. No behavior change for existing
   clients (default user everywhere). Tests: migration/backfill, resolver
   cookie/fallback paths, participant auto-join, `password` never serialized.

2. **PR 2 — Attribution through the orchestrator and UI.**
   Completion routes stamp the resolved user into the persisted entry;
   `_entry_messages` emits `ChatMessage.name` and the multi-author-only
   `<<username>>` prefix; history endpoints return `author`; frontend
   `ChatPair` gains `author`, messages render name/avatar, and the user
   picker ships. Tests: orchestrator prompt-shape tests (single-author
   unchanged, multi-author prefixed), route attribution tests, Jest
   rendering tests for authored/legacy entries.

3. **PR 3 — WebSocket presence and typing.**
   `ChatConnectionManager`, `WS /api/v1/chat/{chat_id}/ws`, protocol above,
   typing TTL, turn/message lifecycle emission from completion routes,
   per-chat run lock + 409. Frontend `useChatPresence` (socket-only) +
   typing indicator and prompting-lockout UI. Tests: manager broadcast/evict
   unit tests, `TestClient` websocket tests (join/typing/TTL/two-client
   broadcast), loopback-security websocket rejection, 409 route test.

4. **PR 4 — Polling fallback and reconnect hardening.**
   Event ring buffer + cursor wired under the manager, long-poll events
   endpoint, typing heartbeat endpoint, `PollingTransport` + transport
   promotion/demotion + degraded mode + cursor-based gap replay. Tests:
   buffer overflow/cursor semantics, long-poll wake and timeout, symmetric
   typing between socket and polling clients, Jest transport-fallback tests
   with a failing mock WebSocket.

5. **PR 5 — Multiplayer E2E and docs.**
   Playwright multi-context scenarios (two browser contexts, distinct users:
   both see attribution, typing indicators, `message_committed` refresh,
   409 lockout; repeat with WebSocket blocked to prove the polling path);
   docs page describing the protocol and identity model; README note that
   this is trusted-network identity, not authentication.

## Validation

- Ruff, formatting, and mypy for changed Python; alembic upgrade/downgrade
  cycle against SQLite and Postgres.
- Focused pytest suites per PR (`tests/api`, `tests/services`,
  `tests/database`), then full backend suite.
- Jest for hooks/components; frontend production build.
- Playwright multiplayer suite (PR 5) in both realtime and forced-polling
  modes.
- Manual: `make run`, two browsers with distinct profiles against one
  session — verify attribution in the prompt sent to a local runner
  (transcript inspection), typing indicators, socket kill → polling
  continuity → socket recovery.
