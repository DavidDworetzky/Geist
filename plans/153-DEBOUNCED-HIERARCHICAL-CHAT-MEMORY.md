# 153 - Debounced Hierarchical Chat Memory

## Status

Implemented and verified on 2026-07-18.

This plan targets `origin/main` at `f40e329` (2026-07-18), not the older local
`main` ref or the current `codex/geist-1` checkout. Implementation should first
integrate the current `origin/main`, which already contains the durable job queue
and native chat orchestration.

## Goal

Turn completed chat activity into searchable, hierarchical memory without adding
latency to the interactive response path:

1. Persist the authoritative chat turn.
2. Debounce the chat for 20 seconds of inactivity.
3. Queue one durable `chat.memory.digest` job.
4. Incrementally summarize the new transcript.
5. Extract a small number of genuinely durable facts.
6. Vectorize summaries and accepted memory records for semantic search.
7. Route the result according to the chat's effective privacy domain:
   - public, unfiled chat -> thread memory plus eligible user-profile memory;
   - private, unfiled chat -> thread memory only, never user memory;
   - chat in a private folder -> thread memory plus that folder's memory, never
     user memory.

In this plan, "transcription" means processing the canonical text transcript
already persisted by chat. Speech-to-text remains the responsibility of the
existing voice pipeline.

## Existing Foundation on `origin/main`

- `app/services/job_queue.py` provides a durable SQLAlchemy-backed queue with:
  - delayed visibility through `run_after`;
  - handler registration by job `kind`;
  - retries with exponential backoff;
  - one in-process worker thread;
  - SQLite and PostgreSQL support.
- `app/models/database/job.py` persists queue state.
- `app/main.py` starts and stops the worker with FastAPI.
- `ChatOrchestrator` persists one authoritative terminal chat entry and already
  bounds recent history by entries and characters.
- `LocalAgent._complete_raw` and `OnlineAgent._complete_raw` provide stateless
  completion paths that do not recursively append to chat history.
- SQLAlchemy can run on SQLite or PostgreSQL.

The existing queue should be extended, not replaced.

## Implementation Outcome

The shipped basic version keeps the hierarchy deliberately small:

- user profile memory for public, unfiled chats;
- exact-folder memory for chats filed in private folders;
- thread memory for every memory-enabled chat;
- no parent-folder inheritance or cross-folder retrieval.

It uses the existing durable queue, a standard-library 256-dimensional
feature-hash embedder, and conservative extractive summarization/fact admission.
No new Python or frontend package was added. The API is intentionally narrower
than some exploratory endpoints below: chat settings own folder assignment and
privacy transitions, while folder CRUD, profile review, record correction,
record deletion, and exact-scope search cover the v1 UI.

## Important Existing Gaps

These are part of the memory work because leaving them unresolved would make the
privacy boundary untrustworthy:

- Chat history reads are still keyed by `chat_session_id` without consistently
  filtering by `user_id`.
- `chat_session.user_id` is nullable for legacy data.
- Job observability accepts a current user but currently returns jobs without
  filtering them by owner.
- Jobs have no coalescing key, cancellation, lease, or recovery for rows left in
  `running` after a process crash.
- Chat history remains one JSON string, so there are no stable per-message IDs.
- There is no folder model, memory model, embedding service, or vector index.

## Core Design Decisions

### 1. The debounce is server-side and durable

The default idle interval is exactly 20 seconds, configurable as
`GEIST_MEMORY_IDLE_SECONDS`.

The timer begins only after an authoritative terminal chat entry is persisted.
It does not begin when the user submits text because generation or tool execution
may still be running.

Eligible terminal entries:

- `completed`: summarize and consider for durable facts;
- `cancelled` and `failed`: retain in raw history but do not treat as completed
  memory material.

No linguistic "is the user finished?" classifier is used. The deterministic idle
window is easier to reason about, test, and recover after restarts. A later turn
simply creates a later transcript revision and reschedules the job.

### 2. One logical job owns the complete digestion pipeline

Job kind:

```text
chat.memory.digest
```

Minimal payload:

```json
{
  "user_id": 17,
  "chat_session_id": 204,
  "expected_revision": 12,
  "pipeline_version": 1
}
```

The payload must not include transcript content, summaries, embeddings, or
secrets. The handler reloads authorized data from the database.

The same job performs:

1. transcript delta loading;
2. incremental thread summarization;
3. durable-fact extraction;
4. existing-memory lookup and conflict classification;
5. embedding generation;
6. atomic memory persistence;
7. user-profile or folder-summary rollup.

This keeps one transcript boundary and one privacy decision across all outputs.

### 3. Revision checks define completion

Add to `ChatSession`:

```text
memory_revision             integer, default 0, non-null
memory_processed_revision   integer, default 0, non-null
memory_last_activity_at     datetime, nullable
memory_enabled              boolean, default true, non-null
memory_mode                 public | private, default public, non-null
folder_id                   nullable FK -> memory_folder
```

Every authoritative terminal write increments `memory_revision` and sets
`memory_last_activity_at`.

When a job begins it verifies:

- the chat belongs to `user_id`;
- its current revision equals `expected_revision`;
- it has been idle for at least the configured interval;
- its memory mode and folder assignment still match the loaded privacy scope.

The job snapshots the transcript delta in memory and performs inference outside
the database transaction. Before committing, it verifies the revision, memory
mode, and folder assignment again. If any changed, all generated output is
discarded and the latest revision is rescheduled. This intentionally prefers
extra background work over committing a summary of an unfinished conversation.

### 4. Extend the existing queue with coalescing and ownership

Add nullable fields to `Job`:

```text
user_id       FK -> geist_user, indexed
dedupe_key    string, indexed
locked_at     datetime, nullable
```

Add `enqueue_or_reschedule(...)`:

- find a queued job with the same `kind`, `user_id`, and `dedupe_key`;
- replace its payload and move `run_after` to `now + 20 seconds`;
- if the matching job is already running, create one queued follow-up;
- otherwise insert a new row.

Memory dedupe key:

```text
chat-memory:{user_id}:{chat_session_id}
```

Portable partial uniqueness across SQLite and PostgreSQL is awkward, so
coalescing is a best-effort queue optimization, not the correctness mechanism.
Revision checks and database uniqueness make duplicate execution harmless.

Add worker recovery:

- set `locked_at` when claiming;
- on worker startup, requeue `running` jobs whose lease is older than a
  configurable timeout;
- keep handlers idempotent.

All job reads over HTTP must filter by `user_id`. Job results for memory work
contain only IDs, revisions, and counts, never memory contents.

## Privacy and Scope Hierarchy

```text
user
├── public user profile
│   ├── compact profile summary
│   └── active durable facts
├── public threads
│   ├── thread summary
│   └── thread episodes/decisions
└── private domain
    ├── unfiled private threads
    │   └── thread summary
    └── private folders
        └── folder
            ├── compact folder summary
            ├── folder-scoped facts/episodes
            └── private threads
                └── thread summary
```

### Public chat

A chat with `memory_mode = public` and `folder_id = NULL`:

- reads the current user-profile summary;
- retrieves relevant active user memories;
- reads its own thread summary and recent transcript;
- may write thread records;
- may promote only high-confidence, durable user facts to user scope.

### Private unfiled chat

A chat with `memory_mode = private` and `folder_id = NULL`:

- reads only its own thread summary and recent transcript;
- writes only thread-scoped records;
- does not read, search, create, confirm, supersede, or otherwise influence the
  global user profile;
- remains semantically searchable only from inside that chat.

This makes privacy a first-class chat setting instead of requiring a folder.

### Private-folder chat

A chat with `folder_id != NULL` always has an effective private scope, regardless
of stale or malformed client input:

- reads the exact folder's summary and records;
- reads its own thread summary and recent transcript;
- writes thread and exact-folder records;
- does not read the global user profile;
- does not search global user memories;
- does not create, confirm, supersede, or otherwise influence user memories.

The server sets `memory_mode = private` when a chat enters a folder. The database
service layer rejects any attempt to make a filed chat public. This is strict
two-way isolation by default. It avoids subtle leakage where a private chat does
not write global memory but global memory is still injected into the private
context. A later explicit "import from profile" feature can relax this
deliberately.

### Nested folders

`MemoryFolder` may include a nullable `parent_folder_id` so the schema does not
block future hierarchy, but v1 retrieval and summarization use the exact folder
only. Parent/child summaries do not automatically inherit from one another.

### Moving chats

Moving a public chat into a private folder is a privacy-changing operation:

1. set `memory_mode = private`, update the folder assignment, and increment
   `memory_revision`;
2. deactivate user-scope facts whose only provenance is that chat;
3. rebuild the public user-profile summary;
4. queue a private re-digest for the folder;
5. retain the raw chat only inside the destination folder.

If a user fact is also supported by public chats, only the private provenance is
removed and the active fact may remain.

Moving a chat out of a private folder leaves `memory_mode = private`. The user
may explicitly switch the now-unfiled chat to public, but this does not
retroactively promote its old content. Only future public turns are eligible
unless the user explicitly promotes selected records.

Switching an unfiled public chat to private performs the same withdrawal and
profile-rebuild steps as moving it into a folder, except it does not create
folder records. Switching an unfiled private chat to public never promotes its
existing private history; it only makes later completed turns eligible for
global memory.

## Data Model

### `MemoryFolder`

Location: `app/models/database/memory_folder.py`

```text
folder_id          integer PK
user_id            non-null FK -> geist_user
parent_folder_id   nullable self FK
name               string
description        text, nullable
revision           integer, default 0
created_at
updated_at
```

All v1 memory folders are private containers. Privacy is enforced by the
chat-folder assignment service, not accepted as a client-controlled folder
flag. A future shared or organizational folder type should be modeled as a new
scope rather than weakening this invariant.

### `MemoryRecord`

Location: `app/models/database/memory_record.py`

A unified record makes summaries and facts searchable through one interface.

```text
memory_id              integer PK
user_id                non-null FK -> geist_user
scope                   user | folder | thread
folder_id               nullable FK -> memory_folder
chat_session_id         nullable FK -> chat_session
record_type             user_profile_summary | user_fact |
                        folder_summary | folder_fact |
                        thread_summary | thread_episode |
                        decision | task | artifact_reference
content                 text
canonical_key           nullable string
importance              float
confidence              float
sensitivity             normal | sensitive | prohibited
source_from_revision    nullable integer
source_through_revision nullable integer
source_metadata         JSON
active                  boolean, default true
supersedes_memory_id    nullable self FK
content_hash            string
pipeline_version        integer
created_at
updated_at
```

Constraints enforce valid scope combinations:

- `user`: no folder or chat ID;
- `folder`: folder ID required;
- `thread`: chat ID required.

Current summaries are active records. Updating a summary inserts a new record and
deactivates/supersedes the prior one, preserving provenance and making rollback
possible.

Useful uniqueness:

- one digest output per `(chat_session_id, source_through_revision,
  record_type, pipeline_version)`;
- one active current summary per user/folder/thread scope, enforced in service
  logic with locking because portable partial unique indexes are limited.

### `MemoryEvidence`

Location: `app/models/database/memory_evidence.py`

```text
memory_evidence_id  integer PK
memory_id           non-null FK -> memory_record
chat_session_id     non-null FK -> chat_session
turn_index          integer
speaker             user | assistant | tool
evidence_hash       string
created_at
```

Until chat messages are normalized, `turn_index` plus a content hash provides
stable-enough provenance without copying the raw message into the memory table.
A later `chat_message` migration can replace `turn_index` with `message_id`.

### `MemoryEmbedding`

Location: `app/models/database/memory_embedding.py`

```text
memory_id        PK/FK -> memory_record
model_id         string
model_revision   string
dimensions       integer
vector           LargeBinary
content_hash     string
created_at
```

The vector is normalized float32 bytes. The embedding is a rebuildable index;
`MemoryRecord.content` remains the source of truth.

## Summarization and Fact Extraction

### Stateless model boundary

Add `MemoryCompletionProvider` with a single structured-completion contract.
It must:

- use a stateless raw completion path;
- disable tools;
- never write to chat history;
- never hydrate user or folder memories into the extraction prompt;
- use low temperature;
- validate structured JSON before returning.

Do not call `complete_text`, because it persists chat history and would create a
recursive memory loop. Existing `_complete_raw` behavior can be adapted behind
the provider interface.

Private-folder processing is local-only. Public memory processing is also local
by default; any future online processing requires an explicit user setting.

### Incremental input

The model receives:

- the previous active thread summary;
- only transcript entries after `memory_processed_revision`;
- stable turn indices and speaker labels;
- existing near-duplicate memories needed for deduplication;
- explicit privacy scope (`public`, `private-thread`, or one folder ID).

It does not receive every historical raw turn on every job.

### Structured output

```json
{
  "thread_summary": {
    "content": "Current concise state of the conversation",
    "open_questions": [],
    "decisions": [],
    "tasks": []
  },
  "candidate_memories": [
    {
      "record_type": "user_fact",
      "canonical_key": "communication.response_style",
      "content": "The user prefers concise answers.",
      "importance": 0.93,
      "confidence": 0.98,
      "stability": "durable",
      "sensitivity": "normal",
      "evidence_turns": [4],
      "operation": "add"
    }
  ]
}
```

For any private job, `user_fact` is not an allowed output type. Validation
rejects it even if the model emits it.

### Profile admission policy

User memory is intentionally sparse. A candidate is accepted automatically only
when:

- it comes from the user's own words;
- it is useful across future public threads;
- it is durable rather than a one-off task detail;
- `importance >= 0.85`;
- `confidence >= 0.85`;
- it is not prohibited or secret-like;
- it is either explicit, repeated, or a direct "remember this" instruction.

Likely profile material:

- preferred communication style;
- stable identity and role facts;
- standing constraints;
- long-lived goals;
- durable tool/workflow preferences;
- recurring project/entity relationships.

Rejected profile material:

- transient requests and current task state;
- assistant guesses or unconfirmed tool output;
- credentials, tokens, passwords, private keys, or secret values;
- speculative personality or medical/legal/financial inferences;
- facts whose main value is confined to one folder or thread.

Sensitive but non-secret facts require explicit user confirmation before public
profile admission. Private folders may retain sensitive summaries within that
folder, but secret-like strings are never embedded.

### Deduplication and contradiction handling

Before applying a candidate:

1. search active records with the same `canonical_key`;
2. search semantically similar active records in the same permitted scope;
3. classify the candidate as `ignore`, `confirm`, `add`, or `supersede`;
4. preserve old records and link supersession instead of overwriting history.

No cross-scope candidate lookup is allowed for private folders.

## Folder-Level Summarization

Each successful private thread digest updates the folder summary in the same job.

Folder rollup input:

- the previous active folder summary;
- the new thread digest;
- a bounded set of active, high-importance records from that exact folder.

The folder summary should describe durable folder-level context, not concatenate
every thread summary. It should emphasize:

- current goals and themes;
- important entities;
- decisions and standing constraints;
- unresolved questions;
- recently changed facts.

Concurrent folder jobs use a locked folder revision or compare-and-swap. If the
folder revision changes during inference, retry the rollup against the newer
summary rather than overwriting it.

Public user-profile rollup follows the same pattern using only active user-scope
facts.

## Embedding Design

### Provider

Add an `EmbeddingProvider` interface:

```python
class EmbeddingProvider(Protocol):
    model_id: str
    model_revision: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

The basic implementation uses a deterministic local feature-hashing provider:

```text
geist-feature-hash-v1
```

It hashes normalized word and character-trigram features into a fixed-size
vector and L2-normalizes the result using only the Python standard library. This
keeps the first version offline, deterministic, and dependency-free. The
provider boundary remains replaceable by a semantic model once retrieval
quality is measured against representative Geist memory queries.

### What gets embedded

Embed:

- thread summaries;
- folder summaries;
- user-profile summaries;
- accepted atomic facts, episodes, decisions, and tasks.

Do not embed:

- entire raw transcripts;
- failed/cancelled tool traces;
- secrets or prohibited memory;
- inactive superseded records unless an administrative reindex explicitly asks.

### Storage and search

Store normalized float32 vectors as `LargeBinary`. This stays portable between
SQLite and PostgreSQL and adds no database extension.

For the basic implementation:

1. filter by `user_id`, exact allowed scope, active state, and record type in SQL;
2. load the bounded candidate vectors;
3. calculate cosine similarity with a small standard-library implementation;
4. combine semantic score with lexical match, importance, confidence, and
   recency;
5. return a token/character-bounded, deduplicated result set.

This is appropriate for early per-user/per-folder memory volume. Add a
`VectorIndex` adapter later for pgvector or another ANN index once measured scope
sizes make linear scoring materially slow. The canonical tables and retrieval
contract should not depend on pgvector.

Embedding failures do not discard summaries. Persist the memory with an
`embedding_pending` state and let the idempotent job retry indexing.

## Retrieval and Prompt Assembly

Add `MemoryContextService.build_context(...)`, called by `ChatOrchestrator`
before recent transcript hydration.

### Public chat context

```text
system instructions
+ compact active user-profile summary
+ top relevant active user facts
+ active current thread summary
+ top relevant thread records
+ recent raw transcript
+ current user message
```

### Private-folder chat context

```text
system instructions
+ compact active exact-folder summary
+ top relevant active exact-folder records
+ active current thread summary
+ top relevant thread records
+ recent raw transcript
+ current user message
```

The current prompt is embedded once for semantic search. Scope filtering happens
before vector scoring; filtering results afterward is not an acceptable privacy
boundary.

Initial budgets:

- profile/folder summary: 2,000 characters;
- thread summary: 3,000 characters;
- retrieved atomic memories: 8 records and 6,000 total characters;
- retain the existing bounded recent transcript logic.

Retrieved memory is labeled as historical context, delimited from instructions,
and treated as untrusted content to reduce prompt-injection risk.

## Services

```text
app/services/memory_scheduler.py
    Called after terminal chat persistence.
    Increments revision and enqueue_or_reschedule(..., delay=20).

app/services/memory_processor.py
    Loads/validates a transcript revision.
    Runs summarization, extraction, dedupe, embedding, and commit.

app/services/memory_completion.py
    Stateless structured local completion interface.

app/services/embedding_service.py
    Lazy local embedding model and batch embedding.

app/services/memory_search.py
    Scope-filtered hybrid retrieval.

app/services/memory_context.py
    Bounded prompt-context assembly.

app/services/memory_scope.py
    Central privacy rules and chat/folder reclassification.
```

Register the `chat.memory.digest` handler explicitly during application startup.
Do not depend on incidental module import side effects.

## API Surface

### Folder management

```text
POST   /api/v1/memory/folders
GET    /api/v1/memory/folders
GET    /api/v1/memory/folders/{folder_id}
PATCH  /api/v1/memory/folders/{folder_id}
DELETE /api/v1/memory/folders/{folder_id}
PUT    /api/v1/memory/folders/{folder_id}/chats/{chat_id}
DELETE /api/v1/memory/folders/{folder_id}/chats/{chat_id}
```

Every operation filters both folder and chat by the current `user_id`.

### Memory inspection and control

```text
GET    /api/v1/memory/profile
GET    /api/v1/memory/records
PATCH  /api/v1/memory/records/{memory_id}
DELETE /api/v1/memory/records/{memory_id}
POST   /api/v1/memory/search
POST   /api/v1/memory/chats/{chat_id}/reprocess
```

Users must be able to see, correct, deactivate, and delete remembered facts.
Manual corrections become high-confidence user-authored records and preserve the
superseded version for audit.

Search accepts either public scope or one exact folder ID. It never accepts an
arbitrary list of scopes from the client.

### Settings

Add settings with conservative defaults:

```text
enable_chat_memory = true
memory_idle_seconds = 20
enable_automatic_profile_memory = true
memory_processing_backend = local
embedding_model = geist-feature-hash-v1
```

Folder privacy is not a toggle on these general settings.

## Frontend

The basic UI needs:

- a private-folder sidebar/tree;
- a chat-level Memory enabled slider that disables all derived memory and
  retrieval for that chat;
- a separate chat-level Private slider that retains thread/folder memory while
  preventing global profile reads and writes;
- create, rename, delete, and move-chat operations;
- a clear privacy badge when a chat is inside a private folder;
- profile-memory review/edit/delete;
- folder-summary view;
- scoped semantic search;
- processing state that distinguishes `waiting_for_idle`, `queued`, `processing`,
  `ready`, and `failed`;
- an explicit warning when moving a public chat into a private folder because
  public memories derived only from that chat will be withdrawn.

The UI must not implement the debounce timer. It may display the server-derived
state, but the database queue remains authoritative.

## Transaction and Idempotency Rules

- A job retry cannot create duplicate digest records for the same revision.
- A stale job cannot overwrite a newer thread, folder, or profile summary.
- A folder move invalidates all queued work through `memory_revision`.
- Embedding writes are upserts keyed by memory ID and embedding model revision.
- Memory records are committed only after the second chat revision/scope check.
- Folder/profile rollups use row locking on PostgreSQL and revision comparison on
  both providers.
- Deleting or deactivating a source memory schedules its summary scope for
  deterministic rebuild.

## Security Rules

- Fix chat and job ownership before enabling memory APIs.
- Never put transcript or memory content in job payloads, results, or normal logs.
- Never search outside the current user's permitted scope.
- Treat transcript and retrieved memory as untrusted model input.
- Disable tools in memory inference.
- Validate model JSON against strict schemas and allowlisted record types.
- Reject secret-like content before embedding.
- Private folders default to local-only inference.
- Deleting a folder deletes or cryptographically irreversibly detaches its
  summaries, embeddings, and derived records according to the chosen chat-delete
  behavior.

## Migration and Backfill

1. Integrate current `origin/main`.
2. Add queue ownership/coalescing/lease fields.
3. Add memory folder, record, evidence, and embedding tables.
4. Add chat memory revision and folder fields.
5. Backfill legacy chats to the default user only where ownership can be
   established; otherwise leave memory disabled for that row.
6. Do not automatically summarize every historical chat during migration.
7. Provide an explicit background backfill/reprocess command that is resumable
   and scope-aware.

The new Alembic migration must be based on the actual migration head after main
integration, not the head in the current older checkout.

## Testing

### Queue and debounce

- terminal turn schedules one delayed job;
- a second turn inside 20 seconds reschedules/coalesces it;
- a stale revision performs no writes;
- a turn arriving during inference discards generated output;
- worker restart reclaims an expired running job;
- duplicate execution is idempotent;
- retries preserve the same privacy scope.

### Memory extraction

- incremental summaries cover only new revisions;
- invalid JSON retries without corrupting state;
- assistant claims do not become user facts;
- low-importance/transient facts are rejected;
- direct durable "remember this" facts are admitted;
- contradictions supersede rather than overwrite;
- secret-like values are not embedded.

Use fake completion and deterministic fake embedding providers in unit tests.
Do not require downloaded model weights for ordinary CI.

### Folder isolation

Mandatory negative tests:

- private-folder content never appears in user records;
- private-folder search never returns public user records;
- public search never returns folder records;
- one folder never returns another folder's records;
- one user never returns another user's chats, jobs, folders, or memories;
- moving into a folder withdraws sole-source public facts;
- moving out does not retroactively promote private history.

### Search

- semantic paraphrases retrieve the expected record;
- exact terms receive lexical credit;
- inactive/superseded records are excluded;
- importance/confidence/recency ranking is deterministic;
- embedding model revision changes trigger reindexing.

### Runtime verification

After focused unit/service/API tests:

1. run the backend tests in Docker with `PYTHONPATH=/opt/geist pytest`;
2. start with `docker compose up -d`;
3. inspect backend and worker logs for startup/job errors;
4. create a chat and verify no memory job runs before 20 seconds;
5. verify the job runs after idle and search finds a paraphrase;
6. verify public profile extraction;
7. verify private-folder isolation;
8. curl the frontend at `http://localhost:3000`;
9. run the native MLX smoke path because memory summarization and embedding touch
   local inference behavior.

## Observability

Log and expose:

- job ID, kind, owner ID, chat ID, expected/processed revision;
- scope type and folder ID;
- stage durations;
- counts of accepted/rejected/superseded records;
- embedding model/revision and vector count;
- stale/no-op/retry reasons.

Do not log transcript text, summary text, fact content, or vectors.

## Non-Goals for the Basic Version

- knowledge graphs or graph databases;
- cross-folder search;
- automatic parent-folder inheritance;
- pgvector or a separate vector database;
- embedding raw transcripts;
- learning from private folders into public profile;
- automatic online processing of private content;
- model fine-tuning;
- deleting raw chat history merely because a summary exists.

## Remaining Product Questions

The implementation uses strict two-way private isolation, feature-hash
embeddings, automatic withdrawal when a chat becomes private, completed turns
only, and leaves chats private but unfiled when a folder is deleted.

1. Should automatic user-profile admission remain enabled by default with the
   high threshold, or require user confirmation for every new fact?
2. Do we want a hard maximum active memory count per user/folder before adding an
   ANN-backed vector index?

## Verification Record

- Frontend unit/component suite: 15 suites, 60 tests passed.
- Frontend production build: passed with only pre-existing Create React App and
  `useVoiceChat` warnings.
- Focused Docker memory/API/queue suite: 26 tests passed.
- Full Docker backend suite: 444 tests passed and 4 pre-existing async tests
  skipped; after the migration test was added, the focused migration round-trip
  also passed.
- Alembic memory revision: upgrade from the real pre-memory job revision,
  downgrade, and schema assertions passed on SQLite.
- Full Playwright suite: 7 tests passed, including 4 privacy-specific flows.
- Runtime: backend healthy, logs showed no startup/job errors, memory API returned
  HTTP 200, and the branch frontend returned HTTP 200 on port 3002 because
  another Geist worktree already owned port 3000.
- Browser visual QA: memory sliders, private state, folder drawer, folder summary
  state, and scoped memory explorer were inspected at the normal viewport with
  no console errors; screenshot-only test data was removed.
- Native MLX smoke was not applicable: this implementation does not change model
  loading, tokenization, generation, runner selection, or any `MLX_BACKEND=1`
  path. Summarization and embeddings are deterministic local Python utilities.
