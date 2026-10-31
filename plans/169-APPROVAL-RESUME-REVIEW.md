# Approval resume and session lifecycle review

## Final lifecycle review 5577327979

Make registry startup explicit and restartable only after previous executor work
has finished, retaining session ownership/caps and retrying failed retired-scope
cleanup. Keep capacity exhaustion fail-closed: one-shot fallback loses the user's
workspace and is not equivalent behavior. Add regression coverage for repeated
lifespans, cleanup recovery, and authenticated approval/routine API boundaries.

Persist bounded routine outcome/error fields through an additive migration;
claim records running status, final updates compare the claimed timestamp and
running status so late workers cannot overwrite timeout/newer runs. Show outcomes,
queued one-off requests and blocked scheduler state in routine settings. Expose
only safe fixed error summaries; actual exceptions remain in logs. Do not spawn
replacement workers while a timed-out worker is still alive. No new dependencies.

Integrate execution parent `75f3756` into PR #308, preserving non-waivable
approval, policy-blocked tool results, validated mounts, and bounded capture.
Address Claude review 5574114357 in focused changes before publishing this layer.

## Approval lifecycle

- Bind one-shot decisions to the exact tool invocation, not a reusable model ID.
  Reconcile with the invocation-bound approval implementation already present in
  descendant #353 rather than creating incompatible contracts.
- Keep approval waits from monopolizing the shared request pool; send SSE
  keepalives and cancel on disconnect. Record the awaiting-approval history state.
- Preserve denial/control messages outside aggregate output truncation, prevent
  repeat prompts after a per-run denial, and distinguish timeout from user denial.
- Bound session-grant lifetime and clear grants with their owning lifecycle.
  Persist always-allow decisions transactionally without lost concurrent grants.
- Surface approval API failures in the UI and restore controls when retry is safe.

## Persistent execution

- Apply the parent's command-rejection policy and validated mount helper to the
  session path before launching a runtime. Use the shared bounded-output capture
  helper for command execution, with a TERM-to-KILL deadline.
- Bound inspect/create/remove operations and the whole operation's overhead so
  wedged daemons cannot consume executor threads indefinitely.
- Serialize session creation and execution for a scope; make reaping aware of
  active scopes. Cap managed sessions and validate TTL/cap configuration.
- Wire owned-session shutdown and chat deletion cleanup. Use instance ownership
  when identifying containers; never reap another live backend's sessions or the
  user's legacy sandbox. Startup reconciliation must fail closed on ownership.
- Do not automatically migrate or delete existing user files or running sessions.

## Routines

- Add scheduler enable/poll controls, bounded due batches, and execution budgets
  that contain slow runs without silently launching unbounded background work.
- Keep a one-off run distinct from enabling recurrence. Recompute next execution
  when intervals change, including before the first run.
- Retain owner settings and traceable routine-to-chat history. Whitelist database
  updates and align indexes/foreign-key lifecycle with supported schema paths.
- Display timestamps with explicit timezone semantics and validate interval bounds
  consistently in the API and UI.

## Verification and scope

Use Docker approval/session/routine and API contract tests, including concurrent
creation, runtime timeouts, invocation-ID reuse, denied-output truncation, expired
grants, disabled one-off routines, and failed approval submissions. Run focused
agent/orchestrator contracts when those change, then broader backend tests and
frontend checks. Exercise isolated startup and browser chat/settings when unlocked.
Native MLX validation is required if inference or local bootstrap paths change.

The selected theme-aware stop/resume UI belongs in #353, not this layer.
Keep workspace-only changes in
#361 and preserve the original #353 worktree's staged main integration and fixes.

## Implemented review follow-up

The main integration retains operator-scoped approval resolution, introduces
nonblocking approval polling for the async SSE adapter, bounds chat grants by
definition fingerprint and TTL, and writes standing grants with optimistic JSON
compare-and-swap. Reused model call IDs receive unique invocation IDs before being
persisted. Denials remain explicit control messages even after output exhaustion.

Persistent sessions are capped, owned by a random backend instance, serialized by
scope, and cleaned up through bounded runtime calls. New containers self-expire
after 24 hours with `--rm`, limiting crash leftovers without touching containers
owned by another backend or older user sessions. There is no chat-deletion API in
this layer; `ToolRegistry.close_chat` is the lifecycle entry point for a future
deletion caller, while expiry/shutdown already clean up temporary state. The
shared hardline policy applies to host-reaching execution, as in the parent.

Routine one-off requests use an additive boolean field; existing routine data and
foreign-key deletion behavior are preserved. Recurrence is never enabled by Run
Now. Scheduler work is bounded to one live run worker, with cooperative
cancellation, owner settings, and a labeled routine prompt in chat history.

Local integration checkpoint: 527 Docker service/database/API/native-tool-turn
tests passed, nine production files passed mypy, 225 frontend tests and the
production build passed (before the latest routine UI and warning-summary edits).
Docker tests exclude the existing TTS/voice suites because the cached image lacks
torchaudio; no package installation was authorized. Native and isolated UI smoke
remain required before publication. Merge the newest execution parent before
publishing so dynamic-source grant eligibility is authoritative.

Latest checks: 108 focused native MLX-mode contract/cache/route tests passed;
225 frontend tests and production build passed after the routine UI changes.
Changed production frontend/hook/test files pass scoped ESLint. Normal commit
hooks hit unchanged cached-whisper mypy Any, main's existing B104 binding, and
91 baseline frontend-test lint findings. The new session subprocess import is
documented and scoped Bandit passes; only those three known baseline hooks are
skipped for the local merge commit, with CI unchanged.

Final integrated validation: 540 Docker backend tests passed (including actual
concurrent permission CAS and data-preserving one-off migration regressions),
103 native MLX-mode contracts passed, 226 frontend tests/build and six-file mypy
passed. Isolated Docker startup/UI200 and synthetic streaming chat completed.
Browser is blocked by the locked Mac, with no user runtime or database changed.
Session grants use authoritative standing-grant eligibility, one-off timeout
errors remain distinct, and sessions inherit writable temporary HOME/cache.

Final review dispositions: M1/M2 now reopen only after prior work stops and retry
owned retired-scope cleanup. M3 deliberately remains a visible capacity error;
one-shot fallback would lose filesystem continuity and conceal the resource cap.
M4/M5 expose blocked-worker state and persist timestamp-bound outcomes without
allowing a late worker to overwrite a timeout or newer occurrence. M6 adds actual
authenticated HTTP approval/routine coverage and repeated-lifespan tests.

L1 adds execution capability requirements for routine mutations. L6 preserves
stale-approval explanations, L7 distinguishes expired/answered UI responses, L8
provides a timeout fallback message, L9 promotes first-turn persistent sessions,
and L10 exposes queued one-off work. The optional prompt/interval editor is deferred.
L2's future chat-deletion hook remains documented; no deletion endpoint exists.
L3 keeps the sync compatibility seam while the async wrapper and real SSE smoke
cover production. L4's approval_wait is an internal cooperative-yield contract,
paced by the async adapter; sleeping inside it would occupy a shared worker again.
L5 retains defense-in-depth fingerprint validation across yield boundaries.
L11/L12 retain the existing index and naive-UTC database convention; serialized
timestamps have explicit UTC offsets. L13 is bounded FIFO, not advertised as LRU.
L14 retains the parent-owned index on downgrade; dropping it would break the
ancestor schema. No published migration history is rewritten.
# Re-review dispositions (5579139548)

N1/N2: retain hard bounds. Replacing a hung thread or excluding unconfirmed
container cleanup from capacity can accumulate live workers/containers across
restarts. Report the degraded state and require process/daemon recovery instead.
N3: record cancellation before worker start. A read-only API projection labels
overdue running outcomes unknown; do not globally rewrite another process's live
claim or automatically replay a possibly completed side effect after a crash.
N5: preserve queued one-off timing when editing the interval. N10: inject the
scheduler through app.state instead of importing main from an endpoint. N4: UI
refreshes while visible and mutations refetch GET (they never replace rows from
mutation responses); an empty routine list has no scheduled work to diagnose.
N6: disabling intentionally cancels queued work; a separate one-off cancellation
affordance remains optional. N7: preserve subsecond cooperative-stop latency.
N8/N9: approval-unavailable wording must cover unknown/expired/stale decisions;
never expose cross-owner existence. N11: default unhandled responses are already
sanitized 500s. N12: Alembic revision IDs are opaque; do not rename a published
migration. N13: transcript stores latest per-call state, not a full consent audit.
N14: legacy-settings repair precedes adoption and is covered by migration tests.

Final review 5579385278: coalesce a refresh requested during an active fetch so
successful mutations cannot leave a stale list (P1). Background polling keeps
mutation errors visible (P2). A rejected standing grant leaves Approve once and
Deny available and hides the rejected standing choices (P3); expired requests
still disable every choice. RoutineResponse uses ConfigDict and API tests disable
the scheduler explicitly (P6/P7). List-only scheduler projection and cross-process
unknown outcomes remain deliberate read-only behavior (P4/P5).
Validation: 234 frontend tests, production build, and 3 Docker API tests pass.
