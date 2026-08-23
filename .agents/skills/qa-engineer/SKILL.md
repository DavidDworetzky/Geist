---
name: qa-engineer
description: Run independent, diff-driven local QA for an open Geist pull request on a clean checked-out branch. Use after a PR is opened when Codex must inspect changed entry points, downstream services, and contracts; rank the riskiest user pathways; delegate one independent QA subagent; exercise applicable Docker, native MLX, llama.cpp, and Transformers behavior with a real local Qwen 3 model; capture short browser-and-terminal MP4 evidence; publish it with GitHub CLI; and return a strict PASS, FAIL, or BLOCKED verdict.
---

# Geist QA Engineer

Produce independent evidence for the riskiest behaviors changed by one open Geist pull request. Keep browser operation focused and brief. Test enough white-box context to select meaningful pathways, not to reconstruct the entire application.

## Non-negotiable contract

1. Run only after the PR exists.
2. Require the checked-out branch to match the PR head, match the pushed head SHA, and have no staged, unstaged, or untracked files.
3. Never read `.env` or `.env.*`. Confirm configuration only by variable name or presence through safe application surfaces.
4. Delegate execution to exactly one independent QA subagent. Do not delegate implementation fixes.
5. Give the subagent the PR identifier, repository path, acceptance criteria, this skill path, and raw diff context. Do not give it an expected verdict or the implementer's conclusions.
6. Forbid the QA subagent from editing code, tests, configuration, tracked files, or PR metadata before publication.
7. Use a real, already-installed local model for required inference checks. Prefer `Qwen/Qwen3-4B`. Do not download models, install packages, or change local configuration without explicit user approval.
8. Keep total interactive browser operation near two to four minutes. Spend setup and focused automated-test time outside the recording when possible.
9. Include failures and blockers in the evidence. Never stage a demonstration or hide a failed attempt.
10. Return exactly one overall verdict: `PASS`, `FAIL`, or `BLOCKED`.

## Step 1: Establish immutable PR context

Run:

```bash
python3 .agents/skills/qa-engineer/scripts/collect_pr_context.py --pr <number-or-url> --output /private/tmp/geist-qa-context.json
```

Stop with `BLOCKED` when the script reports any blocker. Do not clean, stash, commit, reset, or otherwise modify the branch to satisfy the preflight. Ask the user to provide a clean PR branch.

Record the PR number, URL, base SHA, head SHA, branch, and changed-file list. Use the PR base/head SHAs from this context for every later diff command.

## Step 2: Build a shallow white-box risk map

Inspect the PR diff and only enough surrounding code to understand:

- changed external entry points such as routes, CLI commands, browser actions, jobs, or agent calls;
- directly downstream services and persistence boundaries;
- request, response, database, runner, or frontend state contracts changed by the PR;
- existing focused tests and obvious missing assertions;
- platform or backend selection that can cause the same behavior to diverge.

Use `git diff <base_sha>...<head_sha>`, `rg`, direct file reads, and existing tests. Do not perform a full security review or exhaustive call graph unless the diff exposes a specific high-risk issue.

Create a ranked traceability list with these fields:

| Field | Meaning |
|---|---|
| Rank | `1` is highest risk |
| Code path | Changed entry point and important downstream boundary |
| Behavior | User-visible or contract outcome to prove |
| Risk | Why the change could regress |
| Runtime lanes | Docker, MLX, llama.cpp, Transformers, or a justified subset |
| Test | Shortest meaningful automated/API/browser exercise |
| Expected evidence | Visible state, response, log line, or persisted result |

Rank by correctness impact, contract breadth, runtime divergence, persistence risk, and user visibility. Select only the highest-risk pathways that fit the time budget. Always include a changed core contract or inference boundary even when its browser symptom is indirect.

Read [references/runtime-matrix.md](references/runtime-matrix.md) to select applicable runtime lanes and commands.

## Step 3: Hand off one independent QA run

Spawn exactly one subagent with a prompt shaped like:

```text
Use $qa-engineer at <absolute-skill-path> to QA PR <number> in <repo-path>.
The implementing agent has supplied only these acceptance criteria: <criteria>.
Read /private/tmp/geist-qa-context.json, independently inspect the raw PR diff,
rank the riskiest pathways, and execute the applicable runtime matrix. Do not
edit code, tests, tracked files, or PR metadata. Store all temporary evidence
under /private/tmp. Return structured results and artifact paths to the parent.
```

Do not ask the subagent to validate the skill definition. Ask it to perform the QA job. The parent remains responsible for checking the returned evidence, rendering the report, and publishing it.

## Step 4: Execute focused tests and runtime lanes

Require the subagent to:

1. Run the smallest existing tests that directly cover selected paths.
2. Start only the applicable runtime lanes from the matrix.
3. Confirm the actual runner and model from application responses or logs; configuration intent alone is not evidence.
4. Send a real prompt through each required inference lane. Use a deterministic short prompt such as `Reply with exactly: GEIST_QA_OK` unless feature acceptance criteria require another prompt.
5. Exercise the highest-risk user pathway in the browser, inspect visible results and browser console, and correlate them with backend logs or API responses.
6. Restore reversible UI settings and stop processes cleanly.
7. Mark an unavailable required lane `BLOCKED`; do not substitute a unit test or another runner and call it passed.

Reuse `.agents/skills/geist-test-loop/SKILL.md` for Geist startup, Docker, browser, chat, settings, and native MLX practices. Use the installed browser-control skill for interactive UI testing.

## Step 5: Record concise evidence

Before recording:

- arrange the browser and one terminal/log window on the designated display;
- close unrelated windows and disable visible notifications;
- ensure no secrets, tokens, private conversations, or unrelated user data are visible;
- prepare runtimes so the recording focuses on proof rather than downloads or long startup waits.

Start the recorder in a long-running command session:

```bash
.agents/skills/qa-engineer/scripts/record_qa_evidence.sh \
  --seconds 240 \
  --output /private/tmp/geist-pr-<pr>-<short-sha>-qa.mp4
```

During the recording, show:

1. PR number and exact head SHA.
2. The selected traceability items and runtime lane under test.
3. Browser actions for the highest-risk pathway.
4. The corresponding terminal logs, response, or focused-test output.
5. Any failure or blocker encountered.
6. A final terminal line containing `VERDICT: PASS`, `VERDICT: FAIL`, or `VERDICT: BLOCKED`.

Do not exceed five minutes. If evidence cannot fit, prioritize the highest-ranked pathway and summarize the remaining machine-verifiable results in the report.

## Step 6: Decide the verdict

Apply these rules mechanically:

- `PASS`: every selected high-risk test passed, every applicable required lane passed, the real model and runner were confirmed, and usable MP4/report evidence exists.
- `FAIL`: any selected behavior produced a reproducible incorrect result, regression, crash, unsafe state, contract violation, or unexpected error.
- `BLOCKED`: the PR/branch preflight failed; a required runtime, model, browser, permission, or credential was unavailable; evidence could not be recorded; or evidence could not be published.

Failure outranks blocked: report `FAIL` when a reproducible product failure exists even if another lane is blocked. Never emit `PASS WITH CAVEATS`.

## Step 7: Render and publish evidence

Write the subagent result as JSON following [references/result-schema.md](references/result-schema.md), then render it:

```bash
python3 .agents/skills/qa-engineer/scripts/render_qa_report.py \
  --input /private/tmp/geist-qa-result.json \
  --output /private/tmp/geist-qa-report.md
```

Review the MP4 and report before publication. Confirm that the video is playable, contains no secrets or unrelated user data, matches the tested head SHA, and supports the verdict.

Publish with:

```bash
python3 .agents/skills/qa-engineer/scripts/publish_pr_evidence.py \
  --pr <number-or-url> \
  --video /private/tmp/geist-pr-<pr>-<short-sha>-qa.mp4 \
  --report /private/tmp/geist-qa-report.md
```

Stock `gh pr comment` cannot directly attach a local binary. The publisher therefore uses `gh release upload` to place uniquely named assets in a dedicated draft `qa-evidence` release and posts the asset link plus report with `gh pr comment`. Keep that release draft-only. Treat upload or comment failure as `BLOCKED` and do not claim that the PR has evidence.

## Final response

Lead with the verdict, PR URL, and tested head SHA. Then provide:

- the highest-risk pathways tested;
- per-lane results and confirmed runner/model;
- failures or blockers with exact reproduction evidence;
- MP4 and PR-comment URLs;
- exact commands run and any non-applicable lanes with justification.

Do not say the PR passed when the published comment, recording, or a required runtime lane is missing.
