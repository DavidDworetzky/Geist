# QA Engineer Skill Plan

## Goal

Add a Geist-specific QA skill that evaluates a clean pull-request branch with diff-informed white-box analysis, delegates one independent runtime test pass, records concise browser-and-terminal evidence, publishes the evidence through GitHub CLI, and reports an unambiguous `PASS`, `FAIL`, or `BLOCKED` verdict.

## Scope

- Trigger after a pull request exists and require the checked-out PR branch to be clean.
- Inspect the PR diff at entry points, downstream services, and changed contracts without attempting exhaustive call-graph analysis.
- Rank affected behaviors by risk and test only the highest-risk pathways within a few minutes of browser operation.
- Require one independent QA subagent to execute the selected matrix without changing implementation code or tests.
- Cover applicable Docker-default, native MLX with a real Qwen 3 model, llama.cpp, and Transformers pathways.
- Record browser behavior together with terminal or log evidence, including failures and the final verdict.
- Publish the MP4 and a traceability report to the pull request with GitHub CLI.

## Implementation

1. Add `.agents/skills/qa-engineer/SKILL.md` with the preflight, risk-analysis, delegation, runtime, recording, publication, and verdict contracts.
2. Add a runtime reference describing the four Geist pathways, evidence expectations, time bounds, and blocked conditions.
3. Add deterministic scripts to:
   - collect and validate PR/branch/diff context;
   - render a stable QA verdict report from structured results;
   - publish the MP4 through a GitHub CLI-addressable evidence release and link it from the PR comment.
4. Reuse `.agents/skills/geist-test-loop/SKILL.md` for Geist startup and smoke-test commands and the browser-control skill for UI operation.
5. Keep implementation and test mutation outside the independent QA subagent's authority.

## Verdict Rules

- `PASS`: every selected high-risk test and every applicable required runtime lane passed with evidence.
- `FAIL`: at least one selected behavior produced a reproducible product failure or regression.
- `BLOCKED`: prerequisites or required evidence were unavailable, including a dirty branch, missing/open PR mismatch, invalid GitHub authentication, unavailable required runtime/model, or failed evidence publication.

Do not downgrade a required blocked lane into a pass. Include failing and blocked evidence in the recording and report.

## Verification

- Run the skill initializer and skill validator.
- Syntax-check and exercise bundled scripts with local fixtures or non-mutating dry runs.
- Forward-test the completed skill with one independent subagent using raw repository and PR state.
- Confirm that the forward-test respects the clean-branch and open-PR preconditions and does not mutate implementation code.

## GitHub Evidence Constraint

Stock `gh pr comment` accepts comment text but not local file attachments. Use `gh release upload` against a dedicated draft QA-evidence release, resolve the uploaded asset URL, and post that link with `gh pr comment`. Keep the release draft-only so QA artifacts do not appear as product releases. Treat any authentication or publication failure as `BLOCKED`.
