# Claude Ready-for-Review Automation Plan

## Goal

Automatically ask Claude to review a pull request when it transitions from draft
to ready for review.

## Scope

- Add a dedicated GitHub Actions workflow for the `pull_request` event's
  `ready_for_review` activity type.
- Reuse the repository's existing `ANTHROPIC_API_KEY` secret and Claude Code
  GitHub Action.
- Grant read-only repository access and write access only for pull-request
  feedback.
- Instruct Claude to report actionable findings without modifying the branch.

## Validation

- Parse the workflow as YAML.
- Confirm the workflow has only the `ready_for_review` pull-request trigger.
- Inspect the staged diff to ensure no unrelated files are included.
