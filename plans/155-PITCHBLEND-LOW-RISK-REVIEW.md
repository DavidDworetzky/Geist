# Pitchblend Low-Risk Review Plan

## Goal

Create a GitHub App-backed review workflow that evaluates pull requests when an
authorized collaborator comments `@pitchblend-ai review` and approves only
eligible low-risk changes, including copy-only edits.

## Policy

- Only evaluate non-draft pull requests targeting `main`.
- Cap eligible changes at 20 files and 1,000 changed lines (additions plus
  deletions).
- Evaluate semantic eligibility using the explicit classification pass matrix.
- Evaluate review approval independently of CI; repository merge requirements enforce checks.
- Require a changed regression-test file except for low-complexity copy-only changes.
- Allow copy-only changes only when migration, contract, and security flags are false.
- Block data migrations, persistent data model changes, public or shared
  contract changes, security-sensitive changes, dependency changes, and CI/CD
  changes.
- Treat uncertainty as requiring human review.
- Bind each decision and approval to the pull request's current head SHA.

## Implementation

1. Register the `pitchblend` GitHub App with least-privilege pull-request,
   issue, contents, and commit-status access.
2. Add a workflow triggered by new pull-request conversation comments.
3. Authenticate the workflow as the GitHub App and authorize the requesting
   commenter.
4. Fetch pull-request metadata, file statistics and patches via
   the GitHub API without checking out or executing pull-request code.
5. Apply deterministic size and blocked-surface gates before invoking the
   classifier.
6. Classify with OpenAI `gpt-5.6-luna` at high reasoning effort and require
   structured, high-confidence output before approving the exact evaluated
   commit.
7. Comment with an auditable approval or human-review-required result.

## Validation

- Validate workflow YAML syntax and embedded JavaScript syntax.
- Exercise the deterministic policy against representative eligible and
  blocked pull-request fixtures.
- Confirm the workflow has no access to untrusted pull-request execution or
  checkout paths.
- Confirm secrets and GitHub App credentials are referenced only through
  GitHub Actions secrets.
