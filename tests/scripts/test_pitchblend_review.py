from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock


def load_pitchblend() -> ModuleType:
    script_path = (
        Path(__file__).parents[2] / ".github" / "scripts" / "pitchblend_review.py"
    )
    spec = importlib.util.spec_from_file_location("pitchblend_review", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pitchblend = load_pitchblend()


def eligible_pull_request() -> dict:
    return {
        "base": {"ref": "main"},
        "draft": False,
        "mergeable": True,
    }


def pr_330_files() -> list[dict]:
    return [
        {"filename": "agents/model_catalog.py"},
        {"filename": "docs/agents.md"},
        {"filename": "docs/open_weight_models.md"},
        {"filename": "tests/agents/test_model_catalog.py"},
        {"filename": "tests/agents/test_online_agent.py"},
    ]


def pr_341_files() -> list[dict]:
    return [
        {
            "filename": path,
            "status": "modified",
            "additions": 5,
            "deletions": 0,
            "patch": "@@ -1 +1 @@",
        }
        for path in (
            "agents/architectures/registry.py",
            "client/geist/src/Components/__tests__/AgentConfigSection.test.tsx",
            "client/geist/src/Hooks/useAvailableModels.tsx",
            "plans/156-CLAUDE-FABLE-MYTHOS-5-1-ONLINE-CATALOG.md",
            "scripts/model_filter_config.py",
            "tests/agents/test_anthropic_model_registry.py",
        )
    ]


def test_exact_review_command() -> None:
    assert pitchblend.COMMAND_PATTERN.fullmatch("@pitchblend-ai review")
    assert pitchblend.COMMAND_PATTERN.fullmatch("@Pitchblend-AI   review")
    assert not pitchblend.COMMAND_PATTERN.fullmatch("please @pitchblend-ai review")
    assert not pitchblend.COMMAND_PATTERN.fullmatch("@pitchblend-ai approve")


def test_gate_accepts_small_localized_fix_with_test() -> None:
    files = [
        {
            "filename": "app/services/titles.py",
            "status": "modified",
            "additions": 12,
            "deletions": 4,
            "patch": "@@ -1 +1 @@",
        },
        {
            "filename": "tests/services/test_titles.py",
            "status": "modified",
            "additions": 15,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        },
    ]

    result = pitchblend.deterministic_gate(
        eligible_pull_request(), files
    )

    assert result.eligible
    assert result.changed_lines == 32


def test_gate_blocks_migration_contract_security_and_dependency_paths() -> None:
    files = [
        {"filename": "migrations/001.sql", "status": "modified", "patch": "x"},
        {"filename": "agents/base_agent.py", "status": "modified", "patch": "x"},
        {"filename": ".github/workflows/ci.yml", "status": "modified", "patch": "x"},
        {"filename": "uv.lock", "status": "modified", "patch": "x"},
        {"filename": "tests/test_policy.py", "status": "modified", "patch": "x"},
    ]

    result = pitchblend.deterministic_gate(
        eligible_pull_request(), files
    )

    assert not result.eligible
    assert any("data migration" in reason for reason in result.reasons)
    assert any("contract" in reason for reason in result.reasons)
    assert any("security" in reason for reason in result.reasons)
    assert any("dependency" in reason for reason in result.reasons)


def test_gate_blocks_size_limits() -> None:
    files = [
        {
            "filename": f"src/fix_{index}.py",
            "status": "modified",
            "additions": 50,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        }
        for index in range(21)
    ]
    files[0]["filename"] = "tests/test_fix.py"

    result = pitchblend.deterministic_gate(eligible_pull_request(), files)

    assert not result.eligible
    assert any("20-file limit" in reason for reason in result.reasons)
    assert any("1000-line limit" in reason for reason in result.reasons)


def test_model_catalog_scope_matches_pr_330_and_pr_341() -> None:
    assert pitchblend.is_model_catalog_only_change(pr_330_files())
    assert pitchblend.is_model_catalog_only_change(pr_341_files())
    assert not pitchblend.is_model_catalog_only_change(
        [{"filename": "tests/agents/test_model_catalog.py"}]
    )
    assert not pitchblend.is_model_catalog_only_change(
        pr_330_files() + [{"filename": "agents/online_agent.py"}]
    )


def test_gate_allows_scoped_registry_catalog_change_but_blocks_near_miss() -> None:
    result = pitchblend.deterministic_gate(
        eligible_pull_request(), pr_341_files()
    )
    assert result.eligible

    unscoped_files = pr_341_files() + [
        {
            "filename": "agents/online_agent.py",
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "patch": "@@ -1 +1 @@",
        }
    ]
    result = pitchblend.deterministic_gate(
        eligible_pull_request(), unscoped_files
    )
    assert not result.eligible
    assert any("contract" in reason for reason in result.reasons)


def base_classification(**overrides: object) -> dict:
    classification = {
        "change_type": "bugfix",
        "complexity": "low",
        "is_frontend": False,
        "data_migration": False,
        "contract_change": False,
        "security_change": False,
        "regression_test_present": True,
        "reason": "Localized correction.",
        "risk_flags": [],
    }
    classification.update(overrides)
    return classification


def test_classifier_matrix_covers_major_decision_boundaries() -> None:
    for change_type in ("bugfix", "feature", "refactor"):
        for complexity in ("low", "medium"):
            classification = base_classification(
                change_type=change_type,
                complexity=complexity,
                is_frontend=True,
                contract_change=True,
                risk_flags=["subjective diagnostic"],
            )
            assert pitchblend.classification_pass_rule(classification) == (
                f"frontend-{change_type}-low-or-medium"
            )

    assert pitchblend.classification_pass_rule(
        base_classification(is_frontend=True, change_type="feature", complexity="high")
    ) is None
    assert pitchblend.classification_pass_rule(
        base_classification(is_frontend=True, change_type="docs", complexity="low")
    ) is None
    assert pitchblend.classification_pass_rule(
        base_classification(complexity="medium")
    ) == "non-frontend-bugfix-low-or-medium"
    assert pitchblend.classification_pass_rule(
        base_classification(complexity="medium", contract_change=True)
    ) is None


def test_catalog_only_matrix_route_accepts_pr_330_and_pr_341_classifier_shape() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        contract_change=False,
        reason="Adds a hosted OpenRouter model entry and exposes its metadata.",
    )

    assert pitchblend.classification_pass_rule(classification) is None
    for files in (pr_330_files(), pr_341_files()):
        assert (
            pitchblend.classification_pass_rule(
                classification,
                model_catalog_only=pitchblend.is_model_catalog_only_change(files),
            )
            == "model-catalog-low-or-medium"
        )
    approved = pitchblend.format_approved_comment(
        341,
        "8383580aa22b0ff7",
        len(pr_341_files()),
        103,
        classification,
        "model-catalog-low-or-medium",
    )
    assert "Medium-complexity model catalog feature" in approved

    classification["security_change"] = True
    assert (
        pitchblend.classification_pass_rule(
            classification, model_catalog_only=True
        )
        is None
    )

    classification["security_change"] = False
    classification["contract_change"] = True
    assert (
        pitchblend.classification_pass_rule(
            classification, model_catalog_only=True
        )
        is None
    )


def test_geist_pr_325_classifier_shape_passes_despite_subjective_diagnostics() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        is_frontend=True,
        contract_change=True,
        reason=(
            "Adds local-model selection, shared artifact loading, and immediate "
            "settings persistence."
        ),
        risk_flags=[
            "new_persisted_settings_field",
            "shared_global_artifact_fetch_and_listener_state",
            "async_save_failure_and_rollback_behavior",
            "backend_settings_contract_compatibility",
            "full_stack_verification_incomplete",
        ],
    )

    assert (
        pitchblend.classification_pass_rule(classification)
        == "frontend-feature-low-or-medium"
    )


def test_classifier_schema_is_forward_and_backward_compatible() -> None:
    legacy = base_classification()
    legacy.pop("is_frontend")
    legacy["frontend_only"] = True
    legacy["future_field"] = {"ignored": True}

    normalized = pitchblend.normalize_classification(legacy)

    assert normalized["is_frontend"] is True
    assert normalized["future_field"] == {"ignored": True}
    assert pitchblend.CLASSIFICATION_SCHEMA["additionalProperties"] is True


def test_classifier_uses_frontend_boundary_prompt_and_safe_response_budget() -> None:
    classification = base_classification()
    client = mock.Mock()
    client.request.return_value = {
        "status": "completed",
        "output": [
            {"content": [{"type": "output_text", "text": json.dumps(classification)}]}
        ],
    }
    with mock.patch.object(
        pitchblend, "JsonHttpClient", return_value=client
    ) as client_type:
        result = pitchblend.classify_pull_request(
            "key", "gpt-5.6-luna", "high", {"title": "Fix"}, []
        )

    assert result["is_frontend"] is False
    assert client_type.call_args.kwargs["timeout_seconds"] == 300
    payload = client.request.call_args.args[2]
    assert payload["max_output_tokens"] == 16_000
    assert payload["text"]["format"]["strict"] is False
    assert "is_frontend" in payload["text"]["format"]["schema"]["required"]
    assert "confidence" not in payload["text"]["format"]["schema"]["required"]
    assert "Frontend code remains frontend when it calls an existing API" in payload[
        "instructions"
    ]
    assert "model availability and descriptive metadata" in payload["instructions"]
    assert "runner selection, routing, execution behavior" in payload["instructions"]


def test_comments_include_classifier_json_and_objective_matrix_decision() -> None:
    classification = base_classification(
        change_type="feature",
        complexity="medium",
        is_frontend=True,
        contract_change=True,
        risk_flags=["Subjective note"],
    )
    matched_rule = pitchblend.classification_pass_rule(classification)
    assert matched_rule == "frontend-feature-low-or-medium"

    approved = pitchblend.format_approved_comment(
        123, "0123456789abcdef", 4, 120, classification, matched_rule
    )
    blocked = pitchblend.format_blocked_comment(124, ["High complexity"], classification)

    assert '"is_frontend": true' in approved
    assert '"approved": true' in approved
    assert '"matched_pass_rule": "frontend-feature-low-or-medium"' in approved
    assert '"approved": false' in blocked
    assert '"matched_pass_rule": null' in blocked


def gate_pull_request() -> dict:
    return {
        "number": 42,
        "head": {"sha": "current-head"},
        "user": {"login": "pull-author"},
        "html_url": "https://github.test/org/repo/pull/42",
    }


def review(login: str, user_type: str = "User", **overrides: object) -> dict:
    value = {
        "id": 1,
        "state": "APPROVED",
        "commit_id": "current-head",
        "user": {"login": login, "type": user_type},
    }
    value.update(overrides)
    return value


def test_approval_gate_accepts_current_pitchblend_or_write_human() -> None:
    github = mock.Mock()
    app_source = pitchblend.approval_gate_source(
        github,
        "org",
        "repo",
        gate_pull_request(),
        [review("pitchblend-ai[bot]", "Bot")],
    )
    assert app_source == "pitchblend"
    github.request.assert_not_called()

    github.request.return_value = {"permission": "write"}
    human_source = pitchblend.approval_gate_source(
        github, "org", "repo", gate_pull_request(), [review("maintainer")]
    )
    assert human_source == "human:maintainer"


def test_approval_gate_rejects_stale_author_read_and_superseded_reviews() -> None:
    github = mock.Mock()
    github.request.return_value = {"permission": "read"}
    reviews = [
        review("pitchblend-ai[bot]", "Bot", commit_id="old-head"),
        review("pull-author"),
        review("reader"),
        review("former-approver", id=2),
        review("former-approver", id=3, state="CHANGES_REQUESTED"),
    ]

    assert (
        pitchblend.approval_gate_source(
            github, "org", "repo", gate_pull_request(), reviews
        )
        is None
    )


APPROVED_SHA = "a" * 40


def test_stale_pitchblend_approval_counts_only_when_it_carries_forward() -> None:
    github = mock.Mock()
    reviews = [review("pitchblend-ai[bot]", "Bot", commit_id=APPROVED_SHA)]
    with mock.patch.object(
        pitchblend, "approval_carries_forward", return_value=True
    ) as carries_forward:
        source = pitchblend.approval_gate_source(
            github, "org", "repo", gate_pull_request(), reviews
        )
    assert source == f"pitchblend-carried:{APPROVED_SHA}"
    assert carries_forward.call_args.args[-1] == APPROVED_SHA

    with mock.patch.object(pitchblend, "approval_carries_forward", return_value=False):
        assert (
            pitchblend.approval_gate_source(
                github, "org", "repo", gate_pull_request(), reviews
            )
            is None
        )


def test_dismissed_or_human_approval_skips_pitchblend_carry_forward() -> None:
    github = mock.Mock()
    github.request.return_value = {"permission": "write"}
    with mock.patch.object(pitchblend, "approval_carries_forward") as carries_forward:
        dismissed = [
            review("pitchblend-ai[bot]", "Bot", id=1, commit_id=APPROVED_SHA),
            review("pitchblend-ai[bot]", "Bot", id=2, state="DISMISSED"),
        ]
        assert (
            pitchblend.approval_gate_source(
                github, "org", "repo", gate_pull_request(), dismissed
            )
            is None
        )
        human = [
            review("pitchblend-ai[bot]", "Bot", id=1, commit_id=APPROVED_SHA),
            review("maintainer", id=2),
        ]
        assert (
            pitchblend.approval_gate_source(
                github, "org", "repo", gate_pull_request(), human
            )
            == "human:maintainer"
        )
    carries_forward.assert_not_called()


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_files(repository: Path, message: str, files: dict[str, str]) -> str:
    for path, content in files.items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(repository, "add", "--all")
    git(repository, "commit", "--quiet", "-m", message)
    return git(repository, "rev-parse", "HEAD")


class CarryForwardRepository:
    """A local stand-in for GitHub: a bare `org/repo.git` plus an author clone."""

    def __init__(self, root: Path):
        self.remote = root / "server" / "org" / "repo.git"
        self.remote.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "init", "--quiet", "--bare", "-b", "main", str(self.remote)], check=True
        )
        self.author = root / "author"
        self.evaluator = root / "evaluator"
        subprocess.run(["git", "clone", "--quiet", str(self.remote), str(self.author)],
                       check=True, capture_output=True)
        for key, value in (("user.name", "Test"), ("user.email", "test@example.com")):
            git(self.author, "config", key, value)
        git(self.author, "checkout", "--quiet", "-b", "main")
        self.server_url = f"file://{root / 'server'}"

    def push_main(self) -> None:
        git(self.author, "push", "--quiet", "origin", "main")

    def push_pull_head(self) -> str:
        head = git(self.author, "rev-parse", "HEAD")
        git(self.author, "push", "--quiet", "--force", "origin", "HEAD:refs/pull/7/head")
        return head

    def clone_evaluator(self) -> None:
        subprocess.run(["git", "clone", "--quiet", str(self.remote), str(self.evaluator)],
                       check=True, capture_output=True)


CATALOG = "agents/model_catalog.py"
DOCS = "docs/agents.md"
TEST = "tests/agents/test_model_catalog.py"


def pr_380_shaped_repository(tmp_path: Path) -> tuple[CarryForwardRepository, str]:
    """Build PR #380's shape: approve a catalog PR, then main edits the same lines."""
    repository = CarryForwardRepository(tmp_path)
    commit_files(repository.author, "base", {
        CATALOG: "MODELS = [\n    'a',\n]\n\n\ndef route():\n    return 'a'\n",
        DOCS: "Models: A\n",
        TEST: "IDS = ['a']\n",
    })
    repository.push_main()
    git(repository.author, "checkout", "--quiet", "-b", "feature")
    approved = commit_files(repository.author, "add luna", {
        CATALOG: "MODELS = [\n    'a',\n    'luna',\n]\n\n\ndef route():\n    return 'a'\n",
        DOCS: "Models: Luna, A\n",
        TEST: "IDS = ['a', 'luna']\n",
    })
    git(repository.author, "checkout", "--quiet", "main")
    commit_files(repository.author, "add ember on main", {
        CATALOG: "MODELS = [\n    'a',\n]\n\n\ndef route():\n    return 'a'\n\n\nEMBER = 1\n",
        DOCS: "Models: Ember, A\n",
        TEST: "IDS = ['a', 'ember']\n",
    })
    repository.push_main()
    git(repository.author, "checkout", "--quiet", "feature")
    subprocess.run(["git", "-C", str(repository.author), "merge", "--quiet", "main"],
                   capture_output=True)
    commit_files(repository.author, "merge main", {
        DOCS: "Models: Luna, Ember, A\n",
        TEST: "IDS = ['a', 'luna', 'ember']\n",
    })
    return repository, approved


def changed_since(
    repository: CarryForwardRepository, approved: str, monkeypatch
) -> list[str]:
    head = repository.push_pull_head()
    if not repository.evaluator.exists():
        repository.clone_evaluator()
    monkeypatch.chdir(repository.evaluator)
    monkeypatch.setenv("GITHUB_SERVER_URL", repository.server_url)
    monkeypatch.delenv("PITCHBLEND_GITHUB_TOKEN", raising=False)
    pull_request = {"number": 7, "head": {"sha": head}, "base": {"ref": "main"}}
    return pitchblend.paths_changed_since_approval("org", "repo", pull_request, approved)


def test_pr_380_main_sync_with_docs_and_test_conflicts_carries_forward(
    tmp_path: Path, monkeypatch
) -> None:
    repository, approved = pr_380_shaped_repository(tmp_path)

    changed = changed_since(repository, approved, monkeypatch)

    assert changed == [DOCS, TEST]
    assert all(pitchblend.is_carry_forward_safe_path(path) for path in changed)


def test_clean_main_sync_reports_no_changes_since_approval(
    tmp_path: Path, monkeypatch
) -> None:
    repository = CarryForwardRepository(tmp_path)
    commit_files(repository.author, "base", {CATALOG: "MODELS = []\n", DOCS: "A\n"})
    repository.push_main()
    git(repository.author, "checkout", "--quiet", "-b", "feature")
    approved = commit_files(repository.author, "feature", {CATALOG: "MODELS = ['luna']\n"})
    git(repository.author, "checkout", "--quiet", "main")
    commit_files(repository.author, "main docs", {DOCS: "A and B\n"})
    repository.push_main()
    git(repository.author, "checkout", "--quiet", "feature")
    git(repository.author, "merge", "--quiet", "--no-edit", "main")

    assert changed_since(repository, approved, monkeypatch) == []

    git(repository.author, "reset", "--quiet", "--hard", approved)
    git(repository.author, "rebase", "--quiet", "main")
    assert changed_since(repository, approved, monkeypatch) == []


def test_code_edits_after_approval_do_not_carry_forward(
    tmp_path: Path, monkeypatch
) -> None:
    repository, approved = pr_380_shaped_repository(tmp_path)
    commit_files(repository.author, "change routing", {
        CATALOG: "MODELS = [\n    'a',\n    'luna',\n]\n\n\ndef route():\n"
                 "    return 'luna'\n\n\nEMBER = 1\n",
    })

    changed = changed_since(repository, approved, monkeypatch)

    assert changed == [CATALOG, DOCS, TEST]
    assert not pitchblend.is_carry_forward_safe_path(CATALOG)


def test_carry_forward_path_policy_keeps_protected_paths_blocked() -> None:
    assert pitchblend.is_carry_forward_safe_path("docs/agents.md")
    assert pitchblend.is_carry_forward_safe_path("plans/380-LUNA.md")
    assert pitchblend.is_carry_forward_safe_path("client/geist/src/App.test.tsx")
    assert not pitchblend.is_carry_forward_safe_path("tests/security/test_operator.py")
    assert not pitchblend.is_carry_forward_safe_path("docs/schemas/api.json")
    assert not pitchblend.is_carry_forward_safe_path("agents/online_agent.py")
    assert not pitchblend.is_carry_forward_safe_path("README.md")


def catalog_pull_request_files() -> list[dict]:
    return [
        {"filename": path, "status": "modified", "additions": 3, "deletions": 0,
         "patch": "@@ -1 +1 @@"}
        for path in (CATALOG, DOCS, TEST)
    ]


def carry_forward_pull_request(**overrides: object) -> dict:
    value = {
        "number": 7,
        "head": {"sha": "b" * 40},
        "base": {"ref": "main"},
    }
    value.update(overrides)
    return value


def test_carry_forward_requires_main_scope_gates_and_safe_changes() -> None:
    github = mock.Mock()
    github.paginate.return_value = catalog_pull_request_files()
    with mock.patch.object(
        pitchblend, "paths_changed_since_approval", return_value=[DOCS, TEST]
    ):
        assert pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(), APPROVED_SHA
        )
        assert not pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(base={"ref": "dev"}),
            APPROVED_SHA,
        )
        assert not pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(), "--upload-pack=x"
        )

    with mock.patch.object(
        pitchblend, "paths_changed_since_approval", return_value=[CATALOG]
    ):
        assert not pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(), APPROVED_SHA
        )

    github.paginate.return_value = catalog_pull_request_files() + [
        {"filename": "agents/base_agent.py", "status": "modified", "additions": 1,
         "deletions": 0, "patch": "@@ -1 +1 @@"}
    ]
    with mock.patch.object(pitchblend, "paths_changed_since_approval") as changed:
        assert not pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(), APPROVED_SHA
        )
    changed.assert_not_called()


def test_carry_forward_fails_closed_when_git_cannot_verify() -> None:
    github = mock.Mock()
    github.paginate.return_value = catalog_pull_request_files()
    with mock.patch.object(
        pitchblend,
        "paths_changed_since_approval",
        side_effect=pitchblend.ReviewError("git fetch failed: not our ref"),
    ):
        assert not pitchblend.approval_carries_forward(
            github, "org", "repo", carry_forward_pull_request(), APPROVED_SHA
        )


def test_run_git_keeps_installation_token_out_of_argv(monkeypatch) -> None:
    monkeypatch.setenv("PITCHBLEND_GITHUB_TOKEN", "secret-token")
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
    with mock.patch.object(pitchblend.subprocess, "run", return_value=completed) as run:
        assert pitchblend.run_git("rev-parse", "HEAD") == "ok"
    argv = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    assert "secret-token" not in " ".join(argv)
    assert environment["GIT_CONFIG_KEY_0"] == "http.extraheader"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


def test_publish_approval_gate_sets_one_pending_or_success_context() -> None:
    github = mock.Mock()
    pull_request = gate_pull_request()

    pitchblend.publish_approval_gate(github, "org", "repo", pull_request, None)
    pending_payload = github.request.call_args.args[2]
    assert pending_payload == {
        "state": "pending",
        "context": "Pitchblend approval gate",
        "description": "Waiting for Pitchblend or human approval",
        "target_url": pull_request["html_url"],
    }

    pitchblend.publish_approval_gate(
        github, "org", "repo", pull_request, "human:maintainer"
    )
    success_payload = github.request.call_args.args[2]
    assert success_payload["state"] == "success"
    assert success_payload["description"] == "Approved by maintainer"

    pitchblend.publish_approval_gate(
        github, "org", "repo", pull_request, f"pitchblend-carried:{APPROVED_SHA}"
    )
    carried_payload = github.request.call_args.args[2]
    assert carried_payload["state"] == "success"
    assert carried_payload["description"] == (
        f"Pitchblend approval of {APPROVED_SHA[:12]} still applies"
    )


def test_review_workflow_recomputes_gate_and_requests_status_write() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/pitchblend-review.yml"
    ).read_text(encoding="utf-8")
    assert "pull_request_review:" in workflow
    assert "types: [submitted, dismissed]" in workflow
    assert "pull_request_target:" in workflow
    assert "types: [opened, reopened, synchronize, ready_for_review]" in workflow
    assert "permission-statuses: write" in workflow
    assert "fetch-depth: 0" in workflow


def test_pull_request_review_event_recomputes_current_head_gate() -> None:
    event = {
        "repository": {"full_name": "org/repo"},
        "pull_request": {"number": 42},
    }
    pull_request = gate_pull_request()
    github = mock.Mock()
    github.request.side_effect = [pull_request, None]
    github.paginate.return_value = [review("pitchblend-ai[bot]", "Bot")]
    environment = {
        "GITHUB_EVENT_PATH": "/event.json",
        "GITHUB_EVENT_NAME": "pull_request_review",
        "PITCHBLEND_GITHUB_TOKEN": "token",
    }
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(event))),
        mock.patch.object(pitchblend, "JsonHttpClient", return_value=github),
    ):
        assert pitchblend.main() == 0

    status_call = github.request.call_args_list[-1]
    assert status_call.args[0] == "POST"
    assert status_call.args[1].endswith("/statuses/current-head")
    assert status_call.args[2]["state"] == "success"


def test_classifier_requires_every_non_frontend_low_risk_signal() -> None:
    classification = base_classification()

    assert pitchblend.classification_is_eligible(classification)

    classification["contract_change"] = True
    assert not pitchblend.classification_is_eligible(classification)


def test_copy_only_matrix_waives_tests_but_preserves_risk_boundaries() -> None:
    classification = base_classification(change_type="copy", regression_test_present=False)
    assert pitchblend.classification_pass_rule(classification) == "copy-only-low"
    for field, value in (("complexity", "medium"), ("data_migration", True),
                         ("contract_change", True), ("security_change", True)):
        assert pitchblend.classification_pass_rule({**classification, field: value}) is None
    body = pitchblend.format_approved_comment(363, "head", 1, 1, classification, "copy-only-low")
    assert "Regression tests are not required" in body
    assert "check, and regression-test gates passed" not in body


def test_review_copy_without_ci_queries_and_behavioral_fix_requires_tests() -> None:
    files = [{"filename": "client/geist/src/AppShell.tsx", "status": "modified",
              "additions": 0, "deletions": 1,
              "patch": '@@ -309 +309,0 @@\n-<p className="topbar-eyebrow">Runtime</p>'}]
    event = {"repository": {"full_name": "org/repo"},
             "issue": {"number": 363, "pull_request": {}},
             "comment": {"id": 123, "body": "@pitchblend-ai review", "author_association": "OWNER"}}
    event["issue"]["pull_request"] = {"url": "https://github.test/pr/363"}
    pull_request = {**eligible_pull_request(), **gate_pull_request()}
    for change_type in ("copy", "bugfix"):
        github = mock.Mock()
        github.paginate.side_effect = [[], files]
        def request(method, path, payload=None):
            if method == "GET":
                assert path == "/repos/org/repo/pulls/363", path
                return pull_request
            return None
        github.request.side_effect = request
        environment = {"GITHUB_EVENT_PATH": "/event.json", "GITHUB_EVENT_NAME": "issue_comment",
                       "PITCHBLEND_GITHUB_TOKEN": "token", "OPENAI_API_KEY": "key"}
        with (mock.patch.dict(os.environ, environment, clear=True),
              mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(event))),
              mock.patch.object(pitchblend, "JsonHttpClient", return_value=github),
              mock.patch.object(pitchblend, "refresh_approval_gate"),
              mock.patch.object(pitchblend, "classify_pull_request", return_value=base_classification(
                  change_type=change_type, is_frontend=True, regression_test_present=False))):
            assert pitchblend.main() == 0
        approvals = [call for call in github.request.call_args_list
                     if call.args[0] == "POST" and call.args[1].endswith("/reviews")]
        assert bool(approvals) == (change_type == "copy")
        if approvals:
            assert approvals[0].args[2]["commit_id"] == "current-head"
        else:
            assert any("no regression-test file" in str(call) for call in github.request.call_args_list)
