#!/usr/bin/env python3
"""Collect immutable PR context and enforce the clean-branch QA preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def successful_output(arguments: list[str], cwd: Path) -> str:
    result = run_command(arguments, cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def parse_changed_files(diff_output: str) -> list[dict[str, Any]]:
    changed_files: list[dict[str, Any]] = []
    for line in diff_output.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        paths = fields[1:]
        changed_files.append(
            {
                "status": status,
                "path": paths[-1],
                "previous_path": paths[0] if len(paths) > 1 else None,
            }
        )
    return changed_files


def collect_context(repo: Path, pr_selector: str | None) -> dict[str, Any]:
    blockers: list[str] = []
    root = Path(successful_output(["git", "rev-parse", "--show-toplevel"], repo))
    branch = successful_output(["git", "branch", "--show-current"], root)
    current_head = successful_output(["git", "rev-parse", "HEAD"], root)

    status = successful_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    if status:
        blockers.append("The checked-out PR branch is not clean.")

    auth_result = run_command(["gh", "auth", "status", "--hostname", "github.com"], root)
    if auth_result.returncode != 0:
        blockers.append("GitHub CLI authentication for github.com is unavailable or invalid.")

    pr_command = [
        "gh",
        "pr",
        "view",
        *([pr_selector] if pr_selector else []),
        "--json",
        "number,url,state,baseRefName,baseRefOid,headRefName,headRefOid,isDraft",
    ]
    pr_result = run_command(pr_command, root)
    pr: dict[str, Any] | None = None
    if pr_result.returncode != 0:
        blockers.append("No open pull request could be resolved for the checked-out branch.")
    else:
        try:
            pr = json.loads(pr_result.stdout)
        except json.JSONDecodeError:
            blockers.append("GitHub CLI returned malformed pull-request metadata.")

    changed_files: list[dict[str, Any]] = []
    if pr:
        if pr.get("state") != "OPEN":
            blockers.append("The resolved pull request is not open.")
        if branch != pr.get("headRefName"):
            blockers.append("The checked-out branch does not match the pull-request head branch.")
        if current_head != pr.get("headRefOid"):
            blockers.append("The checked-out HEAD does not match the pushed pull-request head SHA.")

        base_sha = str(pr.get("baseRefOid") or "")
        head_sha = str(pr.get("headRefOid") or "")
        base_exists = run_command(["git", "cat-file", "-e", f"{base_sha}^{{commit}}"], root)
        if not base_sha or base_exists.returncode != 0:
            blockers.append("The pull-request base SHA is not present locally; fetch it before QA.")
        elif head_sha:
            diff = successful_output(
                ["git", "diff", "--name-status", "--find-renames", f"{base_sha}...{head_sha}"],
                root,
            )
            changed_files = parse_changed_files(diff)

    return {
        "status": "BLOCKED" if blockers else "READY",
        "blockers": blockers,
        "repo_root": str(root),
        "branch": branch,
        "current_head": current_head,
        "pr": pr,
        "changed_files": changed_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pr", help="Pull-request number, URL, or branch")
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        context = collect_context(args.repo.resolve(), args.pr)
    except (OSError, RuntimeError) as error:
        context = {"status": "BLOCKED", "blockers": [str(error)]}

    serialized = json.dumps(context, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0 if context.get("status") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
