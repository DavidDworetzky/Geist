#!/usr/bin/env python3
"""Upload QA assets with gh and post the evidence report to a pull request."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EVIDENCE_TAG = "qa-evidence"


def run_command(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def require_success(arguments: list[str], cwd: Path) -> str:
    result = run_command(arguments, cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not component:
        raise ValueError("asset name is empty after sanitization")
    return component


def resolve_asset_url(assets: list[dict[str, Any]], name: str) -> str:
    for asset in assets:
        if asset.get("name") == name:
            url = asset.get("url") or asset.get("apiUrl")
            if url:
                return str(url)
    raise RuntimeError(f"Uploaded release asset was not found: {name}")


def publish(pr_selector: str, video: Path, report: Path, repo: Path) -> dict[str, str]:
    root = Path(require_success(["git", "rev-parse", "--show-toplevel"], repo))
    if require_success(["git", "status", "--porcelain=v1", "--untracked-files=all"], root):
        raise RuntimeError("The checked-out PR branch is not clean.")
    require_success(["gh", "auth", "status", "--hostname", "github.com"], root)

    pr = json.loads(
        require_success(
            [
                "gh",
                "pr",
                "view",
                pr_selector,
                "--json",
                "number,url,state,headRefName,headRefOid",
            ],
            root,
        )
    )
    current_branch = require_success(["git", "branch", "--show-current"], root)
    current_head = require_success(["git", "rev-parse", "HEAD"], root)
    if pr.get("state") != "OPEN":
        raise RuntimeError("The pull request is not open.")
    if current_branch != pr.get("headRefName") or current_head != pr.get("headRefOid"):
        raise RuntimeError("The checked-out branch or HEAD no longer matches the pull request.")

    short_sha = str(pr["headRefOid"])[:12]
    prefix = safe_component(f"pr-{pr['number']}-{short_sha}")
    with tempfile.TemporaryDirectory(prefix="geist-qa-publish-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_asset = temp_dir / f"{prefix}-qa.mp4"
        report_asset = temp_dir / f"{prefix}-qa-report.md"
        shutil.copy2(video, video_asset)
        shutil.copy2(report, report_asset)

        release = run_command(["gh", "release", "view", EVIDENCE_TAG, "--json", "tagName"], root)
        if release.returncode != 0:
            require_success(
                [
                    "gh",
                    "release",
                    "create",
                    EVIDENCE_TAG,
                    "--draft",
                    "--target",
                    str(pr["headRefOid"]),
                    "--title",
                    "Geist QA evidence (draft)",
                    "--notes",
                    "Draft-only assets linked from pull-request QA reports.",
                ],
                root,
            )

        require_success(
            [
                "gh",
                "release",
                "upload",
                EVIDENCE_TAG,
                str(video_asset),
                str(report_asset),
                "--clobber",
            ],
            root,
        )
        release_payload = json.loads(
            require_success(["gh", "release", "view", EVIDENCE_TAG, "--json", "assets,url"], root)
        )
        video_url = resolve_asset_url(release_payload.get("assets", []), video_asset.name)
        report_url = resolve_asset_url(release_payload.get("assets", []), report_asset.name)

        report_text = report.read_text(encoding="utf-8").rstrip()
        comment = (
            f"<!-- geist-qa-engineer:{pr['headRefOid']} -->\n"
            f"{report_text}\n\n"
            f"## Published evidence\n\n"
            f"- [MP4 recording]({video_url})\n"
            f"- [Markdown report asset]({report_url})\n"
        )
        comment_path = temp_dir / "pr-comment.md"
        comment_path.write_text(comment, encoding="utf-8")
        comment_url = require_success(
            ["gh", "pr", "comment", str(pr["number"]), "--body-file", str(comment_path)],
            root,
        )

    return {
        "pr_url": str(pr["url"]),
        "comment_url": comment_url,
        "video_url": video_url,
        "report_url": report_url,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", required=True, help="Pull-request number, URL, or branch")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.resolve()
    report = args.report.resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise SystemExit("--video must point to an existing .mp4 file")
    if not report.is_file():
        raise SystemExit("--report must point to an existing Markdown file")
    report_text = report.read_text(encoding="utf-8")
    if not report_text.startswith("# QA Verdict: "):
        raise SystemExit("report does not begin with a QA verdict")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "pr": args.pr,
                    "video": str(video),
                    "report": str(report),
                    "release_tag": EVIDENCE_TAG,
                    "operations": ["gh release upload", "gh pr comment --body-file"],
                },
                indent=2,
            )
        )
        return 0

    try:
        result = publish(args.pr, video, report, args.repo.resolve())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"BLOCKED: {error}") from error
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
