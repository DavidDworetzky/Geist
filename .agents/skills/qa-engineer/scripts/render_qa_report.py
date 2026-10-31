#!/usr/bin/env python3
"""Render a stable Markdown QA report from the independent result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OVERALL_RESULTS = {"PASS", "FAIL", "BLOCKED"}
LANE_RESULTS = OVERALL_RESULTS | {"NOT_APPLICABLE"}


def text(value: Any) -> str:
    return str(value if value is not None else "")


def table_cell(value: Any) -> str:
    return text(value).replace("|", "\\|").replace("\n", "<br>")


def validate_result(payload: dict[str, Any]) -> None:
    if payload.get("verdict") not in OVERALL_RESULTS:
        raise ValueError("verdict must be PASS, FAIL, or BLOCKED")
    pr = payload.get("pr")
    if not isinstance(pr, dict) or not pr.get("number") or not pr.get("head_sha"):
        raise ValueError("pr.number and pr.head_sha are required")
    for lane in payload.get("runtime_lanes", []):
        if lane.get("result") not in LANE_RESULTS:
            raise ValueError(f"invalid runtime result for {lane.get('name', 'unknown')}")
    for item in payload.get("traceability", []):
        if item.get("result") not in OVERALL_RESULTS:
            raise ValueError(f"invalid traceability result for rank {item.get('rank', '?')}")


def render_list(title: str, values: list[Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if values:
        lines.extend(f"- {text(value)}" for value in values)
    else:
        lines.append("- None")
    lines.append("")
    return lines


def render_report(payload: dict[str, Any]) -> str:
    validate_result(payload)
    pr = payload["pr"]
    artifacts = payload.get("artifacts", {})
    lines = [
        f"# QA Verdict: {payload['verdict']}",
        "",
        f"**PR:** [#{pr['number']}]({pr.get('url', '')})  ",
        f"**Head SHA:** `{pr['head_sha']}`  ",
        f"**Branch:** `{pr.get('branch', '')}`  ",
        f"**Summary:** {payload.get('summary', '')}",
        "",
    ]
    if artifacts.get("video_url"):
        lines.extend([f"**MP4 evidence:** [Open recording]({artifacts['video_url']})", ""])
    elif artifacts.get("video_path"):
        lines.extend([f"**Local MP4:** `{artifacts['video_path']}`", ""])

    lines.extend(
        [
            "## Risk-ranked traceability",
            "",
            "| Rank | Code path | Behavior and risk | Runtime lanes | Test | Result | Evidence |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for item in sorted(payload.get("traceability", []), key=lambda row: row.get("rank", 999)):
        behavior_risk = f"{item.get('behavior', '')} — {item.get('risk', '')}"
        lines.append(
            "| {rank} | {path} | {behavior} | {lanes} | {test} | {result} | {evidence} |".format(
                rank=table_cell(item.get("rank")),
                path=table_cell(item.get("code_path")),
                behavior=table_cell(behavior_risk),
                lanes=table_cell(", ".join(item.get("runtime_lanes", []))),
                test=table_cell(item.get("test")),
                result=table_cell(item.get("result")),
                evidence=table_cell(item.get("evidence")),
            )
        )
    lines.append("")

    lines.extend(
        [
            "## Runtime lanes",
            "",
            "| Lane | Result | Runner | Model | Evidence or reason |",
            "|---|---|---|---|---|",
        ]
    )
    for lane in payload.get("runtime_lanes", []):
        evidence = lane.get("evidence") or lane.get("reason") or ""
        lines.append(
            "| {name} | {result} | {runner} | {model} | {evidence} |".format(
                name=table_cell(lane.get("name")),
                result=table_cell(lane.get("result")),
                runner=table_cell(lane.get("runner")),
                model=table_cell(lane.get("model")),
                evidence=table_cell(evidence),
            )
        )
    lines.append("")

    lines.extend(["## Focused tests", ""])
    focused_tests = payload.get("focused_tests", [])
    if focused_tests:
        for test in focused_tests:
            lines.extend(
                [
                    f"- **{test.get('result', '')}:** `{test.get('command', '')}` — {test.get('evidence', '')}",
                ]
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(render_list("Failures", payload.get("failures", [])))
    lines.extend(render_list("Blockers", payload.get("blockers", [])))

    commands = [
        command for lane in payload.get("runtime_lanes", []) for command in lane.get("commands", [])
    ]
    lines.extend(["## Runtime commands", "", "```text"])
    lines.extend(commands or ["None"])
    lines.extend(["```", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = render_report(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
