#!/usr/bin/env python3
"""Scan staged files for high-signal secret patterns without printing values."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    label: str


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("aws access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b")),
    ("openai api key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic api key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("stripe live key", re.compile(r"\b(?:sk|pk)_live_[A-Za-z0-9]{16,}\b")),
)


def run_git(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], check=False, capture_output=True)


def staged_paths() -> list[str]:
    result = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        sys.exit(result.returncode)
    return [line for line in result.stdout.decode("utf-8", errors="replace").splitlines() if line]


def staged_text(path: str) -> str | None:
    result = run_git(["show", f":{path}"])
    if result.returncode != 0:
        return None
    if b"\0" in result.stdout:
        return None
    return result.stdout.decode("utf-8", errors="ignore")


def scan_file(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path=path, line=line_number, label=label))
    return findings


def main() -> int:
    findings: list[Finding] = []
    for path in staged_paths():
        text = staged_text(path)
        if text is None:
            continue
        findings.extend(scan_file(path, text))

    if not findings:
        return 0

    print("Potential secrets found in staged files:")
    for finding in findings:
        print(f"- {finding.path}:{finding.line} matched {finding.label}")
    print("Remove the secret from the commit or explicitly rotate and document the exception.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
