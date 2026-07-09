#!/usr/bin/env python3
"""Check dependency manifests for basic supply-chain hardening rules."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "client" / "geist"
ENV_FILES = [
    ROOT / "linux_environment.yml",
    ROOT / "linux_environment_x86_x64.yml",
    ROOT / "mac_environment_arm.yml",
]

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
UNSAFE_NPM_SPEC_RE = re.compile(r"[\^~*<>|]|\blatest\b|x", re.IGNORECASE)
PINNED_CONDA_RE = re.compile(r"^[A-Za-z0-9_.-]+=[^=\s]+(?:=[^=\s]+)?$")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def check_frontend(errors: list[str]) -> None:
    package_json = FRONTEND / "package.json"
    package_lock = FRONTEND / "package-lock.json"

    package = load_json(package_json)
    lock = load_json(package_lock)
    root_lock = lock.get("packages", {}).get("", {})

    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        deps = package.get(section, {})
        lock_deps = root_lock.get(section, {})
        for name, spec in deps.items():
            if UNSAFE_NPM_SPEC_RE.search(spec) or not SEMVER_RE.match(spec):
                fail(
                    f"{package_json.relative_to(ROOT)}:{section}.{name} must use an exact version, got {spec!r}",
                    errors,
                )
            if lock_deps.get(name) != spec:
                fail(
                    f"{package_lock.relative_to(ROOT)} root {section}.{name} must match package.json exactly",
                    errors,
                )

    scripts = package.get("scripts", {})
    for lifecycle in ("preinstall", "install", "postinstall", "prepare"):
        if lifecycle in scripts:
            fail(f"{package_json.relative_to(ROOT)} must not define lifecycle script {lifecycle!r}", errors)

    for package_path, metadata in lock.get("packages", {}).items():
        if package_path == "":
            continue
        resolved = metadata.get("resolved", "")
        if resolved.startswith("https://registry.npmjs.org/") and "integrity" not in metadata:
            fail(f"{package_lock.relative_to(ROOT)} package {package_path} is missing integrity", errors)

    dockerfile = FRONTEND / "Dockerfile"
    docker_text = dockerfile.read_text(encoding="utf-8")
    if "RUN npm install" in docker_text:
        fail(f"{dockerfile.relative_to(ROOT)} must use npm ci, not npm install", errors)
    if "npm ci" in docker_text and "--ignore-scripts" not in docker_text:
        fail(f"{dockerfile.relative_to(ROOT)} npm ci must include --ignore-scripts", errors)


def check_environment_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        return

    in_pip = False
    in_dependencies = False
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if raw_line.startswith("  - pip:"):
            in_pip = True
            continue
        if raw_line.startswith("  - ") and not raw_line.startswith("      - "):
            in_pip = False
            dep = raw_line.removeprefix("  - ").strip()
            if dep and not dep.startswith("pip") and not PINNED_CONDA_RE.match(dep):
                fail(f"{path.relative_to(ROOT)}:{line_no} conda dependency must be at least name=version: {dep}", errors)
        elif in_pip and raw_line.startswith("      - "):
            dep = raw_line.removeprefix("      - ").strip()
            if dep.startswith("--"):
                continue
            if "==" not in dep or any(token in dep for token in (">=", "<=", "~=", "!=", ">", "<")):
                fail(f"{path.relative_to(ROOT)}:{line_no} pip dependency must use an exact == pin: {dep}", errors)


def check_backend(errors: list[str]) -> None:
    for env_file in ENV_FILES:
        check_environment_file(env_file, errors)


def main() -> int:
    errors: list[str] = []
    check_frontend(errors)
    check_backend(errors)

    if errors:
        print("Dependency policy check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Dependency policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
