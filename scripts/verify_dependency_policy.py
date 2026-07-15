#!/usr/bin/env python3
"""Check dependency manifests for basic supply-chain hardening rules."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "client" / "geist"
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
UNSAFE_NPM_SPEC_RE = re.compile(r"[\^~*<>|]|\blatest\b|x", re.IGNORECASE)
PINNED_PYTHON_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_.,-]+\])?"
    r"==[A-Za-z0-9][A-Za-z0-9.!+_-]*(?:\s*;\s*.+)?$"
)


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
            fail(
                f"{package_json.relative_to(ROOT)} must not define lifecycle script {lifecycle!r}",
                errors,
            )

    for package_path, metadata in lock.get("packages", {}).items():
        if package_path == "":
            continue
        resolved = metadata.get("resolved", "")
        if resolved.startswith("https://registry.npmjs.org/") and "integrity" not in metadata:
            fail(
                f"{package_lock.relative_to(ROOT)} package {package_path} is missing integrity",
                errors,
            )

    dockerfile = FRONTEND / "Dockerfile"
    docker_text = dockerfile.read_text(encoding="utf-8")
    if "RUN npm install" in docker_text:
        fail(f"{dockerfile.relative_to(ROOT)} must use npm ci, not npm install", errors)
    if "npm ci" in docker_text and "--ignore-scripts" not in docker_text:
        fail(f"{dockerfile.relative_to(ROOT)} npm ci must include --ignore-scripts", errors)


def check_python_dependency_group(group_name: str, dependencies: object, errors: list[str]) -> None:
    if not isinstance(dependencies, list):
        fail(f"pyproject.toml:{group_name} must be a dependency list", errors)
        return

    for dependency in dependencies:
        if not isinstance(dependency, str) or not PINNED_PYTHON_RE.fullmatch(dependency):
            fail(
                f"pyproject.toml:{group_name} dependency must use an exact == pin: {dependency!r}",
                errors,
            )


def check_backend(errors: list[str]) -> None:
    if not PYPROJECT.exists():
        fail("pyproject.toml is required", errors)
        return
    if not UV_LOCK.exists():
        fail("uv.lock is required", errors)
        return

    with PYPROJECT.open("rb") as file:
        project_data = tomllib.load(file)
    with UV_LOCK.open("rb") as file:
        lock_data = tomllib.load(file)

    project = project_data.get("project", {})
    check_python_dependency_group("project.dependencies", project.get("dependencies"), errors)

    optional_dependencies = project.get("optional-dependencies", {})
    for group_name, dependencies in optional_dependencies.items():
        check_python_dependency_group(
            f"project.optional-dependencies.{group_name}", dependencies, errors
        )

    for group_name, dependencies in project_data.get("dependency-groups", {}).items():
        check_python_dependency_group(f"dependency-groups.{group_name}", dependencies, errors)

    if not isinstance(lock_data.get("version"), int):
        fail("uv.lock must contain a lockfile version", errors)


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
