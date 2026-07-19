#!/usr/bin/env python3
"""Build the target-native Geist onedir sidecar with PyInstaller.

PyInstaller cannot cross-compile, so ``--target`` is an assertion about the
current host. The resulting ``geist`` directory is the complete unit consumed
by a native host or installer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "client" / "geist"
FRONTEND_BUILD = FRONTEND_ROOT / "build"
ENTRYPOINT = PROJECT_ROOT / "scripts" / "geist_frozen_entrypoint.py"
DEFAULT_SOURCE_DATE_EPOCH = "315532800"  # 1980-01-01, valid for ZIP timestamps.


@dataclass(frozen=True)
class NativeTarget:
    key: str
    system: str
    machine: str
    executable_name: str
    backend: str
    hidden_imports: tuple[str, ...]
    collect_all: tuple[str, ...] = ()
    excluded_modules: tuple[str, ...] = ()


TARGETS: dict[str, NativeTarget] = {
    "win32-x64": NativeTarget(
        key="win32-x64",
        system="Windows",
        machine="x86_64",
        executable_name="geist.exe",
        backend="llama-server",
        hidden_imports=(
            "agents.architectures.llama_server_process",
            "agents.architectures.llama_server_runner",
        ),
        excluded_modules=(
            "mlx",
            "mlx_lm",
            "agents.architectures.mlx_llama_runner",
            "agents.architectures.llama.llama_mlx",
            "agents.architectures.llama.mlx_lm_backend",
        ),
    ),
    "darwin-arm64": NativeTarget(
        key="darwin-arm64",
        system="Darwin",
        machine="arm64",
        executable_name="geist",
        backend="mlx",
        hidden_imports=(
            "agents.architectures.mlx_llama_runner",
            "agents.architectures.llama.llama_mlx",
            "agents.architectures.llama.mlx_lm_backend",
            # The models API imports lifecycle diagnostics on every platform,
            # even though macOS does not stage a llama-server executable.
            "agents.architectures.llama_server_process",
            "safetensors.numpy",
        ),
        collect_all=("mlx", "mlx_lm"),
        excluded_modules=("agents.architectures.llama_server_runner",),
    ),
    "linux-x64": NativeTarget(
        key="linux-x64",
        system="Linux",
        machine="x86_64",
        executable_name="geist",
        backend="llama-server",
        hidden_imports=(
            "agents.architectures.llama_server_process",
            "agents.architectures.llama_server_runner",
        ),
        excluded_modules=(
            "mlx",
            "mlx_lm",
            "agents.architectures.mlx_llama_runner",
            "agents.architectures.llama.llama_mlx",
            "agents.architectures.llama.mlx_lm_backend",
        ),
    ),
}

# Adapter discovery currently enumerates Python filenames before importing the
# matching modules. These portable adapters are included both as hidden imports
# and as data so that discovery keeps working inside a PyInstaller PYZ archive.
# The Torch-backed MMS adapter belongs to the separate voice distribution.
PORTABLE_ADAPTER_MODULES = (
    "adapters.adapter_registry",
    "adapters.async_tool",
    "adapters.base_adapter",
    "adapters.flux_image_adapter",
    "adapters.gemini_image_adapter",
    "adapters.image_gen_base",
    "adapters.image_generation_adapter",
    "adapters.inert_adapter",
    "adapters.job_status_adapter",
    "adapters.log_adapter",
    "adapters.markdown_file_adapter",
    "adapters.search_adapter",
    "adapters.sendgrid_adapter",
    "adapters.sms_adapter",
    "adapters.tool_modes",
    "adapters.tool_schema",
    "adapters.whisper_adapter",
)

COMMON_HIDDEN_IMPORTS = (
    "app.models.database",
    "multipart.multipart",
    "scripts.insert_presets",
    "sqlalchemy.dialects.sqlite",
) + PORTABLE_ADAPTER_MODULES

COMMON_COLLECT_SUBMODULES = (
    "alembic",
    "app.models.database",
    "uvicorn",
)

COMMON_EXCLUDED_MODULES = (
    "adapters.mms_adapter",
    "agents.architectures.qwen3_runner",
    "agents.architectures.transformers_runner",
    "agents.architectures.vllm_runner",
    "torch",
    "torchao",
    "torchaudio",
    "torchtune",
)


def normalize_machine(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    return normalized


def detect_target(system: str | None = None, machine: str | None = None) -> NativeTarget:
    actual_system = system or platform.system()
    actual_machine = normalize_machine(machine or platform.machine())
    for target in TARGETS.values():
        if target.system == actual_system and target.machine == actual_machine:
            return target
    supported = ", ".join(sorted(TARGETS))
    raise RuntimeError(
        f"Unsupported native freezer host {actual_system}/{actual_machine}. "
        f"Supported targets: {supported}. PyInstaller cannot cross-compile."
    )


def resolve_target(requested: str | None) -> NativeTarget:
    detected = detect_target()
    if requested is not None and requested != detected.key:
        raise RuntimeError(
            f"Cannot build {requested} on {detected.key}; PyInstaller requires a native host."
        )
    return detected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Geist and its compiled web app as a native onedir sidecar."
    )
    parser.add_argument(
        "--target",
        choices=tuple(TARGETS),
        help="assert the native target detected from this host",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="PyInstaller dist parent (default: dist/native/<target>)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="temporary PyInstaller work root (default: build/native-sidecar/<target>)",
    )
    parser.add_argument(
        "--skip-web-build",
        action="store_true",
        help="reuse an existing client/geist/build after verifying it",
    )
    parser.add_argument(
        "--codesign-identity",
        default=os.getenv("GEIST_CODESIGN_IDENTITY"),
        help="macOS signing identity passed to PyInstaller (or GEIST_CODESIGN_IDENTITY)",
    )
    return parser


def deterministic_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.setdefault("CI", "true")
    environment.setdefault("GENERATE_SOURCEMAP", "false")
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("SOURCE_DATE_EPOCH", DEFAULT_SOURCE_DATE_EPOCH)
    environment.setdefault("TZ", "UTC")
    return environment


def build_frontend(*, frontend_root: Path = FRONTEND_ROOT) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to compile the Geist web app")
    environment = deterministic_environment()
    _run(
        (npm, "ci", "--ignore-scripts", "--audit=false", "--fund=false"),
        frontend_root,
        environment,
    )
    _run((npm, "run", "build"), frontend_root, environment)


def verify_inputs(
    *,
    project_root: Path = PROJECT_ROOT,
    frontend_build: Path | None = None,
) -> None:
    web_build = frontend_build or project_root / "client" / "geist" / "build"
    required_files = (
        project_root / "alembic.ini",
        project_root / "LICENSE",
        project_root / "migrations" / "env.py",
        project_root / "migrations" / "script.py.mako",
        project_root / "pyproject.toml",
        project_root / "scripts" / "geist_frozen_entrypoint.py",
        project_root / "uv.lock",
        project_root / "client" / "geist" / "package-lock.json",
        web_build / "index.html",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    versions = project_root / "migrations" / "versions"
    if not any(path.name != "__init__.py" for path in versions.glob("*.py")):
        missing.append(f"{versions}{os.sep}*.py")
    for module in PORTABLE_ADAPTER_MODULES:
        source = project_root.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            missing.append(str(source))
    if missing:
        raise FileNotFoundError(
            "Native sidecar inputs are incomplete:\n- " + "\n- ".join(missing)
        )


def missing_freezer_modules(
    target: NativeTarget,
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    required = ["PyInstaller"]
    if target.key == "darwin-arm64":
        required.extend(("mlx", "mlx_lm", "transformers"))
    return tuple(module for module in required if find_spec(module) is None)


def pyinstaller_arguments(
    target: NativeTarget,
    *,
    project_root: Path = PROJECT_ROOT,
    dist_dir: Path,
    work_dir: Path,
    codesign_identity: str | None = None,
) -> list[str]:
    web_build = project_root / "client" / "geist" / "build"
    arguments = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--noupx",
        "--optimize",
        "1",
        "--name",
        "geist",
        "--contents-directory",
        "_internal",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir / "work"),
        "--specpath",
        str(work_dir / "spec"),
        "--paths",
        str(project_root),
        "--copy-metadata",
        "geist",
        "--add-data",
        _data_argument(web_build, "client/geist/build"),
        "--add-data",
        _data_argument(project_root / "migrations", "migrations"),
        "--add-data",
        _data_argument(project_root / "alembic.ini", "."),
        "--add-data",
        _data_argument(project_root / "LICENSE", "."),
    ]
    for module in COMMON_HIDDEN_IMPORTS + target.hidden_imports:
        arguments.extend(("--hidden-import", module))
    for package in COMMON_COLLECT_SUBMODULES:
        arguments.extend(("--collect-submodules", package))
    for package in target.collect_all:
        arguments.extend(("--collect-all", package))
    for module in COMMON_EXCLUDED_MODULES + target.excluded_modules:
        arguments.extend(("--exclude-module", module))
    for module in PORTABLE_ADAPTER_MODULES:
        source = project_root.joinpath(*module.split(".")).with_suffix(".py")
        arguments.extend(("--add-data", _data_argument(source, "adapters")))
    if target.key == "darwin-arm64":
        arguments.extend(("--target-architecture", "arm64"))
        if codesign_identity:
            arguments.extend(("--codesign-identity", codesign_identity))
    elif codesign_identity:
        raise ValueError("--codesign-identity is only valid for the macOS target")
    arguments.append(str(project_root / "scripts" / "geist_frozen_entrypoint.py"))
    return arguments


def verify_runtime_output(runtime_dir: Path, target: NativeTarget) -> Path:
    internal = runtime_dir / "_internal"
    executable = runtime_dir / target.executable_name
    required_files = (
        executable,
        internal / "alembic.ini",
        internal / "client" / "geist" / "build" / "index.html",
        internal / "migrations" / "env.py",
        internal / "migrations" / "script.py.mako",
        internal / "adapters" / "adapter_registry.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    versions = internal / "migrations" / "versions"
    if not any(path.name != "__init__.py" for path in versions.glob("*.py")):
        missing.append(f"{versions}{os.sep}*.py")
    if missing:
        raise FileNotFoundError(
            "Frozen Geist runtime is incomplete:\n- " + "\n- ".join(missing)
        )
    if executable.stat().st_size == 0:
        raise RuntimeError(f"Frozen Geist executable is empty: {executable}")
    return executable


def build_manifest(
    target: NativeTarget,
    *,
    project_root: Path = PROJECT_ROOT,
    python_version: str | None = None,
    pyinstaller_version: str | None = None,
) -> dict[str, object]:
    with (project_root / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    return {
        "schemaVersion": 1,
        "target": target.key,
        "backend": target.backend,
        "geistVersion": project["project"]["version"],
        "pythonVersion": python_version or platform.python_version(),
        "pyinstallerVersion": pyinstaller_version or importlib.metadata.version("pyinstaller"),
        "inputs": {
            "uvLockSha256": _sha256(project_root / "uv.lock"),
            "webPackageLockSha256": _sha256(
                project_root / "client" / "geist" / "package-lock.json"
            ),
        },
    }


def write_manifest(runtime_dir: Path, manifest: dict[str, object]) -> Path:
    destination = runtime_dir / "geist-runtime.json"
    contents = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    destination.write_text(contents, encoding="utf-8", newline="\n")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(
                f"Geist sidecars must be frozen with Python 3.11, not {platform.python_version()}"
            )
        target = resolve_target(args.target)
        if not args.skip_web_build:
            build_frontend()
        verify_inputs()
        missing_modules = missing_freezer_modules(target)
        if missing_modules:
            extras = (
                "packaged and local-mlx extras"
                if target.backend == "mlx"
                else "packaged extra"
            )
            raise RuntimeError(
                f"Missing freezer modules: {', '.join(missing_modules)}. "
                f"Run this script through the locked {extras}."
            )

        dist_dir = (
            args.dist_dir or PROJECT_ROOT / "dist" / "native" / target.key
        ).resolve()
        work_dir = (
            args.work_dir or PROJECT_ROOT / "build" / "native-sidecar" / target.key
        ).resolve()
        command = (
            sys.executable,
            "-m",
            "PyInstaller",
            *pyinstaller_arguments(
                target,
                dist_dir=dist_dir,
                work_dir=work_dir,
                codesign_identity=args.codesign_identity,
            ),
        )
        _run(command, PROJECT_ROOT, deterministic_environment())

        runtime_dir = dist_dir / "geist"
        verify_runtime_output(runtime_dir, target)
        manifest_path = write_manifest(runtime_dir, build_manifest(target))
        print(f"Geist native sidecar: {runtime_dir}")
        print(f"Build manifest: {manifest_path}")
        return 0
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"native-sidecar: {error}", file=sys.stderr)
        return 1


def _data_argument(source: Path, destination: str) -> str:
    return f"{source.resolve()}{os.pathsep}{destination}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], cwd: Path, environment: dict[str, str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
