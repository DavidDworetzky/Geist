from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.build_native_sidecar import (
    TARGETS,
    build_manifest,
    detect_target,
    deterministic_environment,
    missing_freezer_modules,
    pyinstaller_arguments,
    verify_inputs,
    verify_runtime_output,
    write_manifest,
)


def _option_values(arguments: list[str], option: str) -> list[str]:
    return [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == option
    ]


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    (
        ("Windows", "AMD64", "win32-x64"),
        ("Darwin", "aarch64", "darwin-arm64"),
        ("Linux", "x86_64", "linux-x64"),
    ),
)
def test_detect_target_normalizes_supported_architectures(
    system: str, machine: str, expected: str
) -> None:
    assert detect_target(system, machine).key == expected


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Windows", "arm64"), ("Darwin", "x86_64"), ("Linux", "riscv64")),
)
def test_detect_target_rejects_unsupported_hosts(system: str, machine: str) -> None:
    with pytest.raises(RuntimeError, match="cannot cross-compile"):
        detect_target(system, machine)


def test_windows_recipe_is_onedir_and_includes_llama_runner(tmp_path: Path) -> None:
    arguments = pyinstaller_arguments(
        TARGETS["win32-x64"],
        project_root=Path(__file__).resolve().parents[2],
        dist_dir=tmp_path / "dist",
        work_dir=tmp_path / "work",
    )

    assert "--onedir" in arguments
    assert _option_values(arguments, "--contents-directory") == ["_internal"]
    assert "agents.architectures.llama_server_runner" in _option_values(
        arguments, "--hidden-import"
    )
    assert "agents.architectures.mlx_llama_runner" in _option_values(
        arguments, "--exclude-module"
    )
    assert "agents.architectures.transformers_runner" in _option_values(
        arguments, "--exclude-module"
    )
    assert "torch" in _option_values(arguments, "--exclude-module")
    data = _option_values(arguments, "--add-data")
    assert any(value.endswith(f"{os.pathsep}client/geist/build") for value in data)
    assert any(value.endswith(f"{os.pathsep}migrations") for value in data)
    assert any(value.endswith(f"{os.pathsep}.") for value in data)


def test_macos_recipe_collects_mlx_and_excludes_llama_server(tmp_path: Path) -> None:
    arguments = pyinstaller_arguments(
        TARGETS["darwin-arm64"],
        project_root=Path(__file__).resolve().parents[2],
        dist_dir=tmp_path / "dist",
        work_dir=tmp_path / "work",
        codesign_identity="Developer ID Application: Example Company",
    )

    assert set(_option_values(arguments, "--collect-all")) == {"mlx", "mlx_lm"}
    assert "agents.architectures.mlx_llama_runner" in _option_values(
        arguments, "--hidden-import"
    )
    assert "agents.architectures.llama_server_process" in _option_values(
        arguments, "--hidden-import"
    )
    assert "agents.architectures.llama_server_process" not in _option_values(
        arguments, "--exclude-module"
    )
    assert "agents.architectures.llama_server_runner" in _option_values(
        arguments, "--exclude-module"
    )
    assert _option_values(arguments, "--target-architecture") == ["arm64"]
    assert _option_values(arguments, "--codesign-identity") == [
        "Developer ID Application: Example Company"
    ]


def test_codesign_identity_is_rejected_for_non_macos_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for the macOS target"):
        pyinstaller_arguments(
            TARGETS["linux-x64"],
            project_root=Path(__file__).resolve().parents[2],
            dist_dir=tmp_path / "dist",
            work_dir=tmp_path / "work",
            codesign_identity="unexpected",
        )


def test_macos_freezer_preflight_requires_mlx_extra() -> None:
    available = {"PyInstaller", "transformers"}

    missing = missing_freezer_modules(
        TARGETS["darwin-arm64"],
        find_spec=lambda module: object() if module in available else None,
    )

    assert missing == ("mlx", "mlx_lm")
    assert missing_freezer_modules(
        TARGETS["win32-x64"], find_spec=lambda _module: object()
    ) == ()


def test_verify_inputs_requires_spa_and_migrations(tmp_path: Path) -> None:
    (tmp_path / "migrations" / "versions").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "adapters").mkdir()

    with pytest.raises(FileNotFoundError) as error:
        verify_inputs(project_root=tmp_path)

    message = str(error.value)
    assert "alembic.ini" in message
    assert "index.html" in message
    assert f"versions{os.sep}*.py" in message


def test_verify_runtime_output_requires_complete_onedir_tree(tmp_path: Path) -> None:
    runtime = tmp_path / "geist"
    (runtime / "_internal" / "migrations" / "versions").mkdir(parents=True)
    (runtime / "geist.exe").write_bytes(b"executable")

    with pytest.raises(FileNotFoundError) as error:
        verify_runtime_output(runtime, TARGETS["win32-x64"])

    message = str(error.value)
    assert "client" in message and "index.html" in message
    assert "alembic.ini" in message
    assert f"versions{os.sep}*.py" in message


def test_build_manifest_records_locked_inputs_without_timestamps(tmp_path: Path) -> None:
    (tmp_path / "client" / "geist").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "geist"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_bytes(b"uv-lock")
    package_lock = tmp_path / "client" / "geist" / "package-lock.json"
    package_lock.write_bytes(b"web-lock")

    manifest = build_manifest(
        TARGETS["linux-x64"],
        project_root=tmp_path,
        python_version="3.11.9",
        pyinstaller_version="6.16.0",
    )

    assert manifest == {
        "schemaVersion": 1,
        "target": "linux-x64",
        "backend": "llama-server",
        "geistVersion": "1.2.3",
        "pythonVersion": "3.11.9",
        "pyinstallerVersion": "6.16.0",
        "inputs": {
            "uvLockSha256": hashlib.sha256(b"uv-lock").hexdigest(),
            "webPackageLockSha256": hashlib.sha256(b"web-lock").hexdigest(),
        },
    }
    output = tmp_path / "runtime"
    output.mkdir()
    first = write_manifest(output, manifest).read_bytes()
    second = write_manifest(output, manifest).read_bytes()
    assert first == second
    assert json.loads(first) == manifest


def test_deterministic_environment_preserves_explicit_overrides() -> None:
    environment = deterministic_environment({"SOURCE_DATE_EPOCH": "123", "PATH": "tools"})

    assert environment["SOURCE_DATE_EPOCH"] == "123"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["GENERATE_SOURCEMAP"] == "false"
    assert environment["PATH"] == "tools"
