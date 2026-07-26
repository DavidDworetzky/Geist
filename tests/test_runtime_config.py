from pathlib import Path

import pytest

from app.runtime_config import RuntimePaths, default_data_dir, discover_web_dir


def test_default_data_dir_is_platform_specific_and_not_cwd(tmp_path):
    windows_base = tmp_path / "local"

    assert default_data_dir(
        environ={"LOCALAPPDATA": str(windows_base)}, system="Windows", home=tmp_path
    ) == (windows_base / "Geist").resolve()
    assert default_data_dir(environ={}, system="Darwin", home=tmp_path) == (
        tmp_path / "Library" / "Application Support" / "Geist"
    ).resolve()
    assert default_data_dir(environ={}, system="Linux", home=tmp_path) == (
        tmp_path / ".local" / "share" / "geist"
    ).resolve()


def test_explicit_data_dir_owns_all_default_runtime_storage(tmp_path):
    data_dir = tmp_path / "managed-data"
    paths = RuntimePaths.resolve(
        data_dir=data_dir,
        environ={
            "SQLITE_DATABASE_PATH": str(tmp_path / "old.sqlite3"),
            "IMAGE_GEN_OUTPUT_DIR": str(tmp_path / "old-output"),
            "HF_HOME": str(tmp_path / "old-models"),
        },
    )
    environment = {
        "SQLITE_DATABASE_PATH": str(tmp_path / "old.sqlite3"),
        "IMAGE_GEN_OUTPUT_DIR": str(tmp_path / "old-output"),
        "HF_HOME": str(tmp_path / "old-models"),
    }

    paths.apply(environ=environment)

    assert paths.database_path == data_dir / "geist.sqlite3"
    assert paths.output_dir == data_dir / "output" / "images"
    assert paths.model_cache_dir == data_dir / "models" / "huggingface"
    assert environment["GEIST_DATA_DIR"] == str(data_dir)
    assert environment["SQLITE_DATABASE_PATH"] == str(paths.database_path)
    assert environment["IMAGE_GEN_OUTPUT_DIR"] == str(paths.output_dir)
    assert environment["HF_HOME"] == str(paths.model_cache_dir)
    assert paths.output_dir.is_dir()
    assert paths.model_cache_dir.is_dir()


def test_discover_web_dir_requires_a_compiled_index(tmp_path):
    web_dir = tmp_path / "build"
    web_dir.mkdir()

    assert discover_web_dir(web_dir) is None

    (web_dir / "index.html").write_text("compiled", encoding="utf-8")
    assert discover_web_dir(web_dir) == web_dir.resolve()


def test_explicit_missing_web_dir_can_be_required(tmp_path):
    missing = tmp_path / "missing-build"

    with pytest.raises(FileNotFoundError, match="Compiled Geist web app not found"):
        RuntimePaths.resolve(web_dir=missing, require_web=True, environ={})


def test_environment_data_dir_is_resolved(tmp_path):
    configured = tmp_path / "configured"

    paths = RuntimePaths.resolve(environ={"GEIST_DATA_DIR": str(configured)})

    assert paths.data_dir == configured.resolve()
    assert isinstance(paths.data_dir, Path)
