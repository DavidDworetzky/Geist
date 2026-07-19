"""Runtime paths and environment configuration for native Geist processes."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
DEFAULT_APPLICATION_VERSION = "0.1.0"


def application_version() -> str:
    """Return installed package metadata, with a source-checkout fallback."""
    try:
        return importlib.metadata.version("geist")
    except importlib.metadata.PackageNotFoundError:
        return DEFAULT_APPLICATION_VERSION


def default_data_dir(
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return a stable per-user data directory without depending on the cwd."""
    environment = os.environ if environ is None else environ
    configured = environment.get("GEIST_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    platform_name = platform.system() if system is None else system
    user_home = Path.home() if home is None else home
    if platform_name == "Windows":
        base = Path(environment.get("LOCALAPPDATA", user_home / "AppData" / "Local"))
        return (base / "Geist").expanduser().resolve()
    if platform_name == "Darwin":
        return (user_home / "Library" / "Application Support" / "Geist").resolve()

    xdg_data_home = environment.get("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else user_home / ".local" / "share"
    return (base / "geist").resolve()


def discover_web_dir(
    configured: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path | None:
    """Locate a compiled React application, returning ``None`` when absent."""
    environment = os.environ if environ is None else environ
    candidate_value = configured or environment.get("GEIST_WEB_DIR")
    candidate = (
        Path(candidate_value).expanduser()
        if candidate_value
        else project_root / "client" / "geist" / "build"
    )
    candidate = candidate.resolve()
    return candidate if (candidate / "index.html").is_file() else None


@dataclass(frozen=True)
class RuntimePaths:
    data_dir: Path
    database_path: Path
    output_dir: Path
    model_cache_dir: Path
    web_dir: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        data_dir: str | Path | None = None,
        web_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        require_web: bool = False,
    ) -> RuntimePaths:
        environment = os.environ if environ is None else environ
        explicit_data_dir = data_dir is not None
        resolved_data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else default_data_dir(environ=environment)
        )

        configured_database = None if explicit_data_dir else environment.get("SQLITE_DATABASE_PATH")
        database_path = (
            Path(configured_database).expanduser().resolve()
            if configured_database
            else resolved_data_dir / "geist.sqlite3"
        )
        configured_output = None if explicit_data_dir else environment.get("IMAGE_GEN_OUTPUT_DIR")
        output_dir = (
            Path(configured_output).expanduser().resolve()
            if configured_output
            else resolved_data_dir / "output" / "images"
        )
        configured_model_cache = None if explicit_data_dir else environment.get("HF_HOME")
        model_cache_dir = (
            Path(configured_model_cache).expanduser().resolve()
            if configured_model_cache
            else resolved_data_dir / "models" / "huggingface"
        )

        requested_web_dir = web_dir or environment.get("GEIST_WEB_DIR")
        resolved_web_dir = discover_web_dir(
            requested_web_dir,
            environ=environment,
        )
        if require_web and resolved_web_dir is None:
            requested = (
                Path(requested_web_dir).expanduser().resolve()
                if requested_web_dir
                else PROJECT_ROOT / "client" / "geist" / "build"
            )
            raise FileNotFoundError(f"Compiled Geist web app not found at {requested}")

        return cls(
            data_dir=resolved_data_dir,
            database_path=database_path,
            output_dir=output_dir,
            model_cache_dir=model_cache_dir,
            web_dir=resolved_web_dir,
        )

    def apply(
        self,
        *,
        environ: MutableMapping[str, str] | None = None,
        create_directories: bool = True,
    ) -> None:
        """Apply paths before importing modules that create process-wide clients."""
        environment = os.environ if environ is None else environ
        environment["GEIST_DATA_DIR"] = str(self.data_dir)
        environment["SQLITE_DATABASE_PATH"] = str(self.database_path)
        environment["IMAGE_GEN_OUTPUT_DIR"] = str(self.output_dir)
        environment["HF_HOME"] = str(self.model_cache_dir)
        if self.web_dir is not None:
            environment["GEIST_WEB_DIR"] = str(self.web_dir)

        if create_directories:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)


def doctor_report(paths: RuntimePaths) -> dict[str, object]:
    """Build a dependency-light diagnostic report suitable for wrapper startup."""
    dependencies = {
        dependency: _module_available(dependency)
        for dependency in ("alembic", "fastapi", "multipart", "sqlalchemy", "uvicorn")
    }
    data_parent = _existing_parent(paths.data_dir)
    data_parent_writable = os.access(data_parent, os.W_OK)
    python_supported = sys.version_info[:2] == (3, 11)
    return {
        "ok": all(dependencies.values()) and data_parent_writable and python_supported,
        "version": application_version(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_supported,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "paths": {
            "dataDir": str(paths.data_dir),
            "databasePath": str(paths.database_path),
            "outputDir": str(paths.output_dir),
            "modelCacheDir": str(paths.model_cache_dir),
            "webDir": str(paths.web_dir) if paths.web_dir else None,
            "webAvailable": paths.web_dir is not None,
            "dataParentWritable": data_parent_writable,
        },
        "dependencies": dependencies,
    }


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
