"""Contained, text-only workspace primitives for coding agents."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_FILENAMES = {".env", "id_dsa", "id_ed25519", "id_rsa"}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem"}
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_EDIT_FILE_BYTES = 5_000_000


class WorkspaceFileAdapter:
    """Read and mutate UTF-8 text files beneath one configured root."""

    def __init__(self, file_root: str = ".") -> None:
        self.file_root = Path(file_root).expanduser().resolve()
        if not self.file_root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {file_root}")

    def list_files(
        self,
        path: str = "",
        pattern: str = "*",
        limit: int = 200,
    ) -> list[str]:
        search_root = self._resolve(path or ".", require_file=False)
        if not search_root.is_dir():
            raise NotADirectoryError(f"Not a directory: {path or '.'}")

        files: list[str] = []
        for root, directories, filenames in os.walk(search_root, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in DEFAULT_IGNORED_DIRECTORIES
                and not (Path(root) / directory).is_symlink()
            )
            for filename in sorted(filenames):
                candidate = Path(root) / filename
                if candidate.is_symlink():
                    continue
                relative = candidate.relative_to(self.file_root).as_posix()
                if self._is_sensitive(relative) or not fnmatch.fnmatch(relative, pattern):
                    continue
                files.append(relative)
                if len(files) >= limit:
                    return files
        return files

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 100_000,
    ) -> dict[str, Any]:
        target = self._resolve(path, require_file=True)
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")

        selected: list[str] = []
        selected_end = start_line - 1
        character_count = 0
        with target.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if line_number < start_line:
                    continue
                if end_line is not None and line_number > end_line:
                    break
                remaining = max_chars - character_count
                if remaining <= 0:
                    break
                selected.append(line[:remaining])
                character_count += min(len(line), remaining)
                selected_end = line_number
                if len(line) > remaining:
                    break
        return {
            "path": self._relative(target),
            "start_line": start_line,
            "end_line": selected_end,
            "content": "".join(selected),
            "truncated": character_count >= max_chars,
        }

    def search_text(
        self,
        query: str,
        path: str = "",
        pattern: str = "*",
        case_sensitive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        flags = 0 if case_sensitive else re.IGNORECASE
        expression = re.compile(re.escape(query), flags)
        matches: list[dict[str, Any]] = []
        for relative in self.list_files(path=path, pattern=pattern, limit=10_000):
            target = self._resolve(relative, require_file=True)
            if target.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            try:
                with target.open(encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        match = expression.search(line)
                        if match is None:
                            continue
                        matches.append(
                            {
                                "path": relative,
                                "line": line_number,
                                "column": match.start() + 1,
                                "text": line.rstrip("\r\n")[:1000],
                            }
                        )
                        if len(matches) >= limit:
                            return matches
            except UnicodeDecodeError:
                continue
        return matches

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path, allow_missing=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        previous_mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            if previous_mode is not None:
                temporary_path.chmod(previous_mode)
            os.replace(temporary_path, target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return {"path": self._relative(target), "characters_written": len(content)}

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> dict[str, Any]:
        target = self._resolve(path, require_file=True)
        if target.stat().st_size > MAX_EDIT_FILE_BYTES:
            raise ValueError("file is too large for an exact workspace edit")
        content = target.read_text(encoding="utf-8")
        actual_replacements = content.count(old_text)
        if actual_replacements != expected_replacements:
            raise ValueError(
                f"Expected {expected_replacements} matching block(s), found "
                f"{actual_replacements}; file was not changed"
            )
        result = self.write_file(self._relative(target), content.replace(old_text, new_text))
        result["replacements"] = actual_replacements
        return result

    def _resolve(
        self,
        path: str,
        *,
        require_file: bool = False,
        allow_missing: bool = False,
    ) -> Path:
        candidate_path = Path(path)
        if candidate_path.is_absolute():
            raise ValueError("path must be relative to the workspace root")
        candidate = (self.file_root / candidate_path).resolve(strict=False)
        if os.path.commonpath([self.file_root, candidate]) != str(self.file_root):
            raise ValueError("path escapes the workspace root")
        relative = self._relative(candidate)
        if self._is_sensitive(relative):
            raise ValueError("sensitive credential files are not available to workspace tools")
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if require_file and not candidate.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return candidate

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.file_root).as_posix()

    @staticmethod
    def _is_sensitive(path: str) -> bool:
        for part in Path(path).parts:
            lowered = part.lower()
            if lowered in ALLOWED_ENV_TEMPLATES:
                continue
            if lowered in SENSITIVE_FILENAMES or lowered.startswith(".env."):
                return True
            if Path(lowered).suffix in SENSITIVE_SUFFIXES:
                return True
        return False
