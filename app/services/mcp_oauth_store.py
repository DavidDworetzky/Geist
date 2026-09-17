"""Private, atomic credential files, separate from the database and tool workspace."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.runtime_config import default_data_dir


class CredentialStoreError(RuntimeError):
    pass


def _directory() -> Path:
    if sys.platform == "win32" or os.name != "posix":
        raise CredentialStoreError("OAuth credential storage currently requires macOS or Linux")
    directory = default_data_dir() / "mcp-credentials"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise CredentialStoreError("OAuth credential directory must be owned by the runtime user")
    directory.chmod(0o700)
    return directory


def _path(credential_id: str) -> Path:
    if re.fullmatch(r"[a-f0-9]{32}", credential_id) is None:
        raise CredentialStoreError("Invalid OAuth credential reference")
    return _directory() / f"{credential_id}.json"


@contextmanager
def connection_lock(server_id: int, timeout_seconds: float = 30) -> Iterator[None]:
    if sys.platform == "win32":
        raise CredentialStoreError("OAuth credential storage currently requires macOS or Linux")
    directory = _directory()
    import fcntl

    if server_id <= 0:
        raise CredentialStoreError("OAuth requires a saved MCP server")
    lock_path = directory / f"server-{server_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CredentialStoreError("OAuth account is busy; try again") from None
                time.sleep(min(0.05, remaining))
        yield
    finally:
        os.close(descriptor)


def read_credentials(credential_id: str) -> dict:
    if sys.platform == "win32":
        raise CredentialStoreError("OAuth credential storage currently requires macOS or Linux")
    try:
        descriptor = os.open(_path(credential_id), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise CredentialStoreError("OAuth credential file permissions must be private")
            value = json.loads(handle.read(65537))
            if not isinstance(value, dict):
                raise ValueError()
            return value
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise CredentialStoreError(
            "Cannot read OAuth credentials; reconnect the account"
        ) from error


def write_credentials(credential_id: str, value: dict) -> None:
    destination = _path(credential_id)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".oauth-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def delete_credentials(credential_id: str) -> None:
    _path(credential_id).unlink(missing_ok=True)
