"""Create a private local operator token without replacing an existing one."""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path


def ensure_operator_token(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    token = f"geist_{secrets.token_urlsafe(48)}\n".encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Existing operator token path is not a nonempty file") from None
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(token)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: ensure_operator_token.py TOKEN_PATH")
    ensure_operator_token(Path(sys.argv[1]))
