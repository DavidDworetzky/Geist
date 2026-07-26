"""Same-origin serving for the compiled Geist React application."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


PROTECTED_PREFIXES = (
    "/api",
    "/agent",
    "/adapter",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
    "/static",
)


def install_spa(app: FastAPI, web_dir: str | Path) -> Path:
    """Serve a compiled React build and preserve JSON 404s for backend routes."""
    root = Path(web_dir).expanduser().resolve()
    index_path = root / "index.html"
    if not index_path.is_file():
        raise FileNotFoundError(f"Compiled Geist web app not found at {root}")

    static_dir = root / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="geist-static")

    @app.get("/branding.json", include_in_schema=False)
    def branding():
        branding_path = root / "branding.json"
        if branding_path.is_file():
            return _file_response(branding_path)
        return JSONResponse({})

    @app.get("/", include_in_schema=False)
    def spa_root():
        return _index_response(index_path)

    @app.get("/{requested_path:path}", include_in_schema=False)
    def spa_fallback(requested_path: str):
        request_path = f"/{requested_path}"
        if _is_protected(request_path):
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = _safe_candidate(root, requested_path)
        if candidate is not None and candidate.is_file():
            return _file_response(candidate)
        return _index_response(index_path)

    return root


def _safe_candidate(root: Path, requested_path: str) -> Path | None:
    relative_path = PurePosixPath(requested_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    candidate = root.joinpath(*relative_path.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _is_protected(request_path: str) -> bool:
    return any(
        request_path == prefix or request_path.startswith(f"{prefix}/")
        for prefix in PROTECTED_PREFIXES
    )


def _index_response(index_path: Path) -> FileResponse:
    return FileResponse(index_path, headers={"Cache-Control": "no-cache"})


def _file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})
