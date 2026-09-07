from __future__ import annotations

from collections.abc import Collection

from fastapi import HTTPException
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.security.operator import authenticate_operator


DEFAULT_OPERATOR_AUTH_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/docs",
        "/docs/oauth2-redirect",
        "/openapi.json",
        "/redoc",
    }
)


class OperatorAuthenticationMiddleware:
    """Require a request principal across the HTTP and WebSocket surface."""

    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: Collection[str] = DEFAULT_OPERATOR_AUTH_EXEMPT_PATHS,
    ) -> None:
        self.app = app
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        try:
            authenticate_operator(HTTPConnection(scope))
        except HTTPException as error:
            if scope["type"] == "http":
                response = JSONResponse(
                    {"detail": error.detail},
                    status_code=error.status_code,
                    headers=error.headers,
                )
                await response(scope, receive, send)
            else:
                await _close_websocket(send, error)
            return

        await self.app(scope, receive, send)


async def _close_websocket(send: Send, error: HTTPException) -> None:
    code = 4401 if error.status_code == 401 else 4403 if error.status_code == 403 else 1011
    message: Message = {
        "type": "websocket.close",
        "code": code,
        "reason": str(error.detail),
    }
    await send(message)
