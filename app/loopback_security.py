"""ASGI request checks for Geist's loopback-only native server."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send


_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class _Authority:
    host: str
    port: int | None


class LoopbackSecurityMiddleware:
    """Reject requests that cannot belong to the local Geist origin.

    Host validation prevents a public DNS name that resolves to 127.0.0.1 from
    reaching the service. Origin validation adds a browser-specific CSRF guard
    while continuing to permit CLI/API clients, which normally omit Origin.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        host_values = _header_values(scope, b"host")
        authority = _parse_authority(host_values[0]) if len(host_values) == 1 else None
        if authority is None or not _is_loopback_host(authority.host):
            await _reject(scope_type, send, status_code=400, reason="Invalid Host header")
            return

        origin_values = _header_values(scope, b"origin")
        validate_origin = scope_type == "websocket" or (
            scope_type == "http"
            and str(scope.get("method", "GET")).upper() not in _SAFE_HTTP_METHODS
        )
        if validate_origin and origin_values:
            origin_valid = len(origin_values) == 1 and _is_same_origin(
                origin_values[0], authority, str(scope.get("scheme", "http"))
            )
            if not origin_valid:
                await _reject(scope_type, send, status_code=403, reason="Invalid Origin header")
                return

        await self.app(scope, receive, send)


def install_loopback_security(app: FastAPI) -> None:
    """Install the loopback request boundary on a FastAPI application."""

    app.add_middleware(LoopbackSecurityMiddleware)


def _header_values(scope: Scope, name: bytes) -> list[str]:
    return [
        value.decode("latin-1").strip()
        for header_name, value in scope.get("headers", [])
        if header_name.lower() == name
    ]


def _parse_authority(value: str) -> _Authority | None:
    if not value or any(character in value for character in " /\\?#@,"):
        return None

    host: str
    port_text: str | None = None
    if value.startswith("["):
        closing_bracket = value.find("]")
        if closing_bracket < 0:
            return None
        host = value[1:closing_bracket]
        suffix = value[closing_bracket + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                return None
            port_text = suffix[1:]
    else:
        if value.count(":") > 1:
            # IPv6 authorities must use brackets.
            return None
        if ":" in value:
            host, port_text = value.rsplit(":", 1)
        else:
            host = value

    if not host or "%" in host:
        return None

    port: int | None = None
    if port_text is not None:
        if not port_text.isascii() or not port_text.isdigit():
            return None
        port = int(port_text)
        if not 1 <= port <= 65535:
            return None

    return _Authority(host=host.lower(), port=port)


def _is_loopback_host(host: str) -> bool:
    if host.rstrip(".") == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False

    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return False


def _is_same_origin(origin: str, host: _Authority, scope_scheme: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False

    if (
        parsed.scheme not in _DEFAULT_PORTS
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False

    origin_authority = _parse_authority(parsed.netloc)
    if origin_authority is None or not _is_loopback_host(origin_authority.host):
        return False

    expected_scheme = {"ws": "http", "wss": "https"}.get(scope_scheme, scope_scheme)
    if parsed.scheme != expected_scheme:
        return False

    origin_port = origin_authority.port or _DEFAULT_PORTS[parsed.scheme]
    host_port = host.port or _DEFAULT_PORTS.get(expected_scheme)
    return origin_authority.host.rstrip(".") == host.host.rstrip(".") and origin_port == host_port


async def _reject(
    scope_type: str,
    send: Send,
    *,
    status_code: int,
    reason: str,
) -> None:
    if scope_type == "websocket":
        await send({"type": "websocket.close", "code": 1008, "reason": reason})
        return

    body = reason.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-length", str(len(body)).encode("ascii")),
                (b"content-type", b"text/plain; charset=utf-8"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
